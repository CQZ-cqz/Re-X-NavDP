"""Consolidated FM_distillation library module (entry points live in cli.py)."""

from rexnavdp import BASE, ROOT

CLI_STAGE = "collect-joint"  # overridden by the joint-train entry

"""GPU1 InternScenes train capture with shared-encoder RTC-off labeling.

Default is a read-only inventory. --execute is required to launch GPU jobs.
Use a separate smoke output before a full collection. Scene-level restart only.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import zipfile

from FM_distillation.src.dataset import code_hashes, dump, check_frozen, select_physical_gpu
from FM_distillation.src.storage import digest, read_json, atomic_json, writer_lock
from FM_distillation.src.capture import stop_process, check_space, index_observations


def joint_sources():
    return {name:digest(BASE/name) for name in ("../../FM_distillation/src/joint.py","../../FM_distillation/src/labeling.py")}


def inventory(scene_limit=0, episodes=0):
    import numpy as np
    from src.training.scene_assets import load_x_navdp_scene_layout
    root = (BASE/"data/scenes").resolve()
    split = read_json(root/"scene_split.json")
    loaded = load_x_navdp_scene_layout(str(root),split_file=str(root/"scene_split.json"),
                                      split="train",usd_variant="navigation")
    pools = [[r for r in loaded if r["scene_type"] == family] for family in ("home","commercial")]
    ordered = [pool[i] for i in range(max(map(len,pools),default=0)) for pool in pools if i<len(pool)]
    if scene_limit:
        ordered = ordered[:scene_limit]
    rows = []
    for row in ordered:
        scene,family = row["scene_name"],row["scene_type"]
        if scene not in split[family+"_train"] or scene in split.get(family+"_eval",[]):
            raise ValueError("train/eval split leakage")
        pairs = Path(row["pointgoal_path"])
        count = len(np.load(pairs,mmap_mode="r",allow_pickle=False))
        if count < 1:
            raise ValueError("empty episode list")
        config = BASE/"eval/config/eval_pointgoal"/f"humanoid_internscene_{family}.yaml"
        rows.append(dict(scene=scene,split="train",family=family,episodes=min(episodes,count) if episodes else count,
                         pairs=str(pairs),pairs_sha256=digest(pairs),config=str(config),config_sha256=digest(config),
                         usd_path=row["usd_path"],esdf_path=row["esdf_path"]))
    if not rows:
        raise ValueError("no available train scenes")
    return rows


def prepare(run,row,plan):
    import yaml
    cfg = yaml.safe_load(Path(row["config"]).read_text())
    run.mkdir(parents=True)
    for folder in ("observations","labels"):
        (run/folder).mkdir()
    dump(run/"selected_scene.json",{row["family"]+"_train":[row["scene"]],row["family"]+"_eval":[]})
    dump(run/"scene_assignments.json",dict(train=[r["scene"] for r in plan["scenes"]],validation=[],test=[],reserved=[]))
    cfg["environment"].update(scene_split_file=str(run/"selected_scene.json"),scene_split="train",
                               scene_index=0,num_envs=1,device="cuda:0")
    cfg.update(run_root_dir=str(run/"evaluation"),run_id="fm_joint_capture")
    (run/"eval_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    sources = code_hashes()
    meta = dict(schema="x_navdp_fm_v1",run_id=f"{run.name}:{uuid.uuid4().hex[:12]}",
        created_utc=datetime.now(timezone.utc).isoformat(),scene=row["scene"],split="train",
        checkpoint=plan["checkpoint"],teacher_sha256=plan["teacher_sha256"],seed=plan["seed"],
        physical_gpu=plan["physical_gpu"],code_sha256=sources,joint_source_sha256=joint_sources(),
        config_sha256={name:digest(run/name) for name in ("eval_config.yaml","selected_scene.json","scene_assignments.json")},
        audit_every=plan["audit_every"],online=dict(rtc=True,sampler="ddpm",steps=10,recovery="baseline"),
        label=dict(rtc=False,candidates=8,sampler="ddpm",steps=10,joint_label_version=1),
        limits=["synchronous labeling increases response latency; not timing-equivalent to baseline",
                "joint labels have explicit sampled-audit provenance; legacy label loader rejects non-audited rows"])
    with zipfile.ZipFile(run/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name in {**sources,**joint_sources()}:
            archive.write(BASE/name,name)
    dump(run/"manifest.json",meta)
    return meta


def serve(args):
    gpu_uuid = select_physical_gpu(args.physical_gpu)
    run = Path(args.run).resolve()
    meta = check_frozen(run)
    if meta["physical_gpu"] != args.physical_gpu or meta["joint_source_sha256"] != joint_sources():
        raise ValueError("GPU or joint source mismatch")
    if any((run/"observations").iterdir()) or any((run/"labels").iterdir()):
        raise ValueError("do not restart a partial attempt")
    from FM_distillation.src.labeling import joint_agent_class
    from eval.src import policy_server
    dump(run/"capture_runtime.json",dict(physical_gpu=args.physical_gpu,gpu_uuid=gpu_uuid,joint=True))
    policy_server.NavDP_Agent = joint_agent_class(run,meta,meta["audit_every"])
    policy_server.init_app("humanoid",no_visualization=True,device="cuda:0",checkpoint=meta["checkpoint"],
                           seed=meta["seed"],recovery="baseline",rtc_enabled=True)
    policy_server.run_server(port=args.port)


def finalize(run,row):
    from FM_distillation.src.fm_data import load_record
    from FM_distillation.src.labeling import validate_joint_label
    report = index_observations(run,row)
    records = [json.loads(line) for line in (run/"observation_index.jsonl").read_text().splitlines()]
    hashes, audits = {},0
    meta = read_json(run/"manifest.json")
    if {p.name for p in (run/"labels").glob("*.npz")} != {r["file"] for r in records}:
        raise ValueError("observations/labels not one-to-one")
    for record in records:
        path = run/"labels"/record["file"]
        arrays,label = load_record(path)
        validate_joint_label(arrays,label)
        if (label["observation_sha256"] != record["sha256"] or label["teacher_sha256"] != meta["teacher_sha256"]
                or label["run_id"] != meta["run_id"] or label["scene"] != row["scene"] or label["split"] != "train"
                or label["step"] != record["step"] or label["episode_id"] != record["episode_id"]):
            raise ValueError("joint label provenance mismatch")
        audits += label["verification"] == "full"
        hashes[path.name] = digest(path)
    if audits == 0:
        raise ValueError("no full audits recorded")
    dump(run/"labels/COMPLETE.json",dict(status="complete",joint_label_version=1,count=len(records),
                                         full_audits=audits,sha256=hashes,teacher_sha256=meta["teacher_sha256"]))
    report.update(labels=len(records),full_audits=audits,label_complete_sha256=digest(run/"labels/COMPLETE.json"))
    dump(run/"CAPTURE_COMPLETE.json",report)
    return report


def capture(root,row,plan,args):
    from FM_distillation.src.capture import worker_env
    parent = root/"train"/row["scene"]
    number = 1
    while (parent/f"attempt_{number:03d}").exists():
        number += 1
    run = parent/f"attempt_{number:03d}"
    prepare(run,row,plan)
    env = worker_env(args.physical_gpu)
    env["X_NAVDP_MPC_CODEGEN_DIR"] = str(run/"mpc_codegen")
    command = [sys.executable,str(BASE/"../../FM_distillation/cli.py"),CLI_STAGE]
    shared = ["--run",str(run),"--physical-gpu",str(args.physical_gpu),"--port",str(args.port)]
    server = evaluator = None
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",args.port))
        with (run/"server.log").open("w") as slog, (run/"evaluate.log").open("w") as elog:
            server = subprocess.Popen(command+["--worker","serve"]+shared,cwd=BASE,env=env,
                                      stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(90):
                if server.poll() is not None:
                    raise RuntimeError("server failed; inspect server.log")
                if "Running on" in (run/"server.log").read_text(errors="replace"):
                    break
                check_space(root,args.reserve_gib)
                time.sleep(2)
            else:
                raise RuntimeError("server readiness timeout")
            evaluator = subprocess.Popen(command+["--worker","evaluate","--episodes",str(row["episodes"])]+shared,
                cwd=BASE,env=env,stdout=elog,stderr=subprocess.STDOUT,start_new_session=True)
            started = time.monotonic()
            while evaluator.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("server died")
                check_space(root,args.reserve_gib)
                if time.monotonic()-started > args.scene_timeout_hours*3600:
                    raise RuntimeError("scene timeout")
                time.sleep(3)
            if evaluator.returncode:
                raise RuntimeError("evaluation failed; inspect evaluate.log")
        stop_process(server)
        return finalize(run,row)
    except BaseException as exc:
        dump(run/"CAPTURE_FAILED.json",dict(error=type(exc).__name__,message=str(exc)))
        raise
    finally:
        stop_process(evaluator)
        stop_process(server)


"""Public entry for joint capture: explicitly bind safe TRAIN episode pairs.

The generic evaluator always selects eval pairs; this scoped adapter overrides
that lookup inside the evaluation child only. No teacher/training code is edited.
"""

from pathlib import Path
import sys


from FM_distillation.src import dataset as fm_dataset
from FM_distillation.src.storage import atomic_json, digest, read_json


def resolve_training_pairs(meta, directory):
    pairs = Path(meta["pointgoal_path"]).resolve()
    if Path(directory).resolve() != pairs.parent:
        raise ValueError("unexpected scene requested training pairs")
    if digest(pairs) != meta["pointgoal_sha256"]:
        raise ValueError("training episode pairs changed")
    return str(pairs)

