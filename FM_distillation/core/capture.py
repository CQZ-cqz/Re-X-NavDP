"""Consolidated FM_distillation library module (entry points live in cli.py)."""

from rexnavdp import BASE, ROOT

"""Capture-only, sequential GPU-1 scene sweep. No labels or training.

Resume is at scene granularity: completed scenes are skipped; an interrupted
scene restarts in a new attempt directory. Partial attempts are never deleted
or silently included in the completed dataset.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from FM_distillation.core.dataset import code_hashes, dump


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_assignments(include_reserved=False):
    assignments = json.loads((BASE/"../FM_distillation/config/fm_scene_split.json").read_text())
    if include_reserved:
        assignments["train"] += assignments["reserved"]
        assignments["reserved"] = []
        assignments["schema"] = "x_navdp_fm_scene_split_full_v2"
    flat = sum([assignments[k] for k in ("train","validation","test","reserved")],[])
    if len(flat) != len(set(flat)):
        raise ValueError("overlapping scene assignments")
    return assignments


def inventory(scope, include_reserved=False):
    import numpy as np
    assignments = scene_assignments(include_reserved)
    rows = []
    for split in (["train"] if scope == "train" else ["train", "validation"]):
        for scene in assignments[split]:
            family = "cluttered_easy" if scene.startswith("easy_") else "cluttered_hard"
            directory = BASE/"data/scenes"/family/scene
            files = [directory/name for name in ("pointgoal_start_goal_pairs.npy", "pointgoal_start_pair_samples.npy")]
            pairs = next((p for p in files if p.is_file()), None)
            if pairs is None or not (directory/"occupancy.ply").is_file():
                raise FileNotFoundError(f"missing scene metadata: {directory}")
            count = len(np.load(pairs,mmap_mode="r",allow_pickle=False))
            if count < 1:
                raise ValueError(f"empty scene {scene}")
            config = BASE/"eval/config/eval_pointgoal"/f"humanoid_{'clutter_easy' if family=='cluttered_easy' else 'clutter_hard'}.yaml"
            rows.append(dict(scene=scene,split=split,episodes=count,pairs=str(pairs),
                             pairs_sha256=digest(pairs),config=str(config),config_sha256=digest(config)))
    return rows


def completed_metrics(run, expected):
    """Success OR failure counts as an executed episode; exit=0 alone is insufficient."""
    files = list((run/"evaluation").rglob("metric.csv"))
    if len(files) != 1:
        raise ValueError("expected exactly one episode metrics file")
    with files[0].open() as handle:
        rows = list(csv.DictReader(handle))
    ids = [int(r["episode_idx"]) for r in rows]
    if len(ids) != expected or set(ids) != set(range(expected)):
        raise ValueError(f"incomplete episode coverage: {len(set(ids))}/{expected}")
    return rows, files[0]


def index_observations(run, row, cancel=None):
    """CPU-only schema pass and immutable source index after scene completion."""
    from FM_distillation.core.fm_data import load_record, validate_observation, sha256
    metrics, metric_path = completed_metrics(run,row["episodes"])
    manifest = json.loads((run/"manifest.json").read_text())
    by_id = {int(m["episode_idx"]):m for m in metrics}
    counts, stuck, total = {}, 0, 0
    index_path = run/"observation_index.jsonl"
    with index_path.open("x") as index:
        for path in sorted((run/"observations").glob("*.npz")):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("collection cancelled during indexing")
            arrays, meta = load_record(path)
            validate_observation(arrays,meta)
            if (meta["scene"],meta["split"],meta["run_id"]) != (row["scene"],row["split"],manifest["run_id"]):
                raise ValueError("observation provenance mismatch")
            sample = int(meta["sample_idx"])
            if sample not in by_id:
                raise ValueError("observation belongs to an unfinished episode")
            counts[sample] = counts.get(sample,0)+1
            total += 1
            stuck += int(meta["stuck"])
            index.write(json.dumps(dict(file=path.name,sha256=sha256(path),episode_id=meta["episode_id"],
                sample_idx=sample,step=meta["step"],stuck=meta["stuck"],outcome=by_id[sample]))+"\n")
    if set(counts) != set(by_id):
        raise ValueError("some completed episodes have no recorded observations")
    return dict(episodes=len(by_id),observations=total,stuck_observations=stuck,
                metric_file=str(metric_path.relative_to(run)),metric_sha256=digest(metric_path),
                index_sha256=digest(index_path),scene=row["scene"],split=row["split"])


def stop_process(process):
    if process is not None and process.poll() is None:
        # Only signal the process group we created via start_new_session=True.
        try:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()


def check_space(root, reserve_gib):
    free = shutil.disk_usage(root).free/2**30
    if free < reserve_gib:
        raise RuntimeError(f"disk guard: only {free:.1f} GiB free; reserve={reserve_gib} GiB")
    return free


def capture_scene(root,row,args,env):
    cancel = getattr(args,"cancel",None)
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise RuntimeError("collection cancelled")
    check_cancel()
    parent = root/row["split"]/row["scene"]
    parent.mkdir(parents=True,exist_ok=True)
    number = 1
    while (parent/f"attempt_{number:03d}").exists():
        number += 1
    run = parent/f"attempt_{number:03d}"
    command = [sys.executable,str(BASE/"../FM_distillation/cli.py"),"dataset"]
    gpu_flags = ["--physical-gpu",str(getattr(args,"physical_gpu",0))]
    subprocess.run(command+["prepare","--run",str(run),"--checkpoint",args.checkpoint,
        "--config",row["config"],"--scene",row["scene"],"--split",row["split"],"--seed",str(args.seed),
        "--scene-assignments",str(root/"scene_assignments.json")]+gpu_flags,
        cwd=BASE,env=env,check=True)
    server = evaluator = None
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",args.port))
        with (run/"server.log").open("w") as server_log, (run/"evaluate.log").open("w") as eval_log:
            server = subprocess.Popen(command+["serve","--run",str(run),"--port",str(args.port)]+gpu_flags,
                cwd=BASE,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
            ready = False
            for _ in range(90):
                check_cancel()
                if server.poll() is not None:
                    raise RuntimeError("capture server exited; inspect server.log")
                if "Running on" in (run/"server.log").read_text(errors="replace"):
                    ready = True
                    break
                check_space(root,args.reserve_gib)
                time.sleep(2)
            if not ready:
                raise RuntimeError("capture server readiness timeout")
            evaluator = subprocess.Popen(command+["evaluate","--run",str(run),"--port",str(args.port),
                "--episodes",str(row["episodes"]),"--max-steps","0"]+gpu_flags,cwd=BASE,env=env,
                stdout=eval_log,stderr=subprocess.STDOUT,start_new_session=True)
            started = time.monotonic()
            while evaluator.poll() is None:
                check_cancel()
                if server.poll() is not None:
                    raise RuntimeError("capture server died while evaluating")
                check_space(root,args.reserve_gib)
                if time.monotonic()-started > args.scene_timeout_hours*3600:
                    raise RuntimeError("scene watchdog timeout; preserve attempt and investigate")
                time.sleep(3)
            if evaluator.returncode:
                raise RuntimeError(f"evaluator failed with exit {evaluator.returncode}")
        stop_process(server)
        report = index_observations(run,row,cancel)
        dump(run/"CAPTURE_COMPLETE.json",report)
        return run, report
    except BaseException as exc:
        if run.exists():
            dump(run/"CAPTURE_FAILED.json",dict(error=type(exc).__name__,message=str(exc)))
        raise
    finally:
        stop_process(evaluator)
        stop_process(server)


"""Adopt an existing capture collection: two GPUs, one environment per GPU.

No GPU jobs without --execute. Preserve legacy manifests/completed attempts.
Use a collection lock against the old controller and a lock per claimed scene.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import uuid

from FM_distillation.core.dataset import code_hashes, dump


@contextmanager
def exclusive(path):
    with Path(path).open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"lock occupied: {path}; stop the old collector first") from exc
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def check_policy_sources(original,current):
    # This is an explicit scheduler migration, not a blanket ignore-hash switch.
    allowed = {"../FM_distillation/core/dataset.py"}
    changed = [p for p in set(original)|set(current) if p not in allowed and original.get(p)!=current.get(p)]
    if changed:
        raise RuntimeError(f"non-scheduler source changes since original capture: {sorted(changed)}")


def completed_scene(root,row,plan):
    markers = sorted((root/row["split"]/row["scene"]).glob("attempt_*/CAPTURE_COMPLETE.json"))
    if len(markers)>1:
        raise RuntimeError(f"multiple completed attempts for {row['scene']}")
    if not markers:
        return None
    run = markers[0].parent
    report = json.loads(markers[0].read_text())
    _,metrics = completed_metrics(run,row["episodes"])
    if digest(metrics)!=report["metric_sha256"] or digest(run/"observation_index.jsonl")!=report["index_sha256"]:
        raise RuntimeError(f"completed scene index/metrics changed: {run}")
    meta = json.loads((run/"manifest.json").read_text())
    if (meta["teacher_sha256"],meta["seed"],meta["scene"],meta["split"]) != (
            plan["teacher_sha256"],plan["seed"],row["scene"],row["split"]):
        raise RuntimeError(f"completed scene provenance mismatch: {run}")
    if (report["scene"],report["split"],report["episodes"]) != (row["scene"],row["split"],row["episodes"]):
        raise RuntimeError("completed marker identity/count mismatch")
    return dict(run=str(run.relative_to(root)),**report)


def inspect_collection(root):
    plan = json.loads((root/"collection_manifest.json").read_text())
    if inventory(plan["scope"],plan["include_reserved"]) != plan["scenes"]:
        raise RuntimeError("scene metadata/config changed since collection began")
    if json.loads((root/"scene_assignments.json").read_text()) != scene_assignments(plan["include_reserved"]):
        raise RuntimeError("scene assignments changed")
    check_policy_sources(plan["source_sha256"],code_hashes())
    done,pending = [],[]
    for row in plan["scenes"]:
        record = completed_scene(root,row,plan)
        if record is None:
            pending.append(row)
        else:
            done.append(record)
    return plan,done,pending


def atomic_json(path,value):
    path = Path(path)
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def worker_env(gpu):
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER="PCI_BUS_ID",
               ISAAC_ACTIVE_GPU=str(gpu),ISAAC_PHYSICS_GPU="0",OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1",
               OMNI_KIT_ACCEPT_EULA="YES",OMNI_KIT_ALLOW_ROOT="1")
    env.pop("DISPLAY",None)
    acados = env.setdefault("ACADOS_SOURCE_DIR",os.path.expanduser("~/acados"))
    env["LD_LIBRARY_PATH"] = acados+"/lib:"+env.get("LD_LIBRARY_PATH","")
    return env

