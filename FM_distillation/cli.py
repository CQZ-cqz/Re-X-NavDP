"""Re-X-NavDP Flow Matching distillation — unified command line interface.

Replaces the former per-workflow scripts under ``FM_distillation/scripts/``.
Every workflow is reachable as ``python -m FM_distillation.cli <stage> ...``;
the library logic lives in ``FM_distillation.src``.
"""

# Bootstrap so ``python FM_distillation/cli.py`` works from any cwd, before the
# ``rexnavdp`` package (and vendored ``x-navdp`` modules) can be imported.
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_BASE = _ROOT / "baselines/x-navdp"
for _p in (_ROOT, _BASE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import queue as _queue
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rexnavdp import BASE, ROOT
from FM_distillation.src import (capture, dataset, evaluation, fm_dual, joint,
                                  labeling, merging, rtc, scheduling, storage, training)

CLI_PATH = _Path(__file__).resolve()


# --------------------------------------------------------------------------- #
# dataset (fm_dataset): prepare / serve / evaluate / label / validate
# --------------------------------------------------------------------------- #
def cmd_dataset():
    parser = argparse.ArgumentParser(description="P0/P1 teacher dataset pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "serve", "evaluate", "label", "validate"):
        p = sub.add_parser(command)
        p.add_argument("--run", required=True)
        if command in ("prepare", "serve", "evaluate"):
            p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
        if command == "prepare":
            p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/x-navdp_posttrain.ckpt"))
            p.add_argument("--config", default=str(BASE / "eval/config/eval_pointgoal/humanoid_clutter_easy.yaml"))
            p.add_argument("--scene", default="easy_0")
            p.add_argument("--split", choices=("train", "validation", "test", "debug"), default="train")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--scene-assignments", help="explicit versioned scene assignment JSON")
        if command in ("serve", "evaluate"):
            p.add_argument("--port", type=int, default=20015)
        if command == "evaluate":
            p.add_argument("--episodes", type=int, default=1, help="0 = all available episode pairs")
            p.add_argument("--max-steps", type=int, default=0)
        if command in ("label", "validate"):
            p.add_argument("--name", default="labels_smoke")
        if command == "label":
            p.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    if hasattr(args, "limit") and args.limit < 1:
        parser.error("limit must be positive")
    if hasattr(args, "episodes") and (args.episodes < 0 or args.max_steps < 0):
        parser.error("episodes and max-steps must be nonnegative; 0 means no cap")
    if hasattr(args, "name") and (Path(args.name).name != args.name or args.name in (".", "..", "observations")):
        parser.error("name must be a new simple label directory name")
    for name in ("run", "checkpoint", "config", "scene_assignments"):
        if getattr(args, name, None) is not None:
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    os.chdir(BASE)
    getattr(dataset, args.command)(args)


# --------------------------------------------------------------------------- #
# collect-full (collect_fm_full)
# --------------------------------------------------------------------------- #
def cmd_collect_full():
    from FM_distillation.src.fm_data import sha256
    p = argparse.ArgumentParser(description="Capture-only sequential GPU-1 scene sweep.")
    p.add_argument("--output", required=True, help="explicit disk location, preferably a data volume")
    p.add_argument("--scope", choices=("train", "train-validation"), default="train-validation")
    p.add_argument("--include-reserved", action="store_true")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/x-navdp_posttrain.ckpt"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--reserve-gib", type=float, default=30.)
    p.add_argument("--scene-timeout-hours", type=float, default=12.)
    p.add_argument("--episodes", type=int, default=0, help="Cap episodes per scene (0 = all available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=20015)
    args = p.parse_args()
    if args.reserve_gib <= 0 or args.scene_timeout_hours <= 0 or args.episodes < 0:
        p.error("reserve and timeout must be positive; episodes must be nonnegative")
    # Sequential capture is pinned to physical GPU 1 (matches the worker env below).
    args.physical_gpu = 1
    root = Path(args.output).resolve()
    args.checkpoint = str(Path(args.checkpoint).resolve(strict=True))
    rows = capture.inventory(args.scope, args.include_reserved)
    if args.episodes:
        for row in rows:
            row["episodes"] = min(row["episodes"], args.episodes)
    existing = root
    while not existing.exists():
        existing = existing.parent
    print(json.dumps(dict(scenes=rows, total_episodes=sum(r["episodes"] for r in rows),
                         output=str(root), free_gib=shutil.disk_usage(existing).free / 2**30,
                         physical_gpu=1, capture_only=True), indent=2), flush=True)
    if not args.execute:
        print("Inventory only; no directories, GPU processes, labels or training started.")
        return
    capture.check_space(existing, args.reserve_gib)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assignments = capture.scene_assignments(args.include_reserved)
        plan = dict(scope=args.scope, include_reserved=args.include_reserved, scenes=rows,
                    checkpoint=args.checkpoint, teacher_sha256=sha256(args.checkpoint),
                    seed=args.seed, physical_gpu=1, source_sha256=dataset.code_hashes(),
                    collector_sha256=capture.digest(CLI_PATH),
                    split_sha256=capture.digest(BASE / "../../FM_distillation/config/fm_scene_split.json"))
        if (root / "collection_manifest.json").exists():
            if json.loads((root / "collection_manifest.json").read_text()) != plan:
                raise RuntimeError("collection configuration/code changed; use new output root")
            if json.loads((root / "scene_assignments.json").read_text()) != assignments:
                raise RuntimeError("collection scene assignments changed")
        else:
            if any(x.name != "collection.lock" for x in root.iterdir()):
                raise RuntimeError("output is not an empty new collection root")
            dataset.dump(root / "scene_assignments.json", assignments)
            dataset.dump(root / "collection_manifest.json", plan)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   ISAAC_ACTIVE_GPU="1", ISAAC_PHYSICS_GPU="0", OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                   OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_ALLOW_ROOT="1")
        env.pop("DISPLAY", None)
        acados = env.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))
        env["LD_LIBRARY_PATH"] = acados + "/lib:" + env.get("LD_LIBRARY_PATH", "")
        completed = []
        for row in rows:
            parent = root / row["split"] / row["scene"]
            markers = sorted(parent.glob("attempt_*/CAPTURE_COMPLETE.json"))
            if len(markers) > 1:
                raise RuntimeError("multiple completed attempts: explicit deduplication required")
            if markers:
                run = markers[0].parent
                report = json.loads(markers[0].read_text())
                _, metrics = capture.completed_metrics(run, row["episodes"])
                if report["metric_sha256"] != capture.digest(metrics) or \
                        report["index_sha256"] != capture.digest(run / "observation_index.jsonl"):
                    raise RuntimeError("completed capture index/metrics changed")
                print(f"SKIP completed {row['scene']}", flush=True)
            else:
                capture.check_space(root, args.reserve_gib)
                print(f"CAPTURE {row['scene']} ({row['episodes']} episodes)", flush=True)
                run, report = capture.capture_scene(root, row, args, env)
            completed.append(dict(run=str(run.relative_to(root)), **report))
            (root / "progress.json").write_text(json.dumps(completed, indent=2))
        (root / "collection_complete.json").write_text(json.dumps(dict(capture_only=True,
            scenes=completed, episodes=sum(r["episodes"] for r in completed)), indent=2))
        print("Capture completed. Review indices/episode outcomes before offline labelling.")


# --------------------------------------------------------------------------- #
# collect-dual (collect_fm_dual)
# --------------------------------------------------------------------------- #
def cmd_collect_dual():
    from FM_distillation.src.fm_data import sha256
    p = argparse.ArgumentParser(description="Adopt an existing capture collection across two GPUs.")
    p.add_argument("--output", required=True, help="existing single-GPU collection root")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gpus", type=int, nargs="+", choices=(0, 1), default=[1])
    p.add_argument("--only-split", choices=("train", "validation"))
    p.add_argument("--port-base", type=int, default=20015)
    p.add_argument("--reserve-gib", type=float, default=30.)
    p.add_argument("--scene-timeout-hours", type=float, default=12.)
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        p.error("duplicate GPU selection")
    if args.reserve_gib <= 0 or args.scene_timeout_hours <= 0 or not 1024 <= args.port_base < 65535:
        p.error("invalid reserve/timeout/port")
    root = Path(args.output).resolve(strict=True)
    plan, done, pending = capture.inspect_collection(root)
    pending = [row for row in pending if args.only_split is None or row["split"] == args.only_split]
    print(json.dumps(dict(completed_scenes=[x["scene"] for x in done],
        pending_scenes=[x["scene"] for x in pending], completed_episodes=sum(x["episodes"] for x in done),
        pending_episodes=sum(x["episodes"] for x in pending), gpus=args.gpus, num_envs_per_gpu=1,
        ports=[args.port_base + gpu for gpu in args.gpus]), indent=2), flush=True)
    if not args.execute:
        print("Inspection only. No files changed or jobs started. Stop old collector before --execute.")
        return
    with capture.exclusive(root / "collection.lock"):
        plan, done, pending = capture.inspect_collection(root)
        pending = [row for row in pending if args.only_split is None or row["split"] == args.only_split]
        if not pending:
            print("Nothing left to capture.")
            return
        if sha256(plan["checkpoint"]) != plan["teacher_sha256"]:
            raise RuntimeError("teacher checkpoint changed")
        for port in (args.port_base + gpu for gpu in args.gpus):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        capture.check_space(root, args.reserve_gib)
        session = root / "dual_sessions" / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        session.mkdir(parents=True)
        dataset.dump(session / "session.json", dict(legacy_manifest_sha256=capture.digest(root / "collection_manifest.json"),
            source_sha256=capture.code_hashes(), runner_sha256=capture.digest(CLI_PATH),
            capture_helper_sha256=capture.digest(BASE / "../../FM_distillation/src/capture.py"),
            original_teacher_sha256=plan["teacher_sha256"], gpus=args.gpus, num_envs_per_gpu=1,
            ports=[args.port_base + gpu for gpu in args.gpus], args=vars(args),
            note="scheduler migration; legacy manifest retained; each new attempt snapshots current sources"))
        jobs = _queue.Queue()
        for row in pending:
            jobs.put(row)
        cancel = threading.Event()
        progress_lock = threading.Lock()
        (root / "scene_locks").mkdir(exist_ok=True)

        def worker(gpu):
            env = capture.worker_env(gpu)
            env["X_NAVDP_MPC_CODEGEN_DIR"] = str(session / f"gpu{gpu}_mpc_codegen")
            local = argparse.Namespace(checkpoint=plan["checkpoint"], seed=plan["seed"],
                physical_gpu=gpu, port=args.port_base + gpu, reserve_gib=args.reserve_gib,
                scene_timeout_hours=args.scene_timeout_hours, cancel=cancel)
            try:
                while not cancel.is_set():
                    try:
                        row = jobs.get_nowait()
                    except _queue.Empty:
                        return
                    lock_path = root / "scene_locks" / f"{row['split']}_{row['scene']}.lock"
                    with capture.exclusive(lock_path):
                        if capture.completed_scene(root, row, plan) is not None:
                            raise RuntimeError("scene was completed by an unexpected concurrent writer")
                        capture.check_space(root, args.reserve_gib)
                        capture.atomic_json(session / f"gpu{gpu}.json", dict(state="capturing", scene=row["scene"], port=local.port))
                        print(f"[GPU {gpu}] CAPTURE {row['scene']} port={local.port}", flush=True)
                        run, report = capture.capture_scene(root, row, local, env)
                        with progress_lock:
                            done.append(dict(run=str(run.relative_to(root)), **report))
                            capture.atomic_json(root / "progress.json", done)
                        print(f"[GPU {gpu}] COMPLETE {row['scene']}", flush=True)
                        capture.atomic_json(session / f"gpu{gpu}.json", dict(state="scene_complete", scene=row["scene"]))
            except BaseException:
                cancel.set()
                raise

        expected_scenes = {row["scene"] for row in done + pending}
        pool = ThreadPoolExecutor(max_workers=len(args.gpus))
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        try:
            remaining = set(futures)
            while remaining:
                ready, remaining = wait(remaining, timeout=2, return_when=FIRST_COMPLETED)
                for future in ready:
                    future.result()
            if {row["scene"] for row in done} != expected_scenes:
                raise RuntimeError("workers ended without selected scene coverage")
            full = len(done) == len(plan["scenes"])
            if full:
                capture.atomic_json(root / "collection_complete.json", dict(capture_only=True, scenes=done,
                    episodes=sum(x["episodes"] for x in done), dual_session=str(session.relative_to(root))))
            dataset.dump(session / "COMPLETE.json", dict(status="completed", scenes=len(done), full_collection=full))
            print("Selected scenes captured. No labels or training started.")
        except BaseException as exc:
            cancel.set()
            dataset.dump(session / "FAILED.json", dict(error=type(exc).__name__, message=str(exc)))
            print("Stopping this session's children; completed scenes retained.", flush=True)
            raise
        finally:
            cancel.set()
            pool.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------- #
# collect-joint / collect-joint-train (collect_fm_joint + joint_train adapter)
# --------------------------------------------------------------------------- #
def cmd_collect_joint():
    p = argparse.ArgumentParser(description="GPU1 InternScenes train capture with shared-encoder RTC-off labeling.")
    p.add_argument("--output")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/x-navdp_posttrain.ckpt"))
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--port", type=int, default=20016)
    p.add_argument("--scene-limit", type=int, default=0)
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--audit-every", type=int, default=100)
    p.add_argument("--reserve-gib", type=float, default=50.)
    p.add_argument("--scene-timeout-hours", type=float, default=18.)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", choices=("serve", "evaluate"), help=argparse.SUPPRESS)
    p.add_argument("--run", help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.scene_limit, args.episodes) < 0 or args.audit_every < 1 or args.reserve_gib <= 0 or args.scene_timeout_hours <= 0:
        p.error("invalid counts/limits")
    if not 1024 <= args.port <= 65535:
        p.error("invalid port")
    if args.worker:
        if not args.run:
            p.error("worker requires --run")
        os.chdir(BASE)
        if args.worker == "serve":
            joint.serve(args)
        else:
            args.max_steps = 0
            dataset.evaluate(args)
        return
    if not args.output:
        p.error("--output required")
    rows = joint.inventory(args.scene_limit, args.episodes)
    print(json.dumps(dict(scenes=len(rows), home=sum(r["family"] == "home" for r in rows),
        commercial=sum(r["family"] == "commercial" for r in rows), episodes=sum(r["episodes"] for r in rows),
        physical_gpu=args.physical_gpu, online_rtc=True, label_rtc=False, audit_every=args.audit_every), indent=2), flush=True)
    if not args.execute:
        print("Preview only. No GPU jobs or dataset files created.")
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    plan = dict(scenes=rows, checkpoint=str(checkpoint), teacher_sha256=storage.digest(checkpoint),
                seed=args.seed, physical_gpu=args.physical_gpu, audit_every=args.audit_every,
                source_sha256=dataset.code_hashes(), joint_source_sha256=joint.joint_sources(),
                split_sha256=storage.digest(BASE / "data/scenes/scene_split.json"))
    with storage.writer_lock(root):
        plan_path = root / "collection_manifest.json"
        if plan_path.exists():
            if storage.read_json(plan_path) != plan:
                raise ValueError("frozen collection configuration changed")
        else:
            if (root / "train").exists():
                raise ValueError("unrecognized collection")
            dataset.dump(plan_path, plan)
        completed = []
        for row in rows:
            markers = list((root / "train" / row["scene"]).glob("attempt_*/CAPTURE_COMPLETE.json"))
            if len(markers) > 1:
                raise ValueError("multiple completed attempts")
            if markers:
                run = markers[0].parent
                report = storage.read_json(markers[0])
                if (report["index_sha256"] != storage.digest(run / "observation_index.jsonl") or
                        report["metric_sha256"] != storage.digest(run / report["metric_file"]) or
                        report["label_complete_sha256"] != storage.digest(run / "labels/COMPLETE.json") or
                        report["episodes"] != row["episodes"] or report["scene"] != row["scene"]):
                    raise ValueError("completed attempt changed")
            else:
                capture.check_space(root, args.reserve_gib)
                print(f'CAPTURE+LABEL {row["family"]}/{row["scene"]}', flush=True)
                report = joint.capture(root, row, plan, args)
            completed.append(report)
            storage.atomic_json(root / "progress.json", completed)
        storage.atomic_json(root / "collection_complete.json", dict(status="complete", scenes=completed, joint_label_version=1))


def cmd_collect_joint_train():
    """Bind safe TRAIN episode pairs for joint capture (replaces the old install() adapter)."""
    original_sources, original_prepare = joint.joint_sources, joint.prepare
    original_evaluate = dataset.evaluate

    def sources():
        return {**original_sources(), "../../FM_distillation/src/joint.py": storage.digest(BASE / "../../FM_distillation/src/joint.py")}

    def prepare(run, row, plan):
        meta = original_prepare(run, row, plan)
        meta.update(pointgoal_path=row["pairs"], pointgoal_sha256=row["pairs_sha256"])
        storage.atomic_json(run / "manifest.json", meta)
        return meta

    def evaluate(args):
        dataset.select_physical_gpu(args.physical_gpu)
        from eval import environment
        meta = storage.read_json(Path(args.run) / "manifest.json")
        previous = environment.find_eval_pointgoal_path
        environment.find_eval_pointgoal_path = lambda directory: joint.resolve_training_pairs(meta, directory)
        try:
            return original_evaluate(args)
        finally:
            environment.find_eval_pointgoal_path = previous

    joint.joint_sources, joint.prepare = sources, prepare
    dataset.evaluate = evaluate
    joint.CLI_STAGE = "collect-joint-train"
    cmd_collect_joint()


# --------------------------------------------------------------------------- #
# label-validation (label_fm_validation)
# --------------------------------------------------------------------------- #
def cmd_label_validation():
    p = argparse.ArgumentParser(description="Label validation scenes independently of the live train labeler.")
    p.add_argument("--collection", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    for key in ("collection", "snapshot", "output"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    if not args.execute:
        with tempfile.TemporaryDirectory(prefix="fm-val-preview-") as temp:
            snapshot = training.validation_snapshot(args.collection, Path(temp) / "snapshot.json")
        print(json.dumps(dict(scenes=snapshot["scenes"], records=len(snapshot["records"]),
                              physical_gpu=args.physical_gpu, note="preview only; no GPU job"), indent=2))
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with storage.writer_lock(root):
        snapshot = training.validation_snapshot(args.collection, args.snapshot)
        storage.verify_code(snapshot)
        signature = dict(snapshot_sha256=storage.digest(args.snapshot), seed=args.seed,
                         teacher_sha256=snapshot["teacher_sha256"], candidates=8, rtc_enabled=False,
                         pipeline_sha256=training.source_hashes(), validation_runner_sha256=storage.digest(evaluation.__file__))
        manifest = root / "label_manifest.json"
        if manifest.exists():
            if storage.read_json(manifest) != signature:
                raise ValueError("validation resume configuration changed")
        else:
            if any(root.rglob("*.npz")) or (root / "COMPLETE.json").exists():
                raise ValueError("unrecognized output directory")
            storage.atomic_json(manifest, signature)
        dataset.select_physical_gpu(args.physical_gpu)
        import torch
        from FM_distillation.src.fm_data import save_record
        torch.set_num_threads(4)
        os.chdir(BASE)
        teacher, completed = None, []
        for i, row in enumerate(snapshot["records"]):
            path = root / row["id"]
            receipt = path.with_suffix(".json")
            seed = int(hashlib.sha256(f'{args.seed}:{row["id"]}:{row["sha256"]}'.encode()).hexdigest()[:8], 16)
            if path.exists():
                checksum = storage.digest(path)
                if receipt.exists() and storage.read_json(receipt) != {"sha256": checksum}:
                    raise ValueError(f"corrupt validation label: {path}")
                _, meta = storage.LabelCache(root, snapshot, 0).get({**row, "label_sha256": checksum})
                if meta["seed"] != seed:
                    raise ValueError("label seed mismatch")
            else:
                if receipt.exists():
                    raise ValueError("missing label with committed receipt")
                if teacher is None:
                    teacher = training.load_teacher(snapshot, "cuda:0")
                arrays, meta = training.label_one(teacher, row, snapshot, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".pending")
                if temporary.exists():
                    temporary.unlink()
                save_record(temporary, arrays, meta)
                os.replace(temporary, path)
                checksum = storage.digest(path)
            if not receipt.exists():
                storage.atomic_json(receipt, {"sha256": checksum})
            completed.append({**row, "label_sha256": checksum})
            if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(snapshot["records"]):
                print(f'validation label {i+1}/{len(snapshot["records"])}: {row["id"]}', flush=True)
        storage.atomic_json(root / "COMPLETE.json", {**signature, "status": "complete", "records": completed})
        print("Validation labels complete; no training launched.", flush=True)


# --------------------------------------------------------------------------- #
# merge-labels / merge-success
# --------------------------------------------------------------------------- #
def cmd_merge_labels():
    parser = argparse.ArgumentParser(description="CPU-only verified index merge of completed train/validation labels.")
    for key in ("train-snapshot", "train-labels", "val-snapshot", "val-labels", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--inspect", action="store_true")
    merging.merge(parser.parse_args())


def cmd_merge_success():
    p = argparse.ArgumentParser(description="Merge legacy+joint FM labels, filtering failed train episodes only.")
    p.add_argument("--legacy", required=True)
    p.add_argument("--joint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    snapshot, excluded, report = merging.inventory(args.legacy, args.joint)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("use a new immutable output directory")
    output.mkdir(parents=True)
    with storage.writer_lock(output):
        storage.atomic_json(output / "merge_report.json", report)
        import torch
        torch.set_num_threads(1)

        def check(row):
            _, meta = merging.validate_mixed_record(row, snapshot["teacher_sha256"])
            return meta.get("verification", "full")
        scopes = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, scope in enumerate(pool.map(check, snapshot["records"]), 1):
                scopes[scope] += 1
                if i % 5000 == 0 or i == len(snapshot["records"]):
                    print(f'verified {i}/{len(snapshot["records"])} retained labels', flush=True)
        for path, expected in snapshot["source_pins"].items():
            if storage.digest(path) != expected:
                raise ValueError(f"source changed during merge: {path}")
        storage.atomic_json(output / "excluded_failed_train.json", dict(records=excluded,
                    note="References only; all original failed observations/labels preserved."))
        storage.atomic_json(output / "snapshot.json", snapshot)
        storage.atomic_json(output / "COMPLETE.json", dict(status="complete",
            snapshot_sha256=storage.digest(output / "snapshot.json"),
            retained_labels=len(snapshot["records"]), verification_scopes=dict(scopes),
            excluded_manifest_sha256=storage.digest(output / "excluded_failed_train.json"),
            merge_source_sha256=storage.digest(CLI_PATH)))
        print("READY " + str(output), flush=True)


# --------------------------------------------------------------------------- #
# train (train_fm): freeze / label / train / rank
# --------------------------------------------------------------------------- #
def cmd_train():
    p = argparse.ArgumentParser(description="Offline FM pipeline: freeze -> resumable label -> train -> original-Q evaluate.")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="CPU only; immutable completed-scene snapshot")
    f.add_argument("--collection", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--completed-only", action="store_true")
    lab = sub.add_parser("label")
    lab.add_argument("--snapshot", required=True)
    lab.add_argument("--output", required=True)
    lab.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    lab.add_argument("--seed", type=int, default=17)
    lab.add_argument("--reuse-labels")
    lab.add_argument("--max-new-records", type=int, default=0)
    tr = sub.add_parser("train")
    src = tr.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshot", help="merge-labels snapshot; requires --labels")
    src.add_argument("--dataset", help="merge-success mixed dataset root")
    tr.add_argument("--labels", help="labels dir (required with --snapshot)")
    tr.add_argument("--output", required=True)
    tr.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--steps", type=int, default=10000)
    tr.add_argument("--microbatch", type=int, default=8)
    tr.add_argument("--accumulate", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--log-every", type=int, default=25)
    tr.add_argument("--val-per-scene", type=int, default=32)
    tr.add_argument("--resume")
    rk = sub.add_parser("rank")
    rk.add_argument("--snapshot", required=True)
    rk.add_argument("--labels", required=True)
    rk.add_argument("--output", required=True)
    rk.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    rk.add_argument("--seed", type=int, default=17)
    rk.add_argument("--val-per-scene", type=int, default=32)
    rk.add_argument("--student", required=True)
    args = p.parse_args()
    if args.command == "train" and args.snapshot and not args.labels:
        p.error("--labels required with --snapshot")
    for name in ("val_per_scene", "steps", "microbatch", "accumulate", "eval_every", "log_every"):
        if hasattr(args, name) and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if hasattr(args, "lr") and not 0 < args.lr < 1:
        p.error("lr must be in (0,1)")
    if getattr(args, "max_new_records", 0) < 0:
        p.error("max-new-records must be nonnegative")
    for name in ("snapshot", "collection", "output", "labels", "dataset", "resume", "student", "reuse_labels"):
        if getattr(args, name, None):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if args.command == "freeze":
        result = storage.freeze(args.collection, args.output, completed_only=args.completed_only)
        print(json.dumps(dict(scenes=result["scenes"], records=len(result["records"]),
                              skipped_incomplete=result["skipped_incomplete"]), indent=2))
        return
    dataset.select_physical_gpu(args.physical_gpu)
    import torch
    torch.set_num_threads(4)
    os.chdir(BASE)
    {"label": training.label, "train": training.train, "rank": training.rank}[args.command](args)


# --------------------------------------------------------------------------- #
# train-all-scenes (train_fm_all_scenes)
# --------------------------------------------------------------------------- #
def cmd_train_all_scenes():
    p = argparse.ArgumentParser(description="Formal offline FM stage: every optimizer update includes every train scene.")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint")
    source.add_argument("--resume")
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--per-scene", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-per-scene", type=int, default=128)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    if min(args.steps, args.per_scene, args.eval_every, args.val_per_scene, args.log_every) < 1:
        p.error("counts must be positive")
    if not 0 < args.min_lr <= args.lr < 1:
        p.error("require 0 < min-lr <= lr < 1")
    for name in ("snapshot", "labels", "output", "init_checkpoint", "resume"):
        if getattr(args, name, None):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with storage.writer_lock(output):
        if any(f.name != ".writer.lock" for f in output.iterdir()):
            raise ValueError("use a fresh output directory, including when resuming")
        gpu_uuid = dataset.select_physical_gpu(args.physical_gpu)
        import torch
        from FM_distillation.src.flow_generator import CompactFlowGenerator
        torch.set_num_threads(4)
        os.chdir(BASE)
        snapshot, groups, cache = training.labeled_data(args)
        scenes = sorted(k[1] for k in groups if k[0] == "train")
        size = len(scenes) * args.per_scene
        state_path = args.resume or args.init_checkpoint
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        legacy_sources = training.source_hashes()
        sources = {**legacy_sources, "../../FM_distillation/src/training.py": storage.digest(CLI_PATH)}
        identity = dict(snapshot_sha256=storage.digest(args.snapshot),
                        labels_sha256=storage.digest(Path(args.labels) / "COMPLETE.json"),
                        teacher_sha256=snapshot["teacher_sha256"])
        signature = {**identity, "source_sha256": sources, "depth": 4, "width": 384, "time_scale": 9.,
            "precision": "float32", "sampler": "all_train_scenes_equal_per_update_v1", "scenes": scenes,
            "per_scene": args.per_scene, "microbatch": size, "accumulate": 1, "lr": args.lr, "min_lr": args.min_lr,
            "schedule": "cosine_new_stage", "total_steps": args.steps, "seed": args.seed,
            "eval_every": args.eval_every, "val_per_scene": args.val_per_scene}
        previous = state["signature"]
        if args.resume:
            if previous != signature:
                raise ValueError("formal-stage resume configuration mismatch")
            stage_start, best = state["stage_start"], state["best_val_loss"]
        else:
            for key, value in {**identity, "source_sha256": legacy_sources, "depth": 4, "width": 384,
                              "time_scale": 9., "precision": "float32", "lr": args.lr, "seed": args.seed}.items():
                if previous.get(key) != value:
                    raise ValueError(f"legacy checkpoint identity mismatch: {key}")
            stage_start, best = state["step"], float("inf")
        start = state["step"]
        if args.steps <= start:
            raise ValueError("target total steps must exceed checkpoint step")
        torch.manual_seed(args.seed)
        teacher = training.load_teacher(snapshot, "cpu")
        model = CompactFlowGenerator(teacher, depth=4)
        from FM_distillation.src.fm_backbone import backbone_meta
        teacher_state, teacher_meta = teacher.state_dict(), backbone_meta(teacher)
        del teacher
        model.load_state_dict(state["student"], strict=True)
        model = model.to("cuda:0")
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=args.lr, weight_decay=1e-4)
        optimizer.load_state_dict(state["optimizer"])
        rng = torch.Generator()
        rng.set_state(state["batch_rng"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], 0)
        del state
        selected = training.validation_rows(groups, args.val_per_scene)
        storage.atomic_json(output / "config.json", {**signature, **vars(args), "gpu_uuid": gpu_uuid,
            "stage_start": stage_start, "init_step": start, "checkpoint_sha256": storage.digest(state_path),
            "optimizer": "AdamW", "weight_decay": 1e-4, "clip_grad_norm": 1.,
            "validation_best_reset": bool(args.init_checkpoint),
            "transition": "optimizer preserved; sampler/batch/schedule/validation explicitly changed"})
        storage.atomic_json(output / "validation_rows.json", {s: [r["id"] for r in rows] for s, rows in selected.items()})
        print(json.dumps(dict(status="training", start_step=start, total_steps=args.steps, microbatch=size,
                              per_scene=args.per_scene, scenes=scenes, gpu_uuid=gpu_uuid)), flush=True)
        torch.cuda.reset_peak_memory_stats(0)
        wall = time.perf_counter()
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(start + 1, args.steps + 1):
                progress = (step - stage_start - 1) / max(1, args.steps - stage_start - 1)
                lr = args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                model.train()
                optimizer.zero_grad(set_to_none=True)
                began = time.perf_counter()
                rows = training.all_scene_rows(groups, args.per_scene, rng)
                counts = {s: sum(r["scene"] == s for r in rows) for s in scenes}
                if len(rows) != size or any(count != args.per_scene for count in counts.values()):
                    raise RuntimeError("all-scene batch contract violated")
                candidates = torch.randint(8, (size,), generator=rng).tolist()
                inputs = training.batch(cache, rows, candidates, "cuda:0")
                data_time = time.perf_counter() - began
                noise = inputs.pop("initial_noise", None)
                if noise is None:
                    noise = torch.randn(size, 24, 3, generator=rng).to("cuda:0")
                loss = model.flow_loss(**inputs, noise=noise,
                                       t=torch.rand(size, generator=rng).to("cuda:0"))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, train_loss=loss.item(), grad_norm=float(norm), lr=lr,
                              data_prepare_s=data_time, update_wall_s=time.perf_counter() - began)
                if step % args.eval_every == 0 or step == args.steps:
                    record["validation"] = training.evaluate(model, cache, selected, "cuda:0", args.seed + 1000)
                    val = record["validation"]["scene_balanced"]["fm_loss"]
                    improved = val < best
                    best = min(best, val)
                    checkpoint = output / f"step_{step:08d}.pt"
                    training.save_checkpoint(checkpoint, dict(student=model.state_dict(), optimizer=optimizer.state_dict(),
                        signature=signature, step=step, stage_start=stage_start, best_val_loss=best,
                        batch_rng=rng.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0)))
                    storage.atomic_json(output / "latest.json", dict(checkpoint=checkpoint.name, step=step))
                    if improved:
                        storage.atomic_json(output / "best.json", dict(checkpoint=checkpoint.name, val_loss=val))
                if step == start + 1 or step % args.log_every == 0 or "validation" in record:
                    record.update(elapsed_s=time.perf_counter() - wall, scene_counts=counts,
                                  peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                                  peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
                    print(json.dumps(record, allow_nan=False), flush=True)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
        from FM_distillation.src.fm_backbone import export_deploy_checkpoint
        export_deploy_checkpoint(model, teacher_state, teacher_meta, signature, output / "deploy.pt")
        print(f"Deploy checkpoint written: {output / 'deploy.pt'}", flush=True)
        storage.atomic_json(output / "COMPLETE.json", dict(step=args.steps, best_val_loss=best, elapsed_s=time.perf_counter() - wall))


# --------------------------------------------------------------------------- #
# train-all-candidates (train_fm_all_candidates)
# --------------------------------------------------------------------------- #
def cmd_train_all_candidates():
    p = argparse.ArgumentParser(description="All-scene, all-eight-candidate CFM; frozen conditions, no RTC/set/Q loss.")
    p.add_argument("--output", required=True)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--snapshot", help="merge-labels snapshot; requires --labels")
    data.add_argument("--dataset", help="merge-success mixed dataset root")
    p.add_argument("--labels", help="labels dir (required with --snapshot)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint")
    source.add_argument("--resume")
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--per-scene", type=int, default=8)
    p.add_argument("--candidate-microbatch", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-per-scene", type=int, default=128)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dist-lambda-mu", type=float, default=0.0,
                   help="distribution-loss mean-matching weight (0 disables)")
    p.add_argument("--dist-lambda-sigma", type=float, default=0.0,
                   help="distribution-loss spread-matching weight (0 disables)")
    p.add_argument("--dist-rows", type=int, default=4, help="rows sampled for the distribution loss")
    p.add_argument("--dist-steps", type=int, default=4, help="Euler steps for the distribution-loss samples")
    args = p.parse_args()
    if min(args.steps, args.per_scene, args.candidate_microbatch, args.eval_every,
           args.val_per_scene, args.log_every, args.dist_rows, args.dist_steps) < 1 or \
       not 0 < args.min_lr <= args.lr < 1:
        p.error("positive counts and 0 < min-lr <= lr < 1 required")
    if args.dist_lambda_mu < 0 or args.dist_lambda_sigma < 0:
        p.error("distribution-loss lambdas must be nonnegative")
    if args.snapshot and not args.labels:
        p.error("--labels required with --snapshot")
    for name in ("snapshot", "labels", "dataset", "output", "init_checkpoint", "resume"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with storage.writer_lock(output):
        if any(f.name != ".writer.lock" for f in output.iterdir()):
            raise ValueError("use a fresh output directory, including when resuming")
        gpu_uuid = dataset.select_physical_gpu(args.physical_gpu)
        import torch
        from FM_distillation.src.flow_generator import CompactFlowGenerator
        torch.set_num_threads(4)
        os.chdir(BASE)
        if args.dataset:
            snapshot, groups, cache = training.labeled_mixed_data(args.dataset)
        else:
            snapshot, groups, cache = training.labeled_data(args)
        scenes = sorted(k[1] for k in groups if k[0] == "train")
        size = len(scenes) * args.per_scene
        state_path = args.resume or args.init_checkpoint
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        sources = {**training.source_hashes(),
                   "../../FM_distillation/src/training.py": storage.digest(BASE / "../../FM_distillation/src/training.py")}
        if args.dataset:
            identity = dict(dataset_sha256=storage.digest(Path(args.dataset) / "snapshot.json"),
                            dataset_complete_sha256=storage.digest(Path(args.dataset) / "COMPLETE.json"),
                            teacher_sha256=snapshot["teacher_sha256"])
        else:
            identity = dict(snapshot_sha256=storage.digest(args.snapshot),
                            labels_sha256=storage.digest(Path(args.labels) / "COMPLETE.json"),
                            teacher_sha256=snapshot["teacher_sha256"])
        architecture = dict(depth=4, width=384, time_scale=9., precision="float32")
        signature = {**identity, **architecture, "source_sha256": sources,
            "sampler": "all_scenes_all_8_candidates_v1", "scenes": scenes,
            "per_scene": args.per_scene, "observation_batch": size, "candidates": 8,
            "trajectory_batch": size * 8, "candidate_microbatch": args.candidate_microbatch,
            "lr": args.lr, "min_lr": args.min_lr, "schedule": "cosine_new_stage",
            "total_steps": args.steps, "seed": args.seed,
            "dist_lambda_mu": args.dist_lambda_mu, "dist_lambda_sigma": args.dist_lambda_sigma,
            "dist_rows": args.dist_rows, "dist_steps": args.dist_steps,
            "eval_every": args.eval_every, "val_per_scene": args.val_per_scene}
        if args.resume:
            if state["signature"] != signature:
                raise ValueError("all-candidate resume configuration mismatch")
            stage_start, best = state["stage_start"], state["best_val_loss"]
        else:
            for key, value in {**identity, **architecture}.items():
                if state["signature"].get(key) != value:
                    raise ValueError(f"initial checkpoint identity mismatch: {key}")
            for name, checksum in state["signature"]["source_sha256"].items():
                if sources.get(name) != checksum:
                    raise ValueError(f"initial checkpoint source mismatch: {name}")
            stage_start, best = state["step"], float("inf")
        start = state["step"]
        if args.steps <= start:
            raise ValueError("--steps must exceed checkpoint step (total, not additional steps)")
        torch.manual_seed(args.seed)
        teacher = training.load_teacher(snapshot, "cpu")
        model = CompactFlowGenerator(teacher, depth=4)
        from FM_distillation.src.fm_backbone import backbone_meta
        teacher_state, teacher_meta = teacher.state_dict(), backbone_meta(teacher)
        del teacher
        model.load_state_dict(state["student"], strict=True)
        model = model.to("cuda:0")
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                     lr=args.lr, weight_decay=1e-4)
        optimizer.load_state_dict(state["optimizer"])
        rng = torch.Generator()
        rng.set_state(state["batch_rng"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], 0)
        del state
        selected = training.validation_rows(groups, args.val_per_scene)
        storage.atomic_json(output / "config.json", {**signature, **vars(args), "gpu_uuid": gpu_uuid,
            "stage_start": stage_start, "init_step": start, "checkpoint_sha256": storage.digest(state_path),
            "weight_decay": 1e-4, "clip_grad_norm": 1.,
            "transition": "student/AdamW/RNG preserved; all-candidate objective; new cosine stage"})
        storage.atomic_json(output / "validation_rows.json", {s: [r["id"] for r in rows] for s, rows in selected.items()})
        print(json.dumps(dict(status="training", start_step=start, total_steps=args.steps,
            observations=size, trajectories=size * 8, candidate_microbatch=args.candidate_microbatch,
            scenes=scenes, gpu_uuid=gpu_uuid)), flush=True)
        torch.cuda.reset_peak_memory_stats(0)
        wall = time.perf_counter()
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(start + 1, args.steps + 1):
                progress = (step - stage_start - 1) / max(1, args.steps - stage_start - 1)
                lr = args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                model.train()
                optimizer.zero_grad(set_to_none=True)
                began = time.perf_counter()
                rows = training.all_scene_rows(groups, args.per_scene, rng)
                counts = {s: sum(r["scene"] == s for r in rows) for s in scenes}
                if len(rows) != size or any(c != args.per_scene for c in counts.values()):
                    raise RuntimeError("all-scene batch contract violated")
                data = training.all_candidate_batch(cache, rows)
                data_time = time.perf_counter() - began
                errors, times = training.all_candidate_loss(model, data, "cuda:0", rng,
                                                            args.candidate_microbatch, backward=True)
                dist_loss = None
                if args.dist_lambda_mu > 0 or args.dist_lambda_sigma > 0:
                    dist_rows = storage.sample_rows(groups, args.dist_rows, rng)
                    dist_data = training.all_candidate_batch(cache, dist_rows)
                    dist_loss = training.distribution_loss(model, dist_data, "cuda:0", candidates=8,
                                                           steps=args.dist_steps,
                                                           lambda_mu=args.dist_lambda_mu,
                                                           lambda_sigma=args.dist_lambda_sigma)
                    dist_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, train_loss=errors.mean().item(), grad_norm=float(norm), lr=lr,
                    data_prepare_s=data_time, update_wall_s=time.perf_counter() - began,
                    candidate_diagnostics=training.loss_diagnostics(errors, times, data["action_deltas"]),
                    scene_loss={s: errors[[i for i, r in enumerate(rows) if r["scene"] == s]].mean().item()
                                for s in scenes})
                if dist_loss is not None:
                    record["dist_loss"] = float(dist_loss)
                if step % args.eval_every == 0 or step == args.steps:
                    record["validation"] = training.evaluate_candidates(model, cache, selected, "cuda:0",
                                                                        args.seed + 1000, args.candidate_microbatch)
                    val = record["validation"]["scene_balanced"]["fm_loss"]
                    improved = val < best
                    best = min(best, val)
                    checkpoint = output / f"step_{step:08d}.pt"
                    training.save_checkpoint(checkpoint, dict(student=model.state_dict(), optimizer=optimizer.state_dict(),
                        signature=signature, step=step, stage_start=stage_start, best_val_loss=best,
                        batch_rng=rng.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0)))
                    storage.atomic_json(output / "latest.json", dict(checkpoint=checkpoint.name, step=step))
                    if improved:
                        storage.atomic_json(output / "best.json", dict(checkpoint=checkpoint.name, val_loss=val))
                if step == start + 1 or step % args.log_every == 0 or "validation" in record:
                    record.update(elapsed_s=time.perf_counter() - wall, scene_counts=counts,
                        peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
                    print(json.dumps(record, allow_nan=False), flush=True)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
        from FM_distillation.src.fm_backbone import export_deploy_checkpoint
        export_deploy_checkpoint(model, teacher_state, teacher_meta, signature, output / "deploy.pt")
        print(f"Deploy checkpoint written: {output / 'deploy.pt'}", flush=True)
        storage.atomic_json(output / "COMPLETE.json", dict(step=args.steps, best_val_loss=best,
                                                           elapsed_s=time.perf_counter() - wall))


# --------------------------------------------------------------------------- #
# label-dual (dual-branch 4 pointgoal + 4 nogoal teacher labeling)
# --------------------------------------------------------------------------- #
def cmd_label_dual():
    p = argparse.ArgumentParser(description="Offline dual-branch labeling: 4 pointgoal + 4 nogoal per observation.")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    for name in ("snapshot", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    dataset.select_physical_gpu(args.physical_gpu)
    import torch
    torch.set_num_threads(4)
    os.chdir(BASE)
    fm_dual.label_dual(args)


# --------------------------------------------------------------------------- #
# train-dual (dual-branch CFM: beta*Q-weighted pointgoal + alpha*equal nogoal)
# --------------------------------------------------------------------------- #
def cmd_train_dual():
    p = argparse.ArgumentParser(description="Dual-branch CFM: beta*(mild Q-weighted pointgoal) + alpha*(equal nogoal).")
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--snapshot", help="merge-labels snapshot; requires --labels")
    data.add_argument("--dataset", help="merge-success mixed dataset root")
    p.add_argument("--labels", help="labels dir (required with --snapshot)")
    p.add_argument("--output", required=True)
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--candidate-microbatch", type=int, default=160)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--q-lambda", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-per-scene", type=int, default=32)
    args = p.parse_args()
    if args.snapshot and not args.labels:
        p.error("--labels required with --snapshot")
    for name in ("steps", "microbatch", "candidate_microbatch", "accumulate",
                 "eval_every", "log_every", "val_per_scene"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if not 0 < args.lr < 1:
        p.error("lr must be in (0,1)")
    if args.beta < 0 or args.alpha < 0 or (args.beta + args.alpha) <= 0:
        p.error("beta/alpha must be nonnegative and not both zero")
    if not 0 <= args.q_lambda < 1 or args.temperature <= 0:
        p.error("q-lambda in [0,1) and positive temperature required")
    for name in ("snapshot", "labels", "dataset", "output"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    dataset.select_physical_gpu(args.physical_gpu)
    import torch
    torch.set_num_threads(4)
    os.chdir(BASE)
    fm_dual.train_dual(args)


# --------------------------------------------------------------------------- #
# eval-closed-loop (eval_fm_closed_loop)
# --------------------------------------------------------------------------- #
def cmd_eval_closed_loop():
    p = argparse.ArgumentParser(description="FM + frozen original Q + baseline recovery/MPC, no RTC, no recording.")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--student", help="training checkpoint (student + teacher loaded separately)")
    source.add_argument("--deploy", help="self-contained deploy.pt (single weight, no posttrain load)")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/x-navdp_posttrain.ckpt"))
    p.add_argument("--output")
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--port", type=int, default=20015)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtc", choices=("off",), default="off")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--scene-timeout-hours", type=float, default=12.)
    p.add_argument("--worker", choices=("serve", "evaluate"), help=argparse.SUPPRESS)
    p.add_argument("--run", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1 <= args.episodes <= 100 or not 1024 <= args.port <= 65535 or args.scene_timeout_hours <= 0:
        p.error("invalid episodes/port/timeout")
    if args.worker:
        args.run = str(Path(args.run).resolve())
        os.chdir(BASE)
        if args.worker == "serve":
            evaluation.serve_closed_loop(args)
        else:
            args.max_steps = 0
            dataset.evaluate(args)
        return
    if not (args.student or args.deploy) or not args.output:
        p.error("--student/--deploy and --output required")
    print(json.dumps(dict(scenes=["easy_5", "hard_5"], episodes_per_scene=args.episodes,
        physical_gpu=args.physical_gpu, rtc=False, record_num=0, steps=4, candidates=8)), flush=True)
    if not args.execute:
        print("Preview only. Add --execute to run.")
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    teacher = Path(args.checkpoint).resolve(strict=True)
    if args.deploy:
        deploy = Path(args.deploy).resolve(strict=True)
        identity = dict(deploy=str(deploy), deploy_sha256=storage.digest(deploy), checkpoint=str(teacher),
            teacher_sha256=storage.digest(teacher), episodes=args.episodes, seed=args.seed,
            physical_gpu=args.physical_gpu, rtc=False, record_num=0, runner_sha256=storage.digest(evaluation.__file__),
            code_sha256=dataset.code_hashes())
    else:
        student = Path(args.student).resolve(strict=True)
        identity = dict(student=str(student), student_sha256=storage.digest(student), checkpoint=str(teacher),
            teacher_sha256=storage.digest(teacher), episodes=args.episodes, seed=args.seed,
            physical_gpu=args.physical_gpu, rtc=False, record_num=0, runner_sha256=storage.digest(evaluation.__file__),
            code_sha256=dataset.code_hashes())
    with storage.writer_lock(root):
        if (root / "evaluation_manifest.json").exists():
            if storage.read_json(root / "evaluation_manifest.json") != identity:
                raise ValueError("evaluation identity changed; choose a new output")
        else:
            if (root / "easy_5").exists() or (root / "hard_5").exists():
                raise ValueError("unrecognized output")
            dataset.dump(root / "evaluation_manifest.json", identity)
        for scene in ("easy_5", "hard_5"):
            parent = root / scene
            markers = list(parent.glob("attempt_*/EVAL_COMPLETE.json"))
            if len(markers) > 1:
                raise ValueError("multiple completed attempts")
            if markers:
                report = storage.read_json(markers[0])
                _, metric = capture.completed_metrics(markers[0].parent, args.episodes)
                if storage.digest(metric) != report["metric_sha256"]:
                    raise ValueError("completed metrics changed")
                print(f"SKIP completed {scene}", flush=True)
                continue
            number = 1
            while (parent / f"attempt_{number:03d}").exists():
                number += 1
            run = parent / f"attempt_{number:03d}"
            config = BASE / "eval/config/eval_pointgoal" / f"humanoid_clutter_{'easy' if scene == 'easy_5' else 'hard'}.yaml"
            dataset.prepare(argparse.Namespace(run=str(run), checkpoint=str(teacher), config=str(config),
                scene=scene, split="validation", seed=args.seed, physical_gpu=args.physical_gpu))
            meta = storage.read_json(run / "manifest.json")
            meta["online"].update(sampler="fm", steps=4, rtc=False)
            meta["evaluation_only"] = True
            storage.atomic_json(run / "manifest.json", meta)
            storage.atomic_json(run / "fm_settings.json", identity)
            env = capture.worker_env(args.physical_gpu)
            env["X_NAVDP_MPC_CODEGEN_DIR"] = str(run / "mpc_codegen")
            command = [sys.executable, str(CLI_PATH), "eval-closed-loop", "--run", str(run),
                       "--physical-gpu", str(args.physical_gpu), "--port", str(args.port), "--episodes", str(args.episodes)]
            server = evaluator = None
            try:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", args.port))
                with (run / "server.log").open("w") as slog, (run / "evaluate.log").open("w") as elog:
                    server = subprocess.Popen(command + ["--worker", "serve"], cwd=BASE, env=env,
                        stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
                    for _ in range(90):
                        if server.poll() is not None:
                            raise RuntimeError(f"server exit {server.returncode}; inspect server.log")
                        if "Running on" in (run / "server.log").read_text(errors="replace"):
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError("server readiness timeout")
                    evaluator = subprocess.Popen(command + ["--worker", "evaluate"], cwd=BASE, env=env,
                        stdout=elog, stderr=subprocess.STDOUT, start_new_session=True)
                    started = time.monotonic()
                    while evaluator.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError(f"server exit {server.returncode}")
                        if time.monotonic() - started > args.scene_timeout_hours * 3600:
                            raise RuntimeError("scene timeout")
                        time.sleep(3)
                    if evaluator.returncode:
                        raise RuntimeError(f"evaluation exit {evaluator.returncode}; inspect evaluate.log")
                capture.stop_process(server)
                rows, metric = capture.completed_metrics(run, args.episodes)
                report = dict(scene=scene, episodes=len(rows), metric_file=str(metric), metric_sha256=storage.digest(metric),
                              success_rate=sum(float(r["success"]) for r in rows) / len(rows), rtc=False, record_num=0)
                storage.atomic_json(run / "EVAL_COMPLETE.json", report)
                print(json.dumps(report), flush=True)
            except BaseException as exc:
                storage.atomic_json(run / "EVAL_FAILED.json", dict(error=type(exc).__name__, message=str(exc)))
                raise
            finally:
                capture.stop_process(evaluator)
                capture.stop_process(server)
        storage.atomic_json(root / "COMPLETE.json", dict(status="complete", scenes=["easy_5", "hard_5"], episodes=2 * args.episodes))


# --------------------------------------------------------------------------- #
# eval-rtc (eval_fm_rtc)
# --------------------------------------------------------------------------- #
def cmd_eval_rtc():
    p = argparse.ArgumentParser(description="FM trajectory RTC evaluation, 4 steps/8 candidates, no video.")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--student", help="training checkpoint (student + teacher loaded separately)")
    source.add_argument("--deploy", help="self-contained deploy.pt (single weight, no posttrain load)")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/x-navdp_posttrain.ckpt"))
    p.add_argument("--output")
    p.add_argument("--compare-off")
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--port", type=int, default=20017)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtc", choices=("on", "off"), default="on")
    p.add_argument("--rtc-beta", type=float, default=5.)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--scene-timeout-hours", type=float, default=12.)
    p.add_argument("--worker", choices=("serve", "evaluate"), help=argparse.SUPPRESS)
    p.add_argument("--run", help=argparse.SUPPRESS)
    args = p.parse_args()
    if (not 1 <= args.episodes <= 100 or not 1024 <= args.port <= 65535 or args.scene_timeout_hours <= 0
            or not math.isfinite(args.rtc_beta) or args.rtc_beta <= 0):
        p.error("invalid episodes/port/timeout/beta")
    os.chdir(BASE)
    if args.worker:
        if args.worker == "serve":
            evaluation.serve_rtc(args)
        else:
            args.max_steps = 0
            dataset.evaluate(args)
        return
    if not (args.student or args.deploy) or not args.output:
        p.error("--student/--deploy and --output required")
    rtc_on = args.rtc == "on"
    if args.compare_off and not rtc_on:
        p.error("--compare-off requires --rtc on")
    print(json.dumps(dict(scenes=["easy_5", "hard_5"], episodes_per_scene=args.episodes,
        physical_gpu=args.physical_gpu, rtc=rtc_on, rtc_beta=args.rtc_beta, record_num=0, steps=4, candidates=8)), flush=True)
    if not args.execute:
        print("Preview only. Add --execute to run.")
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    teacher = Path(args.checkpoint).resolve(strict=True)
    if args.deploy:
        deploy = Path(args.deploy).resolve(strict=True)
        identity = dict(deploy=str(deploy), deploy_sha256=storage.digest(deploy), checkpoint=str(teacher),
            teacher_sha256=storage.digest(teacher), episodes=args.episodes, seed=args.seed,
            physical_gpu=args.physical_gpu, rtc=rtc_on, rtc_beta=args.rtc_beta, record_num=0,
            runner_sha256=storage.digest(evaluation.__file__),
            rtc_source_sha256={name: storage.digest(BASE / name) for name in
                               ("../../FM_distillation/src/evaluation.py", "../../FM_distillation/src/rtc.py")},
            code_sha256=dataset.code_hashes(),
            guidance="flow endpoint VJP; beta-clipped coefficient; legacy XY prefix and 9*(1-t)<=guidance_step")
    else:
        student = Path(args.student).resolve(strict=True)
        identity = dict(student=str(student), student_sha256=storage.digest(student), checkpoint=str(teacher),
            teacher_sha256=storage.digest(teacher), episodes=args.episodes, seed=args.seed,
            physical_gpu=args.physical_gpu, rtc=rtc_on, rtc_beta=args.rtc_beta, record_num=0,
            runner_sha256=storage.digest(evaluation.__file__),
            rtc_source_sha256={name: storage.digest(BASE / name) for name in
                               ("../../FM_distillation/src/evaluation.py", "../../FM_distillation/src/rtc.py")},
            code_sha256=dataset.code_hashes(),
            guidance="flow endpoint VJP; beta-clipped coefficient; legacy XY prefix and 9*(1-t)<=guidance_step")
    if args.compare_off:
        old = storage.read_json(Path(args.compare_off) / "evaluation_manifest.json")
        source_keys = [k for k in ("student_sha256", "deploy_sha256") if k in old and k in identity]
        if len(source_keys) != 1:
            raise ValueError("baseline and experiment must agree on exactly one student/deploy source")
        for key in (*source_keys, "teacher_sha256", "episodes", "seed", "code_sha256"):
            if old[key] != identity[key]:
                raise ValueError(f"baseline mismatch: {key}")
        for scene in ("easy_5", "hard_5"):
            evaluation.read_completed(args.compare_off, scene)
    with storage.writer_lock(root):
        if (root / "evaluation_manifest.json").exists():
            if storage.read_json(root / "evaluation_manifest.json") != identity:
                raise ValueError("evaluation identity changed; choose new output")
        else:
            if (root / "easy_5").exists() or (root / "hard_5").exists():
                raise ValueError("unrecognized output")
            dataset.dump(root / "evaluation_manifest.json", identity)
        for scene in ("easy_5", "hard_5"):
            parent = root / scene
            if list(parent.glob("attempt_*/EVAL_COMPLETE.json")):
                evaluation.read_completed(root, scene)
                print(f"SKIP completed {scene}", flush=True)
                continue
            number = 1
            while (parent / f"attempt_{number:03d}").exists():
                number += 1
            run = parent / f"attempt_{number:03d}"
            config = BASE / "eval/config/eval_pointgoal" / f"humanoid_clutter_{'easy' if scene == 'easy_5' else 'hard'}.yaml"
            dataset.prepare(argparse.Namespace(run=str(run), checkpoint=str(teacher), config=str(config),
                scene=scene, split="validation", seed=args.seed, physical_gpu=args.physical_gpu))
            meta = storage.read_json(run / "manifest.json")
            meta["online"].update(sampler="fm", steps=4, rtc=rtc_on)
            meta["evaluation_only"] = True
            storage.atomic_json(run / "manifest.json", meta)
            storage.atomic_json(run / "fm_settings.json", identity)
            env = capture.worker_env(args.physical_gpu)
            env["X_NAVDP_MPC_CODEGEN_DIR"] = str(run / "mpc_codegen")
            command = [sys.executable, str(CLI_PATH), "eval-rtc", "--run", str(run),
                       "--physical-gpu", str(args.physical_gpu), "--port", str(args.port), "--episodes", str(args.episodes)]
            server = evaluator = None
            try:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", args.port))
                with (run / "server.log").open("w") as slog, (run / "evaluate.log").open("w") as elog:
                    server = subprocess.Popen(command + ["--worker", "serve"], cwd=BASE, env=env,
                        stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
                    for _ in range(90):
                        if server.poll() is not None:
                            raise RuntimeError(f"server exit {server.returncode}; inspect server.log")
                        if "Running on" in (run / "server.log").read_text(errors="replace"):
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError("server readiness timeout")
                    evaluator = subprocess.Popen(command + ["--worker", "evaluate"], cwd=BASE, env=env,
                        stdout=elog, stderr=subprocess.STDOUT, start_new_session=True)
                    started = time.monotonic()
                    while evaluator.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError(f"server exit {server.returncode}")
                        if time.monotonic() - started > args.scene_timeout_hours * 3600:
                            raise RuntimeError("scene timeout")
                        time.sleep(3)
                    if evaluator.returncode:
                        raise RuntimeError(f"evaluation exit {evaluator.returncode}; inspect evaluate.log")
                capture.stop_process(server)
                rows, metric = capture.completed_metrics(run, args.episodes)
                report = dict(scene=scene, **evaluation.metric_summary(rows), metric_file=str(metric),
                              metric_sha256=storage.digest(metric), rtc=rtc_on, record_num=0)
                storage.atomic_json(run / "EVAL_COMPLETE.json", report)
                print(json.dumps(report), flush=True)
            except BaseException as exc:
                storage.atomic_json(run / "EVAL_FAILED.json", dict(error=type(exc).__name__, message=str(exc)))
                raise
            finally:
                capture.stop_process(evaluator)
                capture.stop_process(server)
        storage.atomic_json(root / "COMPLETE.json", dict(status="complete", scenes=["easy_5", "hard_5"], episodes=2 * args.episodes))
        if args.compare_off:
            comparison = evaluation.compare_roots(args.compare_off, root)
            storage.atomic_json(root / "rtc_comparison.json", comparison)
            print(json.dumps(comparison), flush=True)


# --------------------------------------------------------------------------- #
# queue (queue_fm_rtc_eval)
# --------------------------------------------------------------------------- #
def cmd_queue():
    p = argparse.ArgumentParser(description="Wait for FM training completion and idle GPU, then evaluate RTC.")
    p.add_argument("--training", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--compare-off", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    args = p.parse_args()
    training_dir, student, off, output = (Path(getattr(args, k)).resolve() for k in
                                          ("training", "student", "compare_off", "output"))
    config_path = training_dir / "config.json"
    config = storage.read_json(config_path)
    expected_step = config["total_steps"]
    baseline = storage.read_json(off / "evaluation_manifest.json")
    if baseline["rtc"] or storage.digest(student) != baseline["student_sha256"]:
        raise ValueError("baseline must be RTC-off with identical student")
    if dataset.code_hashes() != baseline["code_sha256"]:
        raise ValueError("baseline core code changed")
    for scene in ("easy_5", "hard_5"):
        evaluation.read_completed(off, scene)
    sources = {str(BASE / name): storage.digest(BASE / name) for name in
               ("../../FM_distillation/src/rtc.py", "../../FM_distillation/src/evaluation.py")}
    pinned = {**sources, str(config_path): storage.digest(config_path), str(student): storage.digest(student),
              str(off / "evaluation_manifest.json"): storage.digest(off / "evaluation_manifest.json")}
    queue_dir = output.with_name(output.name + "_queue")
    queue_dir.mkdir(parents=True, exist_ok=True)
    with storage.writer_lock(queue_dir):
        if (queue_dir / "COMPLETE.json").exists():
            raise ValueError("queue already finished")
        command = [sys.executable, "-u", str(CLI_PATH), "eval-rtc",
            "--student", str(student), "--checkpoint", baseline["checkpoint"],
            "--output", str(output), "--compare-off", str(off), "--physical-gpu", str(args.physical_gpu),
            "--episodes", str(baseline["episodes"]), "--seed", str(baseline["seed"]),
            "--rtc", "on", "--rtc-beta", "5", "--execute"]
        storage.atomic_json(queue_dir / "config.json", dict(training=str(training_dir), expected_step=expected_step,
                                                            pinned=pinned, command=command))
        previous = None
        while True:
            complete = training_dir / "COMPLETE.json"
            done = complete.exists()
            if done and storage.read_json(complete)["step"] != expected_step:
                raise ValueError("training completion step mismatch")
            pids = scheduling.gpu_compute_pids(args.physical_gpu)
            status = dict(state="ready" if done and not pids else "waiting",
                          training_complete=done, gpu=args.physical_gpu, compute_pids=pids)
            storage.atomic_json(queue_dir / "status.json", status)
            if status != previous:
                print(json.dumps(status), flush=True)
                previous = status
            if done and not pids:
                break
            time.sleep(30)
        for name, checksum in pinned.items():
            if storage.digest(name) != checksum:
                raise ValueError(f"queued input/source changed: {name}")
        storage.atomic_json(queue_dir / "status.json", dict(state="running", training_complete=True, gpu=args.physical_gpu))
        result = subprocess.run(command, cwd=BASE)
        report = dict(returncode=result.returncode, state="complete" if result.returncode == 0 else "failed")
        storage.atomic_json(queue_dir / "status.json", report)
        storage.atomic_json(queue_dir / ("COMPLETE.json" if result.returncode == 0 else "FAILED.json"), report)
        raise SystemExit(result.returncode)


# --------------------------------------------------------------------------- #
# bench (benchmark_fm_baseline_single)
# --------------------------------------------------------------------------- #
def cmd_bench():
    p = argparse.ArgumentParser(description="Batch=1 paired full-network latency, not batch-amortized throughput.")
    p.add_argument("--observations", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if min(args.count, args.repeats, args.warmup, args.threads) < 1:
        p.error("counts must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    uuid_val = dataset.select_physical_gpu(args.physical_gpu)
    import numpy as np
    import torch
    from ddim.src.entry import load_observations, build_model
    from FM_distillation.src.flow_generator import CompactFlowGenerator, generate_and_rank
    from ddim.src.diffusion_sampling import sampling_timesteps
    torch.set_num_threads(args.threads)
    records = load_observations(SimpleNamespace(observations=[args.observations], max_observations=args.count))
    if any(r["prev_action"] is None for r in records):
        raise ValueError("RTC history missing")
    model = build_model(args.checkpoint, torch.device("cuda:0"))
    model.requires_grad_(False)
    state = torch.load(args.student, map_location="cpu", weights_only=True)
    if state["signature"]["teacher_sha256"] != storage.digest(args.checkpoint):
        raise ValueError("teacher identity mismatch")
    student = CompactFlowGenerator(model, depth=4).to("cuda:0").eval().requires_grad_(False)
    student.load_state_dict(state["student"], strict=True)
    del state
    configs = [(sampler, steps, rtc_) for sampler, steps in [("ddpm", 10), ("ddim", 5), ("ddim", 6), ("fm", 4)]
               for rtc_ in (False, True)]
    names = [f"{s}{n}_rtc_{'on' if r else 'off'}" for s, n, r in configs]

    def call(config, record, seed):
        sampler, steps, rtc_ = config
        rgb, depth, goal = (record[k][None] for k in ("rgb", "depth", "pointgoal"))
        if goal.shape != (1, 3) or rgb.shape != (1, 8, 224, 224, 3):
            raise ValueError("expected one unbatched saved observation")
        previous = record["prev_action"][None]
        valid = np.array([record["valid_segment_len"]], dtype=np.int32)
        factor = np.array([.5] * 6 + [.05] * 2, dtype=np.float32)
        torch.manual_seed(seed)
        noise = torch.randn(1, 8, 24, 3, device="cuda:0")
        model.rtc_enabled = rtc_
        if sampler != "fm":
            model.sampler = sampler
            model.sampling_timesteps = sampling_timesteps(sampler, steps)

        class GuidedView:
            training = False
            _embodiment = student._embodiment

            def sample(self, *a, **kw):
                return rtc.sample_with_rtc(student, *a, **kw, prev_action=previous, valid_segment_len=valid,
                    guidance_factor=factor, start_index=0, end_index=23, guidance_step=5, beta=5.)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if sampler == "fm":
            with torch.no_grad():
                g = model.point_encoder(torch.as_tensor(goal, device="cuda:0")).unsqueeze(1)
                r = model.rgbd_encoder(rgb, depth)
                result = generate_and_rank(model, GuidedView() if rtc_ else student, g, r,
                    record["embodiment"], steps=4, candidates=8, initial_noise=noise)
                paths, scores, top = (result[k].cpu().numpy() for k in ("trajectories", "scores", "top_trajectories"))
        else:
            paths, scores, top, _ = model.predict_pointgoal_action_with_guidance(goal, rgb, depth, 8, valid,
                previous, 0, 23, factor, guidance_step=5, embodiment=record["embodiment"],
                initial_noise=noise.reshape(8, 24, 3))
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        if not all(np.isfinite(x).all() for x in (paths, scores, top)):
            raise ValueError("nonfinite prediction")
        return elapsed

    print(json.dumps(dict(gpu=uuid_val, batch=1, candidates=8, observations=len(records),
                          repeats=args.repeats, threads=args.threads)), flush=True)
    warm_record = next(r for r in records if r["valid_segment_len"] > 0)
    for config, name in zip(configs, names):
        for i in range(args.warmup):
            call(config, warm_record, i)
        print("warmup " + name, flush=True)
    measurements = {name: [] for name in names}
    rng = np.random.default_rng(17)
    for i, record in enumerate(records):
        for repeat in range(args.repeats):
            for c in rng.permutation(len(configs)):
                ms = call(configs[c], record, 1000 + i * args.repeats + repeat)
                measurements[names[c]].append(dict(observation=i, repeat=repeat, ms=ms,
                                                   valid_segment_len=record["valid_segment_len"]))
        print(f"observation {i+1}/{len(records)} complete", flush=True)
    summary = {}
    for name, rows in measurements.items():
        values = np.array([r["ms"] for r in rows])
        summary[name] = dict(count=len(rows), mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                             p95_ms=float(np.percentile(values, 95)))
    result = dict(metadata={**vars(args), "batch": 1, "candidates": 8, "gpu_uuid": uuid_val,
        "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(0),
        "teacher_sha256": storage.digest(args.checkpoint), "student_sha256": storage.digest(args.student),
        "source_sha256": storage.digest(CLI_PATH),
        "observations": [dict(path=r["path"], sha256=storage.digest(r["path"]), valid_segment_len=r["valid_segment_len"]) for r in records],
        "scope": p.description, "parameters_frozen_for_all": True,
        "configuration_order": "randomized per observation/repeat",
        "rtc": "fixed 6 strong/2 weak; no online stuck bypass; FM beta5, last2 of4 steps guided"},
        summary=summary, measurements=measurements)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


# --------------------------------------------------------------------------- #
# export-deploy: synthesize a self-contained deploy.pt from an existing student
# --------------------------------------------------------------------------- #
def cmd_export_deploy():
    p = argparse.ArgumentParser(description="Bundle an existing training checkpoint + teacher into one deploy.pt.")
    p.add_argument("--student", required=True, help="existing training checkpoint (step_*.pt)")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/x-navdp_posttrain.ckpt"),
                   help="frozen teacher posttrain checkpoint")
    p.add_argument("--output", required=True, help="path for the self-contained deploy.pt")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time-scale", type=float, default=9.0)
    args = p.parse_args()
    import torch
    from FM_distillation.src.flow_generator import CompactFlowGenerator
    from FM_distillation.src.fm_backbone import backbone_meta, export_deploy_checkpoint
    from bridge.teacher_adapter import load_checkpoint

    state = torch.load(args.student, map_location="cpu", weights_only=True)
    signature = state["signature"]
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    expected = signature.get("teacher_sha256")
    if expected and storage.digest(checkpoint) != expected:
        raise ValueError(f"teacher checkpoint sha256 mismatch; expected {expected}")

    teacher = load_checkpoint(str(checkpoint), "cpu")
    student = CompactFlowGenerator(teacher, depth=args.depth, time_scale=args.time_scale)
    student.load_state_dict(state["student"], strict=True)
    student.eval().requires_grad_(False)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    export_deploy_checkpoint(student, teacher.state_dict(), backbone_meta(teacher), signature, output)
    print(f"Deploy checkpoint written: {output}", flush=True)


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
STAGES = {
    "dataset": cmd_dataset,
    "collect-full": cmd_collect_full,
    "collect-dual": cmd_collect_dual,
    "collect-joint": cmd_collect_joint,
    "collect-joint-train": cmd_collect_joint_train,
    "label-validation": cmd_label_validation,
    "merge-labels": cmd_merge_labels,
    "merge-success": cmd_merge_success,
    "train": cmd_train,
    "train-all-scenes": cmd_train_all_scenes,
    "train-all-candidates": cmd_train_all_candidates,
    "label-dual": cmd_label_dual,
    "train-dual": cmd_train_dual,
    "export-deploy": cmd_export_deploy,
    "eval-closed-loop": cmd_eval_closed_loop,
    "eval-rtc": cmd_eval_rtc,
    "queue": cmd_queue,
    "bench": cmd_bench,
}

_HELP = """\
usage: python -m FM_distillation.cli <stage> [args]

stages:
  dataset             prepare / serve / evaluate / label / validate (teacher capture)
  collect-full        sequential GPU-1 capture sweep
  collect-dual        adopt an existing collection across two GPUs
  collect-joint       GPU1 InternScenes train capture + shared-encoder labeling
  collect-joint-train joint capture bound to safe TRAIN episode pairs
  label-validation    label validation scenes independently
  merge-labels        merge completed train/validation label indexes
  merge-success       merge legacy+joint labels, filter failed train episodes
  train               freeze / label / train / rank (base offline FM)
  train-all-scenes    every optimizer update includes every train scene
  train-all-candidates all-scene, all-eight-candidate CFM
  label-dual          4 pointgoal + 4 nogoal teacher labeling
  train-dual          beta*Q-weighted pointgoal + alpha*equal nogoal CFM
  eval-closed-loop    FM + frozen Q + recovery/MPC, no RTC
  eval-rtc            FM trajectory RTC evaluation
  queue               wait for training + idle GPU, then run RTC eval
  bench               batch=1 full-network latency benchmark
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return
    stage = argv[0]
    if stage not in STAGES:
        print(f"unknown stage: {stage}\n\n{_HELP}", file=sys.stderr)
        raise SystemExit(2)
    sys.argv = argv
    STAGES[stage]()


if __name__ == "__main__":
    main()
