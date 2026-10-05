"""P0/P1: prepare, record teacher observations, label offline, validate. No training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

from rexnavdp import BASE, ROOT


def select_physical_gpu(physical_gpu=1):
    # UUID selection is independent of inherited CUDA_VISIBLE_DEVICES ordering.
    if physical_gpu not in (0, 1):
        raise ValueError("capture supports physical GPU 0 or 1")
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={physical_gpu}", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError(f"cannot resolve physical GPU {physical_gpu}")
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      ISAAC_ACTIVE_GPU=str(physical_gpu), ISAAC_PHYSICS_GPU="0")
    return uuid


def default_gpu():
    """Default physical GPU for labelling/training (GPU 1 on the shared server)."""
    return select_physical_gpu(1)


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def code_hashes():
    # Include untracked Python sources; git diff alone would omit new modules.
    paths = set()
    for part in ("eval", "src", "third_party/depth_anything"):
        paths.update((BASE / part).rglob("*.py"))
    paths.add(Path(__file__).resolve())
    for part in ("FM_distillation", "rl", "ddim", "bridge"):
        paths.update((ROOT / part).rglob("*.py"))
    return {os.path.relpath(p, BASE): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def manifest(run):
    return json.loads((run / "manifest.json").read_text())


def check_frozen(run):
    from FM_distillation.core.fm_data import sha256
    meta = manifest(run)
    if sha256(meta["checkpoint"]) != meta["teacher_sha256"]:
        raise RuntimeError("teacher checkpoint changed since prepare")
    if code_hashes() != meta["code_sha256"]:
        raise RuntimeError("source changed since prepare; prepare a new run instead of mixing versions")
    for name, digest in meta["config_sha256"].items():
        if sha256(run / name) != digest:
            raise RuntimeError(f"prepared config changed: {name}")
    return meta


def prepare(args):
    import yaml
    from FM_distillation.core.fm_data import sha256, SCHEMA
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text())
    split_path = Path(getattr(args, "scene_assignments", None) or BASE / "../FM_distillation/config/fm_scene_split.json")
    split = json.loads(split_path.read_text())
    assignments = [split[key] for key in ("train", "validation", "test", "reserved")]
    all_scenes = sum(assignments, [])
    if len(all_scenes) != len(set(all_scenes)):
        raise ValueError("scene split overlaps")
    if args.split != "debug" and args.scene not in split[args.split]:
        raise ValueError(f"{args.scene} is not assigned to {args.split}")
    if args.scene not in all_scenes:
        raise ValueError("scene absent from fixed split")
    family = cfg["environment"]["scene_type"]
    prefix = {"cluttered_easy": "easy_", "cluttered_hard": "hard_"}.get(family)
    if prefix is None or not args.scene.startswith(prefix):
        raise ValueError("scene/config family mismatch")
    if cfg["environment"]["embodiment"] != "unitree_g1":
        raise ValueError("P1 launcher currently supports G1 only")
    # No overwrite or resume: separate runs remain immutable and attributable.
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "observations").mkdir()
    local_split = {f"{family}_train": [args.scene], f"{family}_eval": [args.scene]}
    dump(run / "selected_scene.json", local_split)
    dump(run / "scene_assignments.json", split)
    cfg["environment"].update(scene_split_file=str(run / "selected_scene.json"),
                               scene_split="train", scene_index=0, num_envs=1, device="cuda:0")
    cfg["run_root_dir"] = str(run / "evaluation")
    cfg["run_id"] = "fm_capture"
    (run / "eval_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with (run / "working_tree.diff").open("wb") as handle:
        subprocess.run(["git", "diff", "--binary", "--", "."],
                       cwd=ROOT, stdout=handle, check=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    sources = code_hashes()
    with zipfile.ZipFile(run / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sources:
            archive.write(BASE / relative, (BASE / relative).resolve().relative_to(ROOT))
    meta = {"schema": SCHEMA, "run_id": f"{run.name}:{uuid.uuid4().hex[:12]}", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scene": args.scene, "split": args.split, "checkpoint": str(checkpoint),
            "teacher_sha256": sha256(checkpoint), "base_config": str(config_path),
            "base_config_sha256": sha256(config_path), "code_sha256": sources,
            "config_sha256": {p: sha256(run/p) for p in
                              ("eval_config.yaml", "selected_scene.json", "scene_assignments.json")},
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       capture_output=True).stdout.strip() or "uncommitted-export",
            "git_status": status, "seed": args.seed, "physical_gpu": getattr(args,"physical_gpu",0),
            "online": {"sampler": "ddpm", "steps": 10, "rtc": True, "recovery": "baseline",
                       "stuck_mode": "time", "stuck_window_s": 2., "stuck_distance_m": .25},
            "label": {"sampler": "ddpm", "steps": 10, "rtc": False, "candidates": 8},
            "coordinates": {"path": "robot-local cumulative XYZ positions, meters; channel 2 is not yaw",
                            "color": "baseline tensor order unchanged (server converts RGB to BGR)",
                            "raw": "pre-smoothing action delta; path=cumsum(raw/4)",
                            "quaternion": "world pose SciPy xyzw", "time_interval":
                            "4 is preserved action scaling, NOT an asserted physical waypoint frequency"},
            "limits": ["capture changes I/O latency; not a timing baseline", "no execution success labels yet",
                       "history timestamps describe observations received by policy, not camera exposure timestamps"]}
    dump(run / "manifest.json", meta)
    print(f"Prepared {run}\nScene={args.scene}, split={args.split}; physical GPU {meta['physical_gpu']}")


def serve(args):
    physical_gpu = getattr(args,"physical_gpu",0)
    uuid = select_physical_gpu(physical_gpu)
    run = Path(args.run).resolve()
    meta = check_frozen(run)
    if meta["physical_gpu"] != physical_gpu:
        raise RuntimeError("GPU differs from prepared run")
    if any((run / "observations").iterdir()):
        raise RuntimeError("observations already exist; do not restart capture into the same run")
    from eval.src import policy_server
    from FM_distillation.core.fm_capture_agent import recording_agent_class
    import torch
    dump(run / "capture_runtime.json", {"gpu_uuid": uuid, "physical_gpu": physical_gpu, "torch": torch.__version__,
                                      "cuda": torch.version.cuda, "python": sys.version})
    policy_server.NavDP_Agent = recording_agent_class(run, meta)
    policy_server.init_app("humanoid", no_visualization=True, device="cuda:0",
                           checkpoint=meta["checkpoint"], seed=meta["seed"],
                           recovery="baseline", rtc_enabled=True)
    policy_server.run_server(port=args.port)


def evaluate(args):
    physical_gpu = getattr(args,"physical_gpu",0)
    select_physical_gpu(physical_gpu)
    run = Path(args.run).resolve()
    if check_frozen(run)["physical_gpu"] != physical_gpu:
        raise RuntimeError("GPU differs from prepared run")
    # Force single-GPU Vulkan rendering in addition to CUDA/physics visibility.
    from isaaclab import app
    original = app.AppLauncher
    class SingleGPULauncher(original):
        def __init__(self, *a, **kw):
            kw.update(active_gpu=physical_gpu, physics_gpu=0, multi_gpu=False)
            super().__init__(*a, **kw)
    app.AppLauncher = SingleGPULauncher
    import runpy
    sys.argv = ["evaluate_pointgoal", "--config_file", str(run / "eval_config.yaml"),
                "--scene_index", "0", "--device", "cuda:0", "--num_envs", "1",
                "--record_num", "0",
                "--server_port", str(args.port), "--strict_pointgoal", "--keep_server"]
    if args.episodes:
        sys.argv.extend(["--num_episodes", str(args.episodes)])
    if args.max_steps:
        sys.argv.extend(["--max_steps", str(args.max_steps)])
    runpy.run_module("eval.scripts.evaluate_pointgoal", run_name="__main__")


def label(args):
    uuid = default_gpu()
    import numpy as np
    import torch
    from FM_distillation.core.fm_data import load_record, save_record, validate_observation, validate_label, teacher_tap, sha256
    from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment
    run = Path(args.run).resolve()
    meta = check_frozen(run)
    files = sorted((run / "observations").glob("*.npz"))
    if not files:
        raise ValueError("no observations; capture first")
    # Even temporal coverage for a small smoke subset, not a random frame split.
    files = [files[i] for i in np.linspace(0, len(files)-1, min(args.limit, len(files)), dtype=int)]
    destination = run / args.name
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(meta["seed"])
    model = NavDP_Policy_Embodiment(temporal_depth=16, device="cuda:0", rtc_enabled=False).to("cuda:0")
    state = torch.load(meta["checkpoint"], map_location="cpu", weights_only=True)
    keys = model.load_state_dict(state, strict=False)
    if keys.missing_keys:
        raise RuntimeError(f"missing weights: {keys.missing_keys}")
    del state
    model.eval()
    provenance = {"status": "started; COMPLETE.json is authoritative", "teacher_sha256": meta["teacher_sha256"],
                  "gpu_uuid": uuid, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "python": sys.version, "precision": "float32", "unexpected_keys": keys.unexpected_keys,
                  "sources": {p.name: sha256(p) for p in files}, "verification": "every record"}
    dump(destination / "label_manifest.json", provenance)
    label_hashes = {}
    for i, path in enumerate(files):
        obs, src = load_record(path)
        validate_observation(obs, src)
        if (src["run_id"], src["scene"], src["split"]) != (meta["run_id"], meta["scene"], meta["split"]):
            raise ValueError("observation provenance differs from manifest")
        seed = meta["seed"]+i
        torch.manual_seed(seed)
        initial = torch.randn(8, 24, 3, device="cuda:0")
        rng = torch.cuda.get_rng_state(0)
        cpu_rng = torch.get_rng_state()
        def predict():
            torch.cuda.set_rng_state(rng, 0)
            torch.set_rng_state(cpu_rng)
            return model.predict_pointgoal_action_with_guidance(
                obs["pointgoal"][None], obs["rgb"][None], obs["depth"][None], 8,
                np.array([src["valid_segment_len"]]), obs["prev_action"][None], 0, 23,
                np.array(src["guidance_factor"]), guidance_step=5,
                embodiment=src["embodiment"], initial_noise=initial)
        reference = predict()
        with teacher_tap(model) as tapped:
            actual = predict()
        for a, b in zip(reference[:3], actual[:3]):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            kwargs = dict(num_points=25, smooth_factor=.5, weight=model.weight)
            smooth = model.smooth_trajectory(tapped["raw"], **kwargs)
            qpath = model.smooth_cumulative_trajectory(torch.cumsum(tapped["raw"]/4, 1), **kwargs)
            torch.testing.assert_close(smooth, tapped["smoothed_actions"])
            torch.testing.assert_close(qpath, tapped["q_path"])
            q1, q2 = model.predict_pointgoal_q(qpath, tapped["rgbd"], tapped["goal"],
                                              is_target=False, embodiment=src["embodiment"])
            torch.testing.assert_close(q1, tapped["q1"])
            torch.testing.assert_close(q2, tapped["q2"])
            indices = (-(q1+q2)/2).argsort()[:2]
        def cpu(x):
            return x.detach().cpu().numpy()
        arrays = {"raw_action_deltas": cpu(tapped["raw"]), "smoothed_actions": cpu(smooth),
                  "q_path": cpu(qpath), "trajectories": actual[0][0], "scores": actual[1][0],
                  "q1": cpu(q1), "q2": cpu(q2), "top_indices": cpu(indices),
                  "top_trajectories": actual[2][0], "rgbd_embed": cpu(tapped["rgbd"])[0],
                  "goal_embed": cpu(tapped["goal"])[0], "initial_noise": cpu(initial),
                  "sampler_cuda_rng_state": cpu(rng), "sampler_cpu_rng_state": cpu_rng.numpy()}
        record_meta = {**src, "kind": "label", "rtc_enabled": False, "candidates": 8,
                       "teacher_sha256": meta["teacher_sha256"], "observation": path.name,
                       "observation_sha256": sha256(path), "seed": seed,
                       "target_space": "raw_pre_smoothing_action_deltas", "tap_equivalence": True,
                       "q_recompute_verified": True}
        validate_label(arrays, record_meta)
        save_record(destination / path.name, arrays, record_meta)
        label_hashes[path.name] = sha256(destination / path.name)
        print(f"[{i+1}/{len(files)}] {path.name}: tap/coordinates/Q PASS", flush=True)
    dump(destination / "COMPLETE.json", {"count": len(files), "status": "complete", "sha256": label_hashes})
    validate(args)


def validate(args):
    import numpy as np
    from FM_distillation.core.fm_data import load_record, validate_label, validate_observation, sha256
    run = Path(args.run).resolve()
    meta = manifest(run)
    directory = run / args.name
    complete = json.loads((directory / "COMPLETE.json").read_text())
    provenance = json.loads((directory / "label_manifest.json").read_text())
    files = sorted(directory.glob("*.npz"))
    if complete.get("status") != "complete" or not files or len(files) != complete["count"] or {p.name for p in files} != set(provenance["sources"]):
        raise ValueError("label shard incomplete or file set changed")
    if provenance["teacher_sha256"] != meta["teacher_sha256"]:
        raise ValueError("label manifest teacher mismatch")
    episodes, unique_steps, stuck = set(), set(), 0
    qvalues, backward = [], 0
    for path in files:
        if sha256(path) != complete["sha256"][path.name]:
            raise ValueError("label checksum mismatch")
        arrays, record = load_record(path)
        validate_label(arrays, record)
        if Path(record["observation"]).name != record["observation"]:
            raise ValueError("source must be a simple filename")
        source = run / "observations" / record["observation"]
        if sha256(source) != record["observation_sha256"] or sha256(source) != provenance["sources"][source.name]:
            raise ValueError("observation checksum mismatch")
        obs, src = load_record(source)
        validate_observation(obs, src)
        if record["teacher_sha256"] != meta["teacher_sha256"]:
            raise ValueError("teacher mismatch")
        for key in ("run_id", "scene", "split"):
            if record[key] != meta[key] or src[key] != meta[key]:
                raise ValueError(f"provenance mismatch: {key}")
        for key in ("episode_id", "step", "embodiment", "valid_segment_len", "stuck"):
            if record[key] != src[key]:
                raise ValueError(f"label/source mismatch: {key}")
        identifier = (record["episode_id"], record["step"])
        if identifier in unique_steps:
            raise ValueError("duplicate episode step")
        unique_steps.add(identifier)
        episodes.add(record["episode_id"])
        stuck += int(record["stuck"])
        qvalues.extend(arrays["scores"].tolist())
        backward += int((arrays["trajectories"][:, 5, 0] < -.05).sum())
    report = {"status": "PASS", "records": len(files), "episodes": len(episodes),
              "scene": meta["scene"], "split": meta["split"], "stuck_observations": stuck,
              "candidate_count": len(qvalues), "q_min": min(qvalues), "q_max": max(qvalues),
              "backward_at_waypoint5_count": backward,
              "warnings": ["no collision/success claims from Q or backward count",
                           "single scene / short capture does not establish coverage",
                           "not cleared for P2 until report reviewed"]}
    # A repeatable report may be refreshed; immutable samples are never overwritten.
    (directory / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "serve", "evaluate", "label", "validate"):
        p = sub.add_parser(command)
        p.add_argument("--run", required=True)
        if command in ("prepare", "serve", "evaluate"):
            p.add_argument("--physical-gpu",type=int,choices=(0,1),default=1)
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
    globals()[args.command](args)


if __name__ == "__main__":
    main()
