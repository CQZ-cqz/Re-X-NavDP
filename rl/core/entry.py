"""Shared helpers for the rl CLI entry points (scenes, scheduling, benchmarks)."""

from rexnavdp import BASE
ROOT = BASE  # x-navdp baseline directory

"""MPC-free direct tracker (ReX-NavDP) training on training-split scenes.

Mirrors ../rl/scripts/train_reactive.py but the policy directly commands [v, w]:
this script never imports or constructs BatchMPCController, so the direct
run has no Acados solver and no libhpipm dependency at runtime.
"""

import argparse
import json
import os
import random
from pathlib import Path
import socket
import subprocess
import sys
import time


def training_scene(cfg, index):
    from eval.environment import _scene_entries
    dataset_dir = getattr(cfg.environment, 'dataset_dir', None)
    if dataset_dir or getattr(cfg.environment, 'scene_split_file', None):
        cfg.environment.scene_split = 'train'
    entries = _scene_entries(cfg)
    if index < 0 or index >= len(entries):
        raise ValueError(f'Training scene index {index} outside 0..{len(entries)-1}')
    scene = entries[index]
    if dataset_dir:
        folder = Path(dataset_dir)/'pointgoal_start_pair'/scene['scene_name']
        safe = folder/'pointgoal_start_pair_samples_safe.npy'
        if safe.is_file():
            scene['pointgoal_path'] = str(safe)
    for key in ('usd_path', 'esdf_path', 'pointgoal_path'):
        if not scene.get(key) or not Path(scene[key]).exists():
            raise FileNotFoundError(f'Training scene {scene["scene_name"]}: missing {key}={scene.get(key)}')
    return scene


def load_navigable_bounds(ply_path, margin=1.0):
    """Return (min_xy, max_xy) of a navigable.ply ESDF point cloud, expanded by margin."""
    import re
    import numpy as np
    with open(ply_path, 'rb') as handle:
        header = b''
        while b'end_header' not in header:
            line = handle.readline()
            if not line:
                raise ValueError(f'Malformed PLY header in {ply_path}')
            header += line
        match = re.search(rb'element vertex (\d+)', header)
        if not match:
            raise ValueError(f'No vertex count in {ply_path}')
        count = int(match.group(1))
        binary = handle.read()
    if b'property uchar red' in header:
        dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    else:
        dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    vertices = np.frombuffer(binary, dtype=dtype, count=count)
    xy = np.stack((vertices['x'], vertices['y']), axis=-1)
    min_xy = (xy.min(axis=0) - margin).astype(np.float32).tolist()
    max_xy = (xy.max(axis=0) + margin).astype(np.float32).tolist()
    return min_xy, max_xy


"""Single-G1, one-control-step RGB-D residual PPO training on training-split scenes."""

import argparse
import json
import os
import random
from pathlib import Path
import socket
import subprocess
import sys
import time


def reactive_training_scene(cfg, index):
    from eval.environment import _scene_entries
    if not getattr(cfg.environment, 'dataset_dir', None):
        raise ValueError('Training requires metadata with an explicit train/eval scene split')
    cfg.environment.scene_split = 'train'
    entries = _scene_entries(cfg)
    if index < 0 or index >= len(entries):
        raise ValueError(f'Training scene index {index} outside 0..{len(entries)-1}')
    scene = entries[index]
    folder = Path(cfg.environment.dataset_dir)/'pointgoal_start_pair'/scene['scene_name']
    safe = folder/'pointgoal_start_pair_samples_safe.npy'
    if safe.is_file():
        scene['pointgoal_path'] = str(safe)
    for key in ('usd_path', 'esdf_path', 'pointgoal_path'):
        if not scene.get(key) or not Path(scene[key]).exists():
            raise FileNotFoundError(f'Training scene {scene["scene_name"]}: missing {key}={scene.get(key)}')
    return scene


"""Collect BC across all home_train scenes, then merge into one dataset.

Runs ../rl/scripts/collect_direct_bc.py once per scene (N episodes each) as separate
Isaac processes (one clean app per scene), is idempotent (skips scene .pt files
already present), and concatenates the per-scene episodes into a single .pt that
train_direct_bc.py consumes.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil

import torch



def scene_count():
    split_path = ROOT / 'data/scenes/scene_split.json'
    if split_path.is_file():
        with open(split_path) as handle:
            return len(json.load(handle).get('home_train', []))
    return 49


"""Sequential, resumable RGB-D direct-tracker training over a complete scene split.

Each scene is trained in a fresh Isaac process because the simulator cannot swap
USD scenes in-place reliably.  One PPO checkpoint is carried across scenes; only
the first scene is initialized from the behavior-cloning policy.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import signal
import socket
import subprocess
import sys
import time

import yaml

STATE_VERSION = 1


class StageCheckpointMissing(RuntimeError):
    """The tracker exited successfully without completing a PPO update."""


def resolve_root(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_train_scenes(scene_config):
    with open(scene_config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    env = config['environment']
    scene_type = env.get('scene_type') or 'home'
    scene_dir = resolve_root(env['scene_dir'])
    if not env.get('dataset_dir'):
        split_value = env.get('scene_split_file')
        if split_value:
            split_file = resolve_root(split_value)
            if not split_file.is_file():
                raise FileNotFoundError(f'Missing clutter split file: {split_file}')
            with open(split_file, encoding='utf-8') as handle:
                split_data = json.load(handle)
            key = f'{scene_type}_train'
            if key not in split_data:
                raise KeyError(f'Missing {key!r} in {split_file}')
            return list(split_data[key]), split_file
        # Self-contained clutter scenes have no split JSON. Ignore auxiliary
        # directories and train over folders with all three required assets.
        scene_names = []
        for directory in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            has_usd = any(directory.glob('*.usd'))
            has_pointgoal = any('pointgoal' in path.name.lower()
                                for path in directory.glob('*.npy'))
            if has_usd and has_pointgoal and (directory/'occupancy.ply').is_file():
                scene_names.append(directory.name)
        if not scene_names:
            raise FileNotFoundError(f'No complete clutter scenes found under {scene_dir}')
        return scene_names, scene_dir
    split_file = resolve_root(env.get('scene_split_file', scene_dir / 'scene_split.json'))
    if not split_file.is_file():
        dataset_dir = resolve_root(env['dataset_dir'])
        split_file = dataset_dir.parent / 'scene_split.json'
    with open(split_file, encoding='utf-8') as handle:
        split_data = json.load(handle)
    key = f'{scene_type}_train'
    if key not in split_data:
        raise KeyError(f'Missing {key!r} in {split_file}')
    return list(split_data[key]), split_file


def parse_scene_indices(spec, count):
    if spec is None:
        return list(range(count))
    result = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            first, last = (int(value) for value in item.split('-', 1))
            if last < first:
                raise ValueError(f'Invalid descending scene range {item!r}')
            result.extend(range(first, last + 1))
        else:
            result.append(int(item))
    result = list(dict.fromkeys(result))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f'--scene-indices must select values in 0..{count - 1}')
    return result


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    temporary.replace(path)


def update_latest_link(run_dir, checkpoint):
    target = run_dir / 'latest.pt'
    temporary = run_dir / '.latest.pt.tmp'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(checkpoint, run_dir))
    temporary.replace(target)


def final_metrics(directory):
    path = directory / 'metrics.jsonl'
    if not path.is_file():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def completed_stage(prefix):
    """Resolve a successful stage and reject silent Isaac/CUDA shutdowns."""
    candidates = sorted(prefix.parent.glob(prefix.name + '_*'),
                        key=lambda path: path.stat().st_mtime)
    if not candidates or not (candidates[-1]/'latest.pt').is_file():
        raise StageCheckpointMissing(
            f'Trainer exited without a checkpoint for prefix {prefix}')
    stage_dir = candidates[-1]
    metrics = final_metrics(stage_dir)
    if not metrics or 'iteration' not in metrics:
        raise StageCheckpointMissing(
            f'Trainer checkpoint has no completed iteration in {stage_dir}')
    return stage_dir, (stage_dir/'latest.pt').resolve(), metrics


def wait_for_listener_exit(host, port, timeout=30.0):
    """Wait for an earlier scene's policy server to stop accepting connections."""
    deadline = time.monotonic() + timeout
    while True:
        with socket.socket() as probe:
            probe.settimeout(.5)
            listening = probe.connect_ex((host, port)) == 0
        if not listening:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f'Policy-server port {host}:{port} is still actively listening after {timeout:.0f}s')
        time.sleep(.5)


def terminate_process_group(process_group, grace=5.0):
    """Reap a trainer's descendants even when the trainer itself was SIGKILLed."""
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_stage(command, timeout):
    # A separate session makes the tracker the process-group leader. Its policy
    # server inherits this group, so it can be reaped after an OOM SIGKILL.
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    previous_handlers = {}
    def stop_stage(signum, _frame):
        # The tracker starts a policy-server descendant. Forward an interrupt
        # to the whole independent process group before stopping the scheduler.
        terminate_process_group(process.pid)
        raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop_stage)
    try:
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(process.pid)
            process.wait()
            raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    terminate_process_group(process.pid)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def recover_partial_stage(run_dir, job_number, job, after_iteration):
    """Return the newest valid per-iteration checkpoint left by a failed attempt."""
    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in job['scene_name'])
    pattern = (f"{job_number:03d}_e{job['epoch']:02d}_scene"
               f"{job['scene_index']:02d}_{safe_name}_a*_*")
    best = None
    for directory in (run_dir/'stages').glob(pattern):
        checkpoint = directory/'latest.pt'
        metrics = final_metrics(directory)
        if not checkpoint.is_file() or not metrics or 'iteration' not in metrics:
            continue
        iteration = int(metrics['iteration'])
        if iteration > after_iteration and (best is None or iteration > best[0]):
            best = (iteration, directory.resolve(), checkpoint.resolve(), metrics)
    return best


def initial_job_iteration(state):
    if state.get('completed'):
        metrics = state['completed'][-1].get('metrics') or {}
        if 'iteration' in metrics:
            return int(metrics['iteration'])
    if state['settings']['initial_kind'] == 'bc':
        return 0
    # This path is used only when a brand-new full run starts from a PPO file.
    import torch
    checkpoint = torch.load(state['settings']['initial_checkpoint'], map_location='cpu', weights_only=False)
    return int(checkpoint['iteration'])


def new_state(args):
    default_scene = str(ROOT/'eval/config/eval_pointgoal/humanoid_internscene_home.yaml')
    scene_configs = [resolve_root(p) for p in (args.scene_config or [default_scene])]
    config = resolve_root(args.config)
    checkpoint = resolve_root(args.checkpoint)
    bc_init = resolve_root(args.bc_init) if args.bc_init else None
    resume_checkpoint = resolve_root(args.resume_checkpoint) if args.resume_checkpoint else None
    for label, path in (('tracker config', config), ('X-NavDP checkpoint', checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {label}: {path}')
    for scene_config in scene_configs:
        if not scene_config.is_file():
            raise FileNotFoundError(f'Missing scene config: {scene_config}')
    if bc_init and not bc_init.is_file():
        raise FileNotFoundError(f'Missing BC checkpoint: {bc_init}')
    if resume_checkpoint and not resume_checkpoint.is_file():
        raise FileNotFoundError(f'Missing PPO checkpoint: {resume_checkpoint}')
    if bool(bc_init) == bool(resume_checkpoint):
        raise ValueError('Select exactly one of --bc-init and --resume-checkpoint')

    # Load each config's train split, then interleave them round-robin so the
    # difficulty levels alternate (easy_0, hard_0, easy_1, hard_1, ...).
    per_config = []   # list of (config_path, [(scene_index, scene_name), ...])
    split_files = []
    for scene_config in scene_configs:
        scene_names, split_file = load_train_scenes(scene_config)
        selected = parse_scene_indices(args.scene_indices, len(scene_names))
        per_config.append((scene_config, [(index, scene_names[index]) for index in selected]))
        split_files.append(str(split_file))
    jobs = []
    for epoch in range(args.epochs):
        shuffled = []
        for scene_config, pairs in per_config:
            pairs_copy = pairs.copy()
            if args.shuffle_scenes:
                random.Random(args.seed + epoch).shuffle(pairs_copy)
            shuffled.append((scene_config, pairs_copy))
        max_len = max((len(pairs) for _, pairs in shuffled), default=0)
        for pos in range(max_len):
            for scene_config, pairs in shuffled:
                if pos < len(pairs):
                    scene_index, scene_name = pairs[pos]
                    jobs.append(dict(epoch=epoch, order_in_epoch=len(jobs),
                                     scene_index=scene_index, scene_name=scene_name,
                                     scene_config=str(scene_config)))
    initial_kind = 'bc' if bc_init else 'ppo'
    initial_checkpoint = bc_init or resume_checkpoint
    return {
        'format_version': STATE_VERSION,
        'status': 'ready',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scene_split_files': split_files,
        'scene_count_in_split': sum(len(pairs) for _, pairs in per_config),
        'selected_scene_count': sum(len(pairs) for _, pairs in per_config),
        'settings': {
            'scene_config': str(scene_configs[0]),
            'scene_configs': [str(c) for c in scene_configs],
            'config': str(config), 'checkpoint': str(checkpoint),
            'initial_kind': initial_kind, 'initial_checkpoint': str(initial_checkpoint),
            'device': args.device, 'port': args.port,
            'iterations_per_scene': args.iterations_per_scene,
            'rollout_steps': args.rollout_steps, 'num_envs': args.num_envs,
            'episode_seconds': args.episode_seconds,
            'success_distance': args.success_distance, 'seed': args.seed,
            'epochs': args.epochs, 'shuffle_scenes': args.shuffle_scenes,
            'max_job_attempts': args.max_job_attempts,
            'stage_timeout_seconds': args.stage_timeout_minutes * 60.0,
        },
        'jobs': jobs,
        'next_job': 0,
        'latest_checkpoint': None,
        'completed': [],
        'attempts': {},
    }


def print_plan(state, run_dir=None):
    settings = state['settings']
    transitions = (len(state['jobs']) * settings['iterations_per_scene'] *
                   settings['rollout_steps'] * settings['num_envs'])
    configs = settings.get('scene_configs', [settings['scene_config']])
    print(f"Scene configs ({len(configs)}): {', '.join(Path(c).name for c in configs)}")
    print(f"Train split: {state['scene_count_in_split']} scenes; selected: "
          f"{state['selected_scene_count']}; epochs: {settings['epochs']}")
    print(f"Jobs: {len(state['jobs'])}; PPO updates: "
          f"{len(state['jobs']) * settings['iterations_per_scene']}; "
          f"control transitions: {transitions}")
    print(f"Initial checkpoint ({settings['initial_kind']}): {settings['initial_checkpoint']}")
    if run_dir:
        print(f'Run directory: {run_dir}')
    preview = ', '.join(str(job['scene_name']) for job in state['jobs'][:12])
    suffix = ' ...' if len(state['jobs']) > 12 else ''
    print(f'Scene order: {preview}{suffix}')


"""Isolated RGB-D visual-encoder latency benchmark (plan §14).

Measures the frozen RGB-D feature backbone and the downstream trainable policy
(adapter + cross-attention + GRU) for one visual backend, reporting P50/P95/P99
and the 25 Hz control-deadline (40 ms) miss rate.

Run once per backend and compare the "visual total" rows:

    python ../rl/scripts/benchmark_visual_encoder.py --backend dav2 \
        --checkpoint checkpoints/x-navdp_posttrain.ckpt
    python ../rl/scripts/benchmark_visual_encoder.py --backend yolo26_depth \
        --weights checkpoints/yolo26n-depth.pt

Same GPU / batch / input resolution / warm-up / runs gives an apples-to-apples
comparison. The YOLO backend also emits a per-branch breakdown (RGB vs depth,
preprocess vs extraction).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


from rl.core.encoder import (build_rgbd_encoder, resolve_policy_visual_config,
                                  preprocess_yolo_rgb)
from rl.core.policy import PolicyConfig, ReactiveActorCritic


def percentile(samples, p):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))]


def run_timed(fn, runs, warmup, device):
    """Return per-call wall latencies in milliseconds.

    Wall clock (with a device synchronize) is used rather than CUDA events
    because DA-V2 preprocessing is CPU-bound (cv2) while YOLO preprocessing is
    GPU-bound; wall time captures the true end-to-end latency of both.
    """
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1e3)
    return samples


def summarize(name, samples, deadline_ms=40.0):
    return {
        "stage": name,
        "p50_ms": round(percentile(samples, 0.50), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "p99_ms": round(percentile(samples, 0.99), 3),
        "deadline_miss_pct": round(100.0 * sum(s > deadline_ms for s in samples) / len(samples), 2),
    }


def make_encoder_and_policy(args, device):
    if args.backend == "dav2":
        if not args.checkpoint:
            raise SystemExit("--backend dav2 requires --checkpoint (X-NavDP checkpoint)")
        encoder = build_rgbd_encoder({"type": "depth_anything_v2",
                                      "model": "depth-anything-v2-vits",
                                      "frozen": True}, args.checkpoint, device)
    elif args.backend == "yolo26_depth":
        if not args.weights:
            raise SystemExit("--backend yolo26_depth requires --weights (yolo26n-depth.pt)")
        visual = {"type": "yolo26_depth", "weights": args.weights, "model_name": "yolo26n-depth",
                  "feature_layer": "depth_head_p3", "image_size": args.size, "grid_size": args.grid_size}
        if args.compile_mode:
            visual["compile_mode"] = args.compile_mode
        encoder = build_rgbd_encoder(visual, None, device)
    else:
        raise SystemExit(f"unknown backend {args.backend!r}")
    policy_config = PolicyConfig(**resolve_policy_visual_config(dict(state_dim=args.state_dim,
                                                                    path_points=args.path_points), encoder))
    policy = ReactiveActorCritic(policy_config).to(device)
    return encoder, policy


def make_obs(features, policy_config, device, batch):
    leading = (batch,)
    return {
        "state": torch.zeros(*leading, policy_config.state_dim, device=device),
        "rgb_tokens": features["rgb_tokens"],
        "depth_tokens": features["depth_tokens"],
        "path": torch.zeros(*leading, policy_config.path_points, 4, device=device),
        "path_mask": torch.ones(*leading, policy_config.path_points, device=device),
    }

