"""Re-X-NavDP RL execution — unified command line interface."""

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_BASE = _ROOT / "baselines/x-navdp"
for _p in (_ROOT, _BASE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rexnavdp import BASE, ROOT as REPO_ROOT
ROOT = BASE  # x-navdp baseline directory
from rl.src.entry import (StageCheckpointMissing, STATE_VERSION, atomic_json,
                           completed_stage, final_metrics, initial_job_iteration,
                           load_navigable_bounds, load_train_scenes, make_encoder_and_policy,
                           make_obs, new_state, parse_scene_indices, percentile, print_plan,
                           reactive_training_scene, recover_partial_stage, resolve_root,
                           run_stage, run_timed, scene_count, summarize,
                           terminate_process_group, training_scene, update_latest_link,
                           wait_for_listener_exit)

import argparse
import json
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import yaml

# The policy-server subprocess (`python -m eval.src.policy_server`) imports
# `bridge`/`ddim` from the repo root; propagate the repo root and baseline via
# PYTHONPATH so the child resolves them without re-running the bootstrap.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(BASE), os.environ.get("PYTHONPATH", "")) if p)

def cmd_collect():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config', default=str(ROOT/'eval/config/eval_pointgoal/humanoid_internscene_home.yaml'))
    parser.add_argument('--config', default=str(ROOT/'../../rl/config/reactive_rgbd_direct_g1.yaml'))
    parser.add_argument('--checkpoint', default=str(REPO_ROOT/'checkpoints/x-navdp_posttrain.ckpt'))
    parser.add_argument('--scene-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=20004)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--episodes-per-scene', type=int, default=None,
                        help='Stop after this many episodes (takes precedence over --steps).')
    parser.add_argument('--output', default=str(ROOT/'outputs/direct_bc_data.pt'))
    parser.add_argument('--resume', help='Resume from a .resume checkpoint written during collection.')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--legacy-blanket-scale', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    os.environ.setdefault('ISAAC_ACTIVE_GPU', '1')
    os.environ.setdefault('ISAAC_PHYSICS_GPU', '0')
    os.environ.pop('DISPLAY', None)
    os.environ.setdefault('ACADOS_SOURCE_DIR', os.path.expanduser("~/acados"))  # teacher uses MPC
    # libacados.so links libhpipm/libblasfeo; a runtime os.environ LD_LIBRARY_PATH
    # change is ignored by the dynamic loader, so preload the deps by full path
    # (deps first) before the MPC solver dlopen's libacados.so.
    import ctypes as _ctypes
    for _lib in ('libblasfeo.so', 'libhpipm.so', 'libacados.so'):
        _ctypes.CDLL(str(Path(os.environ['ACADOS_SOURCE_DIR']) / 'lib' / _lib))
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    from eval.config_utils import load_default_config
    from eval.environment import namespace_to_dict
    cfg = load_default_config(args.scene_config)
    if cfg.environment.embodiment != 'unitree_g1':
        raise ValueError('BC collection is G1 only')
    scene = training_scene(cfg, args.scene_index)
    checkpoint = str(Path(args.checkpoint).resolve(strict=True))
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    min_xy, max_xy = load_navigable_bounds(scene['esdf_path'])
    config['navigation_training']['oob_min'] = min_xy
    config['navigation_training']['oob_max'] = max_xy
    config['x_navdp_checkpoint'] = checkpoint
    directory = Path(args.output).parent
    directory.mkdir(parents=True, exist_ok=True)

    from src.training.worker import configure_mdl_system_path
    configure_mdl_system_path(cfg.environment.scene_dir)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    server_log = open(directory/'bc_policy_server.log', 'w')
    process = subprocess.Popen([sys.executable, '-u', '-m', 'eval.src.policy_server',
        '--port', str(args.port), '--embodiment', 'humanoid', '--checkpoint', checkpoint,
        '--device', args.device, '--no-visualization', '--recovery-mode', 'baseline', '--seed', str(args.seed)],
        cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    app = env = backend = None
    try:
        deadline = time.monotonic()+90.
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Planner server exited; see {directory}/bc_policy_server.log')
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=1.):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Planner server startup timed out')
                time.sleep(.5)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app
        from src.environment import create_dingoeval_environment
        from src.utils import BatchMPCController
        from rl.src.encoder import build_rgbd_encoder, resolve_policy_visual_config
        from rl.src.policy import PolicyConfig, ReactiveActorCritic
        from rl.src.train_env import DirectReactiveTrainEnv, AsyncPlanner
        from rl.src.isaac_backend import IsaacReactiveBackend
        from eval.src.client_utils import navigator_reset, pointgoal_step
        env, controller = create_dingoeval_environment(cfg.environment.scene_dir, 0, 1,
            scene_scale=getattr(cfg.environment, 'scene_scale', None), device=args.device,
            embodiment='unitree_g1', scene_data=scene, controller_config=namespace_to_dict(cfg.controller), seed=args.seed,
            preserve_blanket_scale=not args.legacy_blanket_scale)
        backend = IsaacReactiveBackend(env, controller, config['navigation_training'])
        mpc = BatchMPCController(batch=1, **namespace_to_dict(cfg.mpc))
        def reset_planner():
            intrinsic = env.unwrapped.scene['camera_sensor'].data.intrinsic_matrices[0].cpu().numpy()
            navigator_reset(intrinsic, batch_size=1, port=args.port, scene_name=scene['scene_name'],
                sample_indices=[int(env.unwrapped._sample_idx[0])])
        def predict(state):
            return pointgoal_step(state['goal_xy'], state['rgb'], state['depth'], port=args.port,
                robot_pos=state['position'], robot_quat=state['quaternion'])[0]
        planner = AsyncPlanner(predict, reset_planner)
        encoder = build_rgbd_encoder(config.get("visual_encoder"), checkpoint, args.device, training=True)
        config["policy"] = resolve_policy_visual_config(config["policy"], encoder)
        adapter = DirectReactiveTrainEnv(backend, encoder, planner, config, args.device)
        adapter.reset()  # warm the initial plan + RGB-D

        episodes = []
        cur = {k: [] for k in ('rgb_tokens', 'depth_tokens', 'path', 'path_mask', 'state', 'teacher')}
        recorded = 0
        start_step = 0
        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            episodes = ckpt['episodes']
            recorded = int(ckpt.get('recorded', 0))
            start_step = int(ckpt.get('control_steps', 0))
            print(f'Resumed: {recorded} recorded steps across {len(episodes)} episodes, '
                  f'continuing from control step {start_step}', flush=True)
        episodes_collected = 0
        def flush():
            nonlocal cur, episodes_collected
            if len(cur['teacher']):
                episodes.append({k: torch.stack(v) for k, v in cur.items()})
                cur = {k: [] for k in cur}
                episodes_collected += 1
        for step in range(start_step, args.steps):
            if args.episodes_per_scene is not None and episodes_collected >= args.episodes_per_scene:
                break
            prepared = adapter.prepared
            obs = prepared['observation']
            valid_path = prepared['valid_path']
            path = obs['path'].cpu().numpy()
            mask = obs['path_mask'].cpu().numpy()
            nominal = np.zeros((1, 2), dtype=np.float32)
            ok = np.zeros(1, dtype=bool)
            if valid_path[0]:
                points = path[0, mask[0].astype(bool), :2]
                if len(points) > 1:
                    reference = np.zeros((1, len(points)+1, 3), dtype=np.float32)
                    reference[0, 1:, :2] = points
                    nominal[0] = mpc.solve(reference)[0][0, 0]
                    ok[0] = int(mpc.last_statuses[0]) == 0 and np.isfinite(nominal[0]).all()
            if ok[0]:
                cur['rgb_tokens'].append(obs['rgb_tokens'][0].cpu())
                cur['depth_tokens'].append(obs['depth_tokens'][0].cpu())
                cur['path'].append(obs['path'][0].cpu())
                cur['path_mask'].append(obs['path_mask'][0].cpu())
                cur['state'].append(obs['state'][0].cpu())
                cur['teacher'].append(torch.from_numpy(nominal[0]))
                recorded += 1
            else:
                flush()
            adapter.control.apply_command(nominal if ok[0] else np.zeros((1, 2), np.float32))
            following, terminated, truncated, _ = backend.step(nominal if ok[0] else np.zeros((1, 2), np.float32))
            if (terminated | truncated).astype(bool).any():
                flush()
                adapter._clear(following['now'])
                adapter.state = following
                adapter.control.reset_time[:] = -float('inf')
                adapter.prepared = adapter._prepare(following, warm=True)
            else:
                adapter.state = following
                adapter.prepared = adapter._prepare(following)
            if (recorded % 2000) < 1 and recorded:
                print(f'collected {recorded} steps', flush=True)
                torch.save({'episodes': episodes, 'recorded': recorded, 'control_steps': step+1,
                    'encoder_fingerprint': encoder.fingerprint, 'visual_encoder_metadata': encoder.metadata, 'control_dt': backend.dt},
                    args.output + '.resume')
        flush()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'episodes': episodes, 'encoder_fingerprint': encoder.fingerprint,
            'visual_encoder_metadata': encoder.metadata, 'control_dt': backend.dt, 'steps': args.steps, 'recorded': recorded,
            'num_episodes': len(episodes)}, args.output)
        print(f'Saved {recorded} recorded steps across {len(episodes)} episodes to {args.output}', flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            if 'adapter' in locals() and adapter is not None:
                adapter.close()
            elif backend is not None:
                backend.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            server_log.close()
            if app is not None:
                app.close()


def cmd_collect_all():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config',
                        default=str(ROOT / 'eval/config/eval_pointgoal/humanoid_internscene_home.yaml'))
    parser.add_argument('--checkpoint', default=str(REPO_ROOT / 'checkpoints/x-navdp_posttrain.ckpt'))
    parser.add_argument('--episodes-per-scene', type=int, default=10)
    parser.add_argument('--num-scenes', type=int, default=None,
                        help='Number of home_train scenes (default: read from scene_split.json)')
    parser.add_argument('--start', type=int, default=0, help='First scene index to collect')
    parser.add_argument('--max-failures', type=int, default=3,
                        help='Stop after this many consecutive scene failures (0 = never stop)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=20004)
    parser.add_argument('--output', default=str(ROOT / 'outputs/direct_bc_yolo.pt'))
    parser.add_argument('--tmp-dir', default=str(ROOT / 'outputs/direct_bc_yolo_scenes'))
    args = parser.parse_args()

    num_scenes = args.num_scenes or scene_count()
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    episodes, encoder_fingerprint, encoder_metadata, control_dt = [], None, None, None
    consecutive_failures = 0
    for i in range(args.start, num_scenes):
        out = tmp_dir / f'scene_{i:02d}.pt'
        if not out.is_file():
            # Free any policy_server orphaned by a previous SIGKILLed scene; it would
            # otherwise hold --port and make the next probe fail with EADDRINUSE.
            subprocess.run(['pkill', '-f', 'eval.src.policy_server'], cwd=ROOT)
            # Isaac's texture cache balloons to tens of GB over many scenes and its
            # launch-time GC spikes RAM past the OOM threshold; clear it per scene.
            shutil.rmtree(Path.home() / '.cache/ov/texturecache', ignore_errors=True)
            cmd = [sys.executable, '-u', str(ROOT / '../../rl/cli.py'), 'collect',
                   '--scene-config', args.scene_config, '--checkpoint', args.checkpoint,
                   '--scene-index', str(i), '--episodes-per-scene', str(args.episodes_per_scene),
                   '--device', args.device, '--port', str(args.port), '--output', str(out)]
            print(f'[scene {i}/{num_scenes}] running: {Path(cmd[1]).name} --scene-index {i}', flush=True)
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0 or not out.is_file():
                consecutive_failures += 1
                print(f'[scene {i}] FAILED (rc={result.returncode}); '
                      f'{consecutive_failures} consecutive failures', flush=True)
                if args.max_failures and consecutive_failures >= args.max_failures:
                    print(f'Stopping after {consecutive_failures} consecutive failures; '
                          f'inspect the log and re-run to resume from scene {i+1}.', flush=True)
                    break
                continue
            consecutive_failures = 0
        data = torch.load(out, map_location='cpu', weights_only=False)
        n = len(data.get('episodes', []))
        episodes.extend(data['episodes'])
        if encoder_fingerprint is None:
            encoder_fingerprint = data.get('encoder_fingerprint')
            encoder_metadata = data.get('visual_encoder_metadata')
            control_dt = data.get('control_dt')
        print(f'[scene {i}] +{n} episodes (running total {len(episodes)})', flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'episodes': episodes, 'encoder_fingerprint': encoder_fingerprint,
                'visual_encoder_metadata': encoder_metadata, 'control_dt': control_dt,
                'num_episodes': len(episodes), 'num_scenes': num_scenes}, args.output)
    print(f'Merged {len(episodes)} episodes across {num_scenes} scenes -> {args.output}', flush=True)


def cmd_train_bc():
    import numpy as np
    import torch
    from rl.src.policy import PolicyConfig, ReactiveActorCritic
    from rl.src.runner import CONTROL_MODE_DIRECT, ACTION_MAPPING_DIRECT, warm_start_policy
    from rl.src.observation import DIRECT_STATE_VERSION
    from rl.src.encoder import PREPROCESS_VERSION
    from rl.src.bc import train_bc, evaluate_bc_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'../../rl/config/reactive_rgbd_direct_g1.yaml'))
    parser.add_argument('--data', required=True, help='BC dataset .pt from collect_direct_bc.py')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--output', default=str(ROOT/'outputs/direct_bc.pt'))
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--init', help='Initialize network weights from an existing BC policy.')
    initial.add_argument('--resume', help='Resume a new-format interrupted BC run, including Adam state.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.chunk < 1 or args.batch_size < 1 or not (0. <= args.val_fraction < 1.):
        parser.error('epochs/chunk/batch-size/val-fraction out of range')

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    dataset = torch.load(args.data, map_location='cpu', weights_only=False, mmap=True)
    episodes = dataset['episodes']
    encoder_fingerprint = dataset['encoder_fingerprint']
    encoder_metadata = dataset.get('visual_encoder_metadata')
    if encoder_metadata is not None:
        config['policy']['token_dim'] = encoder_metadata['visual_token_dim']
        config['policy']['grid_size'] = encoder_metadata['visual_grid_size']
    if not episodes:
        raise ValueError('BC dataset has no episodes')
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes)*args.val_fraction))
    val_idx = set(order[:n_val].tolist())
    train = [e for i, e in enumerate(episodes) if i not in val_idx]
    val = [e for i, e in enumerate(episodes) if i in val_idx]

    policy_config = PolicyConfig(**config['policy'])
    start_epoch, best_mse = 0, float('inf')
    if args.init:
        from rl.src.runner import warm_start_policy
        policy = warm_start_policy(args.init, policy_config, encoder_metadata or encoder_fingerprint,
            args.device, CONTROL_MODE_DIRECT)
    else:
        policy = ReactiveActorCritic(policy_config).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        policy.load_state_dict(checkpoint['policy'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint['bc_epoch'])
        best_mse = float(checkpoint.get('bc_best_val_mse', best_mse))
    initial_metrics = evaluate_bc_metrics(policy, val, args.device)
    print(f"initial held-out command MSE: {initial_metrics['physical_mse']:.6f}; "
          f"std(v,w)=({initial_metrics['prediction_std_v']:.4f},"
          f"{initial_metrics['prediction_std_w']:.4f})", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_output = output.with_name(output.stem + '.last.pt')
    def save(path, epoch, loss, metrics):
        payload = {'format_version': 1, 'policy_config': asdict(policy.config),
            'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
            'control_mode': CONTROL_MODE_DIRECT, 'state_version': DIRECT_STATE_VERSION,
            'action_mapping_version': ACTION_MAPPING_DIRECT,
            'preprocess_version': (encoder_metadata or {}).get('preprocess_version', PREPROCESS_VERSION), 'encoder_fingerprint': encoder_fingerprint,
            'visual_encoder_metadata': encoder_metadata,
            'bc_epoch': epoch, 'bc_best_val_mse': best_mse,
            'bc_train_mse': float(loss), 'bc_val_mse': float(metrics['physical_mse']),
            'bc_val_latent_mse': float(metrics['latent_mse']),
            'zero_actor_val_mse': float(initial_metrics['physical_mse']),
            'bc_metrics': metrics, 'bc_args': vars(args)}
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary); temporary.replace(path)

    if start_epoch == 0:
        # Preserve a strong --init policy if fine-tuning makes validation worse.
        best_mse = initial_metrics['physical_mse']
        save(output, 0, initial_metrics['physical_mse'], initial_metrics)
    loss = float('nan')
    for epoch in range(start_epoch, args.epochs):
        before = policy.actor[-1].weight.detach().clone()
        loss = train_bc(policy, train, optimizer, chunk=args.chunk,
            batch_size=args.batch_size, device=args.device, rng=rng)
        metrics = evaluate_bc_metrics(policy, val, args.device)
        update = float(torch.linalg.vector_norm(policy.actor[-1].weight-before))
        improved = metrics['physical_mse'] < best_mse
        if improved:
            best_mse = metrics['physical_mse']
        save(last_output, epoch+1, loss, metrics)
        if improved:
            save(output, epoch+1, loss, metrics)
        print(f"epoch {epoch+1}/{args.epochs}: train command MSE {loss:.6f}, "
              f"held-out {metrics['physical_mse']:.6f}, "
              f"corr(v,w)=({metrics['correlation_v']:.3f},{metrics['correlation_w']:.3f}), "
              f"actor update {update:.5f}{' [best]' if improved else ''}", flush=True)
    print(f'Saved best BC policy to {output} (held-out command MSE {best_mse:.6f})', flush=True)


def cmd_train_tracker():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config', default=str(ROOT/'eval/config/eval_pointgoal/humanoid_internscene_home.yaml'))
    parser.add_argument('--config', default=str(ROOT/'../../rl/config/reactive_rgbd_direct_g1.yaml'))
    parser.add_argument('--checkpoint', default=str(REPO_ROOT/'checkpoints/x-navdp_posttrain.ckpt'))
    parser.add_argument('--scene-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=20003)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--num-envs', type=int, default=1,
        help='Parallel G1 environments (vectorized direct training).')
    parser.add_argument('--episode-seconds', type=float)
    parser.add_argument('--success-distance', type=float,
        help='Curriculum success radius in metres (stage 1: 0.30, final: 0.15).')
    parser.add_argument('--output', default=str(ROOT/'outputs/direct_train'),
        help='Run prefix; a YYYYMMDD_HHMMSS timestamp is appended to make each run unique.')
    parser.add_argument('--resume')
    parser.add_argument('--bc-init', help='BC warm-start checkpoint from train_direct_bc.py (fresh PPO optimizer).')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--legacy-blanket-scale', action='store_true',
        help='Use legacy additional .01 blanket scaling (can produce invalid PhysX shapes).')
    args = parser.parse_args()
    if (args.iterations < 1 or args.rollout_steps < 2 or args.num_envs < 1 or
            (args.episode_seconds is not None and args.episode_seconds <= 0) or
            (args.success_distance is not None and args.success_distance <= 0)):
        parser.error('Iterations, rollout steps, num-envs and episode duration must be positive')
    # Headless + GPU-1 defaults (shell overrides win); drop DISPLAY to avoid the GLX hang.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    os.environ.setdefault('ISAAC_ACTIVE_GPU', '1')
    os.environ.setdefault('ISAAC_PHYSICS_GPU', '0')
    # Set before importing torch. Isaac and recurrent PPO have alternating
    # allocation sizes; expandable segments reduce allocator fragmentation.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.pop('DISPLAY', None)
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    from eval.config_utils import load_default_config
    from eval.environment import namespace_to_dict
    cfg = load_default_config(args.scene_config)
    if cfg.environment.embodiment != 'unitree_g1':
        raise ValueError('Initial training adapter is G1 only')
    scene = training_scene(cfg, args.scene_index)
    checkpoint = str(Path(args.checkpoint).resolve(strict=True))
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    # OOB uses the scene's actual navigable bounding box, not a radius from origin.
    min_xy, max_xy = load_navigable_bounds(scene['esdf_path'])
    config['navigation_training']['oob_min'] = min_xy
    config['navigation_training']['oob_max'] = max_xy
    if args.episode_seconds is not None:
        config['navigation_training']['episode_seconds'] = args.episode_seconds
    if args.success_distance is not None:
        config['navigation_training']['success_distance_m'] = args.success_distance
    config['training'].update(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    # RSL-RL recurrent batches split by environment, so choose the largest
    # requested divisor rather than silently forcing every run to one batch.
    requested_batches = min(config['ppo']['num_mini_batches'], args.num_envs)
    config['ppo']['num_mini_batches'] = max(
        n for n in range(1, requested_batches+1) if args.num_envs % n == 0)
    config['x_navdp_checkpoint'] = checkpoint
    directory = Path(args.output + time.strftime('_%Y%m%d_%H%M%S', time.localtime()))
    directory.mkdir(parents=True, exist_ok=False)
    with open(directory/'run.json', 'w') as handle:
        json.dump(dict(args=vars(args), config=config, scene=scene, scene_split='train'), handle, indent=2)
    from src.training.worker import configure_mdl_system_path
    configure_mdl_system_path(cfg.environment.scene_dir)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    # Own this server explicitly. Never reuse or stop the user's existing port 20001 server.
    # SO_REUSEADDR lets the next scene pass this availability probe while the
    # previous server's accepted connections are still in TCP TIME_WAIT. An
    # active listener on the same address/port still makes bind() fail.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    server_log = open(directory/'policy_server.log', 'w')
    process = subprocess.Popen([sys.executable, '-u', '-m', 'eval.src.policy_server',
        '--port', str(args.port), '--embodiment', 'humanoid', '--checkpoint', checkpoint,
        '--device', args.device, '--no-visualization', '--recovery-mode', 'baseline', '--seed', str(args.seed)],
        cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    app = env = adapter = None
    try:
        deadline = time.monotonic()+90.
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Planner server exited; see {directory}/policy_server.log')
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=1.):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Planner server startup timed out')
                time.sleep(.5)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app
        from src.environment import create_dingoeval_environment
        from rl.src.encoder import build_rgbd_encoder, resolve_policy_visual_config
        from rl.src.policy import PolicyConfig, ReactiveActorCritic
        from rl.src.runner import ReactiveRunner, warm_start_policy, CONTROL_MODE_DIRECT
        from rl.src.train_env import DirectReactiveTrainEnv, AsyncPlanner
        from rl.src.isaac_backend import IsaacReactiveBackend
        from eval.src.client_utils import navigator_reset, pointgoal_step
        env, controller = create_dingoeval_environment(cfg.environment.scene_dir, 0, args.num_envs,
            scene_scale=getattr(cfg.environment, 'scene_scale', None), device=args.device,
            embodiment='unitree_g1', scene_data=scene, controller_config=namespace_to_dict(cfg.controller), seed=args.seed,
            preserve_blanket_scale=not args.legacy_blanket_scale)
        backend = IsaacReactiveBackend(env, controller, config['navigation_training'])
        def reset_planner():
            intrinsic = env.unwrapped.scene['camera_sensor'].data.intrinsic_matrices[0].cpu().numpy()
            navigator_reset(intrinsic, batch_size=args.num_envs, port=args.port, scene_name=scene['scene_name'],
                sample_indices=[int(env.unwrapped._sample_idx[i]) for i in range(args.num_envs)])
        def predict(state):
            return pointgoal_step(state['goal_xy'], state['rgb'], state['depth'], port=args.port,
                robot_pos=state['position'], robot_quat=state['quaternion'])[0]
        planner = AsyncPlanner(predict, reset_planner)
        encoder = build_rgbd_encoder(config.get("visual_encoder"), checkpoint, args.device, training=True)
        config["policy"] = resolve_policy_visual_config(config["policy"], encoder)
        with open(directory/'run.json') as handle:
            manifest = json.load(handle)
        manifest['encoder_fingerprint'] = encoder.fingerprint
        manifest['visual_encoder_metadata'] = encoder.metadata
        manifest['config']['policy'] = config['policy']
        manifest['control_dt'] = backend.dt
        manifest['control_mode'] = CONTROL_MODE_DIRECT
        manifest['preserve_blanket_scale'] = not args.legacy_blanket_scale
        with open(directory/'run.json', 'w') as handle:
            json.dump(manifest, handle, indent=2)
        # NOTE: no MPC solver is constructed here; the direct tracker has no Acados dependency.
        adapter = DirectReactiveTrainEnv(backend, encoder, planner, config, args.device, directory/"transitions.csv")
        if args.bc_init:
            policy = warm_start_policy(args.bc_init, PolicyConfig(**config['policy']),
                encoder.metadata, args.device, CONTROL_MODE_DIRECT)
        else:
            policy = ReactiveActorCritic(PolicyConfig(**config['policy'])).to(args.device)
        runner = ReactiveRunner(policy, args.num_envs, steps=args.rollout_steps,
            ppo_config=config['ppo'], encoder_fingerprint=encoder.fingerprint, encoder_metadata=encoder.metadata,
            control_mode=CONTROL_MODE_DIRECT)
        if args.resume:
            runner.load(args.resume)
        with open(directory/'metrics.jsonl', 'w') as log:
            for _ in range(args.iterations):
                started = time.perf_counter()
                before = policy.actor[-1].weight.detach().clone()
                metrics = runner.collect_and_update(adapter)
                metrics.update(adapter.metrics(), wall_s=time.perf_counter()-started,
                    actor_update_norm=float(torch.linalg.vector_norm(policy.actor[-1].weight-before)))
                if any(p.grad is not None for p in encoder.parameters()):
                    raise RuntimeError('Frozen encoder unexpectedly acquired gradients')
                line = json.dumps(metrics)
                print(line, flush=True); log.write(line+'\n'); log.flush()
                runner.save(directory/'latest.pt')
        print(f'Direct training completed: {directory}/latest.pt', flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()  # Isaac shutdown can terminate the interpreter before an exception is displayed.
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
            elif env is not None:
                env.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            server_log.close()
            if app is not None:
                app.close()


def cmd_train_full():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config', action='append', default=None,
                        help='Scene config(s). Pass twice (e.g. easy + hard) to interleave two splits.')
    parser.add_argument('--config', default=str(ROOT/'../../rl/config/reactive_rgbd_direct_g1.yaml'))
    parser.add_argument('--checkpoint', default=str(REPO_ROOT/'checkpoints/x-navdp_posttrain.ckpt'))
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--bc-init')
    initial.add_argument('--resume-checkpoint')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--iterations-per-scene', type=int, default=5)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--scene-indices', help='Optional comma/range subset, e.g. 0-9,12; default is the full train split.')
    parser.add_argument('--shuffle-scenes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--episode-seconds', type=float)
    parser.add_argument('--success-distance', type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=20003)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-job-attempts', type=int, default=5,
        help='Retry a failed scene subprocess this many times before stopping the full run.')
    parser.add_argument('--stage-timeout-minutes', type=float, default=60.0,
        help='Kill and retry a scene that makes no exit within this wall-clock limit.')
    parser.add_argument('--output-root', default=str(ROOT/'outputs/direct_rgbd_full'))
    parser.add_argument('--resume-run', help='Existing full-run directory containing state.json.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.resume_run and not args.bc_init and not args.resume_checkpoint:
        args.bc_init = str(ROOT/'outputs/direct_bc.pt')
    if min(args.epochs, args.iterations_per_scene, args.rollout_steps,
           args.num_envs, args.max_job_attempts, args.stage_timeout_minutes) < 1:
        parser.error('epochs, iterations-per-scene, rollout-steps, num-envs, attempts and timeout must be positive')

    if args.resume_run:
        run_dir = resolve_root(args.resume_run)
        with open(run_dir/'state.json', encoding='utf-8') as handle:
            state = json.load(handle)
        if state.get('format_version') != STATE_VERSION:
            raise ValueError(f'Unsupported state format in {run_dir}')
    else:
        state = new_state(args)
        if args.dry_run:
            print_plan(state)
            return
        output_root = resolve_root(args.output_root)
        run_dir = Path(str(output_root) + time.strftime('_%Y%m%d_%H%M%S'))
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir/'stages').mkdir()
        atomic_json(run_dir/'state.json', state)

    print_plan(state, run_dir)
    if args.dry_run:
        print(f"Remaining jobs: {len(state['jobs']) - state['next_job']}")
        return

    settings = state['settings']
    while state['next_job'] < len(state['jobs']):
        job_number = state['next_job']
        job = state['jobs'][job_number]
        key = str(job_number)
        progress_map = state.setdefault('job_progress', {})
        if key not in progress_map:
            start_iteration = initial_job_iteration(state)
            progress_map[key] = {
                'start_iteration': start_iteration,
                'target_iteration': start_iteration + int(settings['iterations_per_scene']),
                'iteration': start_iteration,
                'checkpoint': state['latest_checkpoint'] or settings['initial_checkpoint'],
            }
        progress = progress_map[key]
        recovered = recover_partial_stage(run_dir, job_number, job, int(progress['iteration']))
        if recovered is not None:
            iteration, stage_dir, checkpoint, metrics = recovered
            progress.update(iteration=iteration, checkpoint=str(checkpoint),
                recovered_stage_dir=str(stage_dir))
            state['latest_checkpoint'] = str(checkpoint)
            update_latest_link(run_dir, checkpoint)
        remaining_iterations = int(progress['target_iteration']) - int(progress['iteration'])
        if remaining_iterations <= 0:
            recovered_dir = Path(progress['recovered_stage_dir'])
            recovered_metrics = final_metrics(recovered_dir)
            state['completed'].append(dict(job_number=job_number,
                attempt=int(state['attempts'].get(key, 0)), stage_dir=str(recovered_dir),
                checkpoint=progress['checkpoint'], num_envs=None, recovered_partial=True,
                metrics=recovered_metrics, **job))
            state['next_job'] += 1
            state['in_progress'] = None
            state['status'] = 'complete' if state['next_job'] == len(state['jobs']) else 'ready'
            atomic_json(run_dir/'state.json', state)
            continue

        attempt = int(state['attempts'].get(key, 0)) + 1
        state['attempts'][key] = attempt
        state['status'] = 'running'
        state['in_progress'] = dict(job_number=job_number, attempt=attempt,
            remaining_iterations=remaining_iterations, **job)
        atomic_json(run_dir/'state.json', state)

        safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in job['scene_name'])
        prefix = run_dir/'stages'/(
            f"{job_number:03d}_e{job['epoch']:02d}_scene{job['scene_index']:02d}_{safe_name}_a{attempt:02d}")
        oom_retries = state.setdefault('oom_retries', {})
        if key not in oom_retries:
            previous = state.get('last_error') or {}
            oom_retries[key] = int(previous.get('job_number') == job_number and
                previous.get('returncode') == -signal.SIGKILL)
        job_num_envs = max(1, int(settings['num_envs']) // (2 ** int(oom_retries[key])))
        command = [sys.executable, '-u', str(ROOT/'../../rl/cli.py'), 'train-tracker',
            '--scene-config', job.get('scene_config', settings['scene_config']),
            '--config', settings['config'],
            '--checkpoint', settings['checkpoint'], '--scene-index', str(job['scene_index']),
            '--device', settings['device'], '--port', str(settings['port']),
            '--iterations', str(remaining_iterations),
            '--rollout-steps', str(settings['rollout_steps']), '--num-envs', str(job_num_envs),
            '--output', str(prefix), '--seed', str(settings['seed'])]
        if settings.get('episode_seconds') is not None:
            command.extend(['--episode-seconds', str(settings['episode_seconds'])])
        if settings.get('success_distance') is not None:
            command.extend(['--success-distance', str(settings['success_distance'])])
        source_checkpoint = progress['checkpoint']
        if state['latest_checkpoint'] or settings['initial_kind'] == 'ppo':
            command.extend(['--resume', source_checkpoint])
        else:
            command.extend(['--bc-init', source_checkpoint])

        print(f"\n[{job_number + 1}/{len(state['jobs'])}] epoch {job['epoch'] + 1}, "
              f"scene {job['scene_index']} ({job['scene_name']}), attempt {attempt}", flush=True)
        print(shlex.join(command), flush=True)
        wait_for_listener_exit('127.0.0.1', int(settings['port']))
        try:
            run_stage(command, float(settings.get('stage_timeout_seconds', 1800.0)))
            stage_dir, checkpoint, metrics = completed_stage(prefix)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                StageCheckpointMissing) as error:
            max_attempts = int(settings.get('max_job_attempts', 5))
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            missing_checkpoint = isinstance(error, StageCheckpointMissing)
            returncode = (error.returncode
                          if isinstance(error, subprocess.CalledProcessError) else None)
            state['last_error'] = {
                'job_number': job_number, 'attempt': attempt,
                'returncode': returncode, 'timed_out': timed_out,
                'kind': ('missing_checkpoint' if missing_checkpoint else
                         'timeout' if timed_out else 'subprocess_exit'),
                'message': str(error),
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            # Isaac may swallow a Python CUDA OOM during shutdown and return 0.
            # A missing checkpoint is therefore treated like a memory failure:
            # the retry halves vector environments if the smaller mini-batch
            # alone is still insufficient. A wall-clock timeout does not imply
            # memory pressure and must retain the configured vector env count.
            if returncode == -signal.SIGKILL or missing_checkpoint:
                oom_retries[key] = int(oom_retries[key]) + 1
            recovered = recover_partial_stage(run_dir, job_number, job, int(progress['iteration']))
            if recovered is not None:
                iteration, recovered_dir, checkpoint, metrics = recovered
                progress.update(iteration=iteration, checkpoint=str(checkpoint),
                    recovered_stage_dir=str(recovered_dir))
                state['latest_checkpoint'] = str(checkpoint)
                update_latest_link(run_dir, checkpoint)
            state['status'] = 'retrying' if attempt < max_attempts else 'failed'
            atomic_json(run_dir/'state.json', state)
            if attempt >= max_attempts:
                raise
            reason = ('no completed checkpoint' if missing_checkpoint else
                      'timeout' if timed_out else f'exit code {returncode}')
            print(f'Scene subprocess failed with {reason}; retrying in 10 s '
                  f'({attempt}/{max_attempts}).', flush=True)
            time.sleep(10.)
            continue

        progress.update(iteration=int(metrics['iteration']), checkpoint=str(checkpoint),
            completed_stage_dir=str(stage_dir.resolve()))
        state['latest_checkpoint'] = str(checkpoint)
        state['completed'].append(dict(job_number=job_number, attempt=attempt,
            stage_dir=str(stage_dir.resolve()), checkpoint=str(checkpoint),
            num_envs=job_num_envs, metrics=metrics, **job))
        state['next_job'] += 1
        state['in_progress'] = None
        state['status'] = 'complete' if state['next_job'] == len(state['jobs']) else 'ready'
        update_latest_link(run_dir, checkpoint)
        atomic_json(run_dir/'state.json', state)

    print(f"\nFull RGB-D training completed: {run_dir/'latest.pt'}", flush=True)


def cmd_train_reactive():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config', default=str(ROOT/'eval/config/eval_pointgoal/humanoid_internscene_home.yaml'))
    parser.add_argument('--config', default=str(ROOT/'../../rl/config/reactive_rgbd_g1.yaml'))
    parser.add_argument('--checkpoint', default=str(REPO_ROOT/'checkpoints/x-navdp_posttrain.ckpt'))
    parser.add_argument('--scene-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=20002)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--rollout-steps', type=int, default=64)
    parser.add_argument('--num-envs', type=int, default=1,
        help='Parallel G1 environments (vectorized reactive training).')
    parser.add_argument('--episode-seconds', type=float)
    parser.add_argument('--output', default=str(ROOT/'outputs/reactive_train'),
        help='Run prefix; a YYYYMMDD_HHMMSS timestamp is appended to make each run unique.')
    parser.add_argument('--resume')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--legacy-blanket-scale', action='store_true',
        help='Use legacy additional .01 blanket scaling (can produce invalid PhysX shapes).')
    args = parser.parse_args()
    if args.iterations < 1 or args.rollout_steps < 2 or args.num_envs < 1 or (args.episode_seconds is not None and args.episode_seconds <= 0):
        parser.error('Iterations, rollout steps, num-envs and episode duration must be positive')
    # Headless + GPU-1 defaults (shell overrides win); drop DISPLAY to avoid the GLX hang.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    os.environ.setdefault('ISAAC_ACTIVE_GPU', '1')
    os.environ.setdefault('ISAAC_PHYSICS_GPU', '0')
    os.environ.pop('DISPLAY', None)
    # acados (MPC) solver path. NOTE: libhpipm.so/libblasfeo.so are resolved by the
    # dynamic linker at process startup, so LD_LIBRARY_PATH must be set in the SHELL
    # (see ../../rl/scripts/launch_reactive_train.sh); os.environ cannot affect dlopen.
    os.environ.setdefault('ACADOS_SOURCE_DIR', os.path.expanduser("~/acados"))
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    from eval.config_utils import load_default_config
    from eval.environment import namespace_to_dict
    cfg = load_default_config(args.scene_config)
    if cfg.environment.embodiment != 'unitree_g1':
        raise ValueError('Initial training adapter is G1 only')
    scene = reactive_training_scene(cfg, args.scene_index)
    checkpoint = str(Path(args.checkpoint).resolve(strict=True))
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    if args.episode_seconds is not None:
        config['navigation_training']['episode_seconds'] = args.episode_seconds
    config['training'].update(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    config['ppo']['num_mini_batches'] = 1
    config['x_navdp_checkpoint'] = checkpoint
    directory = Path(args.output + time.strftime('_%Y%m%d_%H%M%S', time.localtime()))
    directory.mkdir(parents=True, exist_ok=False)
    with open(directory/'run.json', 'w') as handle:
        json.dump(dict(args=vars(args), config=config, scene=scene, scene_split='train'), handle, indent=2)
    from src.training.worker import configure_mdl_system_path
    configure_mdl_system_path(cfg.environment.scene_dir)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    # Own this server explicitly. Never reuse or stop the user's existing port 20001 server.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    server_log = open(directory/'policy_server.log', 'w')
    process = subprocess.Popen([sys.executable, '-u', '-m', 'eval.src.policy_server',
        '--port', str(args.port), '--embodiment', 'humanoid', '--checkpoint', checkpoint,
        '--device', args.device, '--no-visualization', '--recovery-mode', 'baseline', '--seed', str(args.seed)],
        cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    app = env = adapter = None
    try:
        deadline = time.monotonic()+90.
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Planner server exited; see {directory}/policy_server.log')
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=1.):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Planner server startup timed out')
                time.sleep(.5)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app
        from src.environment import create_dingoeval_environment
        from src.utils import BatchMPCController
        from rl.src.encoder import build_rgbd_encoder, resolve_policy_visual_config
        from rl.src.policy import PolicyConfig, ReactiveActorCritic
        from rl.src.runner import ReactiveRunner
        from rl.src.train_env import ReactiveTrainEnv, AsyncPlanner
        from rl.src.isaac_backend import IsaacReactiveBackend
        from eval.src.client_utils import navigator_reset, pointgoal_step
        env, controller = create_dingoeval_environment(cfg.environment.scene_dir, 0, args.num_envs,
            scene_scale=getattr(cfg.environment, 'scene_scale', None), device=args.device,
            embodiment='unitree_g1', scene_data=scene, controller_config=namespace_to_dict(cfg.controller), seed=args.seed,
            preserve_blanket_scale=not args.legacy_blanket_scale)
        backend = IsaacReactiveBackend(env, controller, config['navigation_training'])
        def reset_planner():
            intrinsic = env.unwrapped.scene['camera_sensor'].data.intrinsic_matrices[0].cpu().numpy()
            navigator_reset(intrinsic, batch_size=args.num_envs, port=args.port, scene_name=scene['scene_name'],
                sample_indices=[int(env.unwrapped._sample_idx[i]) for i in range(args.num_envs)])
        def predict(state):
            return pointgoal_step(state['goal_xy'], state['rgb'], state['depth'], port=args.port,
                robot_pos=state['position'], robot_quat=state['quaternion'])[0]
        planner = AsyncPlanner(predict, reset_planner)
        encoder = build_rgbd_encoder(config.get("visual_encoder"), checkpoint, args.device, training=True)
        config["policy"] = resolve_policy_visual_config(config["policy"], encoder)
        with open(directory/'run.json') as handle:
            manifest = json.load(handle)
        manifest['encoder_fingerprint'] = encoder.fingerprint
        manifest['visual_encoder_metadata'] = encoder.metadata
        manifest['config']['policy'] = config['policy']
        manifest['control_dt'] = backend.dt
        manifest['preserve_blanket_scale'] = not args.legacy_blanket_scale
        with open(directory/'run.json', 'w') as handle:
            json.dump(manifest, handle, indent=2)
        mpc = BatchMPCController(batch=1, **namespace_to_dict(cfg.mpc))
        adapter = ReactiveTrainEnv(backend, encoder, mpc, planner, config, args.device, directory/"transitions.csv")
        policy = ReactiveActorCritic(PolicyConfig(**config['policy'])).to(args.device)
        runner = ReactiveRunner(policy, args.num_envs, steps=args.rollout_steps,
            ppo_config=config['ppo'], encoder_fingerprint=encoder.fingerprint, encoder_metadata=encoder.metadata)
        if args.resume:
            runner.load(args.resume)
        with open(directory/'metrics.jsonl', 'w') as log:
            for _ in range(args.iterations):
                started = time.perf_counter()
                before = policy.actor[-1].weight.detach().clone()
                metrics = runner.collect_and_update(adapter)
                metrics.update(adapter.metrics(), wall_s=time.perf_counter()-started,
                    actor_update_norm=float(torch.linalg.vector_norm(policy.actor[-1].weight-before)))
                if any(p.grad is not None for p in encoder.parameters()):
                    raise RuntimeError('Frozen encoder unexpectedly acquired gradients')
                line = json.dumps(metrics)
                print(line, flush=True); log.write(line+'\n'); log.flush()
                runner.save(directory/'latest.pt')
        print(f'Reactive training completed: {directory}/latest.pt', flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()  # Isaac shutdown can terminate the interpreter before an exception is displayed.
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
            elif env is not None:
                env.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            server_log.close()
            if app is not None:
                app.close()


def cmd_bench():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("dav2", "yolo26_depth"), required=True)
    parser.add_argument("--checkpoint", help="X-NavDP checkpoint (DA-V2 backend)")
    parser.add_argument("--weights", help="yolo26n-depth.pt path (YOLO backend)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--size", type=int, default=224, help="YOLO/DA-V2 feature input size")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--path-points", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--deadline-ms", type=float, default=40.0, help="25 Hz control deadline")
    parser.add_argument("--compile-mode", default=None,
                        help="torch.compile mode for the YOLO backend (e.g. reduce-overhead)")
    parser.add_argument("--breakdown", action="store_true",
                        help="Also time per-branch preprocessing/extraction stages")
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, policy = make_encoder_and_policy(args, device)
    policy.initialize_hidden(args.batch)
    print(json.dumps({"backend": args.backend,
                      "visual_encoder": encoder.metadata,
                      "policy_token_dim": policy.config.token_dim,
                      "policy_grid_size": policy.config.grid_size}, indent=2), flush=True)

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (args.batch, args.height, args.width, 3), dtype=np.uint8)
    depth = rng.random((args.batch, args.height, args.width, 1), dtype=np.float32) * 5.0

    rows = []
    with torch.inference_mode():
        def visual_total():
            return encoder(rgb, depth)

        features = visual_total()  # seed a real obs for the fixed policy timing
        obs = make_obs(features, policy.config, device, args.batch)

        def policy_step():
            return policy.act_inference(obs)

        def end_to_end():
            return policy.act_inference(make_obs(encoder(rgb, depth), policy.config, device, args.batch))

        rows.append(summarize("visual_total", run_timed(visual_total, args.runs, args.warmup, device),
                              args.deadline_ms))
        rows.append(summarize("policy", run_timed(policy_step, args.runs, args.warmup, device),
                              args.deadline_ms))
        rows.append(summarize("end_to_end", run_timed(end_to_end, args.runs, args.warmup, device),
                              args.deadline_ms))

        if args.breakdown:
            if args.backend == "yolo26_depth":
                from rl.src.encoder import MetricDepthPreprocessor
                image = preprocess_yolo_rgb(rgb, device, args.size)
                distance = MetricDepthPreprocessor()(depth, device, args.size)[0]
                rows.append(summarize("rgb_preprocess",
                                      run_timed(lambda: preprocess_yolo_rgb(rgb, device, args.size),
                                                args.runs, args.warmup, device), args.deadline_ms))
                rows.append(summarize("rgb_extract",
                                      run_timed(lambda: encoder.rgb_model(image), args.runs, args.warmup, device),
                                      args.deadline_ms))
                rows.append(summarize("depth_preprocess",
                                      run_timed(lambda: MetricDepthPreprocessor()(depth, device, args.size),
                                                args.runs, args.warmup, device), args.deadline_ms))
                rows.append(summarize("depth_extract",
                                      run_timed(lambda: encoder.depth_model(distance), args.runs, args.warmup, device),
                                      args.deadline_ms))
            else:
                from rl.src.encoder import preprocess_rgbd
                image, distance, _ = preprocess_rgbd(rgb, depth)
                rows.append(summarize("preprocess",
                                      run_timed(lambda: preprocess_rgbd(rgb, depth), args.runs, args.warmup, device),
                                      args.deadline_ms))
                rows.append(summarize("extract",
                                      run_timed(lambda: encoder.encode_preprocessed(image, distance),
                                                args.runs, args.warmup, device), args.deadline_ms))

    print(f"\n{'stage':<20}{'p50_ms':>10}{'p95_ms':>10}{'p99_ms':>10}{'miss@40ms%':>12}")
    for row in rows:
        print(f"{row['stage']:<20}{row['p50_ms']:>10.3f}{row['p95_ms']:>10.3f}"
              f"{row['p99_ms']:>10.3f}{row['deadline_miss_pct']:>11.2f}%")
    print(f"\nDeadline: {args.deadline_ms} ms (25 Hz control); feature period 100 ms for reference.")


STAGES = {
    "collect": cmd_collect,
    "collect-all": cmd_collect_all,
    "train-bc": cmd_train_bc,
    "train-tracker": cmd_train_tracker,
    "train-full": cmd_train_full,
    "train-reactive": cmd_train_reactive,
    "bench": cmd_bench,
}

_HELP = """\
usage: python -m rl.cli <stage> [args]

  collect         MPC-teacher BC collection (single scene)
  collect-all     BC collection across all home_train scenes
  train-bc        behavior-cloning warm-start
  train-tracker   MPC-free direct tracker PPO training
  train-full      sequential multi-scene direct-tracker training
  train-reactive  single-G1 residual PPO training
  bench           visual-encoder latency benchmark
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
