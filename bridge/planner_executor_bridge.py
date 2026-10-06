"""Optional evaluation adapter; no Isaac imports, so contracts can be tested on CPU."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import csv
import time
import numpy as np
import torch
import yaml
from dataclasses import asdict
from rl.src.actions import ActionLimits, DirectLimits
from rl.src.encoder import build_rgbd_encoder, resolve_policy_visual_config
from rl.src.policy import PolicyConfig, ReactiveActorCritic
from rl.src.runner import load_policy, CONTROL_MODE_DIRECT
from rl.src.runtime import LatestRGBDWorker, ReactiveExecutor, DirectReactiveExecutor


def _inference_policy_config(config):
    """Policy fields that affect deterministic inference outputs/shapes."""
    ignored = {'init_std', 'init_std_v', 'init_std_w', 'std_min', 'std_max'}
    return {key: value for key, value in asdict(config).items() if key not in ignored}


class ReactiveEvalBridge:
    def __init__(self, encoder, policy, mpc, num_envs, config, log_path=None):
        self.worker = LatestRGBDWorker(encoder)
        runtime = config['runtime']
        self.executor = ReactiveExecutor(policy, num_envs, ActionLimits(**config['actions']),
            runtime['max_plan_age_s'], runtime['max_rgbd_age_s'], runtime['max_nominal_age_s'])
        self.mpc, self.batch = mpc, num_envs
        self.epoch = 0
        self.frame_id = 0
        self.camera_frames = None
        self.feature_period = runtime['feature_period_s']
        self.last_capture = -float('inf')
        self.accepted_plan = -1
        self.log = open(log_path, 'w', newline='') if log_path else None
        self.writer = None
        self.last_result = None

    @classmethod
    def from_args(cls, args, mpc, num_envs, log_path):
        with open(args.reactive_config) as handle:
            config = yaml.safe_load(handle)
        encoder = build_rgbd_encoder(config.get("visual_encoder"), args.reactive_encoder_checkpoint, args.reactive_device)
        config["policy"] = resolve_policy_visual_config(config["policy"], encoder)
        if args.reactive_zero:
            policy = ReactiveActorCritic(PolicyConfig(**config['policy'])).to(args.reactive_device)
        else:
            policy = load_policy(args.reactive_checkpoint, encoder.metadata, args.reactive_device)
            if _inference_policy_config(policy.config) != _inference_policy_config(
                    PolicyConfig(**config['policy'])):
                raise ValueError('Reactive YAML policy differs from checkpoint')
        return cls(encoder, policy, mpc, num_envs, config, log_path)

    def reset(self, epoch, now):
        # The planner invalidates its whole batch on a partial environment reset.
        # Invalidate our batch too, to avoid mixing plans from different RPC epochs.
        self.epoch = epoch
        self.worker.reset()
        self.camera_frames = None
        self.last_capture = -float('inf')
        self.accepted_plan = -1
        for i in range(self.batch):
            self.executor.reset_env(i, epoch, now)

    def submit_frame(self, rgb, depth, camera_frames, now, positions, quaternions):
        frames = np.asarray(camera_frames)
        changed = self.camera_frames is None or np.all(frames != self.camera_frames)
        if not changed or now-self.last_capture < self.feature_period-1e-6:
            return False
        self.frame_id += 1
        self.worker.submit(rgb, depth, self.frame_id, now, positions, quaternions)
        self.camera_frames = frames.copy()
        self.last_capture = now
        return True

    def publish(self, plan):
        if plan is None or plan['epoch'] != self.epoch or plan['plan_id'] <= self.accepted_plan:
            return False
        for i in range(self.batch):
            self.executor.publish_plan(i, plan['trajectory'][i], plan['positions'][i],
                plan['quaternions'][i], plan['capture_time'], plan['plan_id'], self.epoch)
        self.accepted_plan = plan['plan_id']
        return True

    @torch.no_grad()
    def step(self, positions, quaternions, velocity, gravity, goal_xy, now, dt):
        started = time.perf_counter()
        path, mask, _, valid = self.executor.reference(positions, quaternions, now)
        nominal = np.zeros((self.batch, 2), dtype=np.float32)
        mpc_status = np.zeros(self.batch, dtype=np.int32)
        mpc_start = time.perf_counter()
        for i in range(self.batch):
            if valid[i]:
                points = path[i, mask[i].astype(bool), :2]
                # Never include masked zero points: they would send MPC back to origin.
                reference = np.zeros((1, len(points)+1, 3), dtype=np.float32)
                reference[0, 1:, :2] = points
                nominal[i] = self.mpc.solve(reference)[0][0, 0]
                mpc_status[i] = getattr(self.mpc, 'last_statuses', [0])[0]
                if mpc_status[i] != 0 or not np.isfinite(nominal[i]).all():
                    mpc_status[i] = mpc_status[i] or -1
                    nominal[i] = 0.
        mpc_s = time.perf_counter()-mpc_start
        packet = self.worker.snapshot()
        result = self.executor.step(packet, positions, quaternions, velocity, gravity,
            goal_xy, nominal, np.full(self.batch, now),
            [t.plan_id for t in self.executor.trackers], now, dt, emergency=mpc_status != 0)
        command = result['command'].cpu().numpy()
        # .cpu() above synchronizes the policy result before wall timing ends.
        total_s = time.perf_counter()-started
        self.last_result = result
        if self.log:
            for i in range(self.batch):
                row = dict(mpc_status=int(mpc_status[i]), sim_time=now, env_id=i, epoch=self.epoch, plan_id=self.accepted_plan,
                    frame_id=-1 if packet is None else packet.frame_id,
                    rgbd_age_s=-1 if packet is None else now-packet.capture_time,
                    plan_age_s=float(result['plan_age'][i]), mpc_wall_s=mpc_s,
                    control_wall_s=total_s, control_deadline_missed=total_s > dt,
                    feature_encode_wall_s=-1 if packet is None else packet.encode_wall_s,
                    feature_queue_wall_s=-1 if packet is None else packet.queue_wall_s,
                    feature_publish_age_wall_s=-1 if packet is None else time.perf_counter()-packet.published_wall_time,
                    nominal_v=float(nominal[i, 0]), nominal_w=float(nominal[i, 1]),
                    command_v=float(command[i, 0]), command_w=float(command[i, 1]),
                    invalid_input=bool(result['invalid_input'][i]),
                    limited=bool(result['limited'][i]))
                if self.writer is None:
                    self.writer = csv.DictWriter(self.log, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
            self.log.flush()
        return command

    def close(self):
        try:
            self.worker.close()
        finally:
            if self.log:
                self.log.close()


class DirectReactiveEvalBridge:
    """MPC-free evaluation adapter; no Acados, no nominal controller.

    Same async contract as ReactiveEvalBridge but the policy directly commands
    [v, w] and this bridge never instantiates an MPC solver.
    """
    def __init__(self, encoder, policy, num_envs, config, log_path=None):
        self.worker = LatestRGBDWorker(encoder)
        runtime = config['runtime']
        self.executor = DirectReactiveExecutor(policy, num_envs, DirectLimits(**config['actions']),
            runtime['max_plan_age_s'], runtime['max_rgbd_age_s'])
        self.batch = num_envs
        self.epoch = 0
        self.frame_id = 0
        self.camera_frames = None
        self.feature_period = runtime['feature_period_s']
        self.last_capture = -float('inf')
        self.accepted_plan = -1
        self.log = open(log_path, 'w', newline='') if log_path else None
        self.writer = None
        self.last_result = None

    @classmethod
    def from_args(cls, args, num_envs, log_path):
        with open(args.direct_config) as handle:
            config = yaml.safe_load(handle)
        encoder = build_rgbd_encoder(config.get("visual_encoder"), args.reactive_encoder_checkpoint, args.reactive_device)
        config["policy"] = resolve_policy_visual_config(config["policy"], encoder)
        if args.direct_zero:
            policy = ReactiveActorCritic(PolicyConfig(**config['policy'])).to(args.reactive_device)
        else:
            policy = load_policy(args.direct_checkpoint, encoder.metadata, args.reactive_device,
                control_mode=CONTROL_MODE_DIRECT)
            if _inference_policy_config(policy.config) != _inference_policy_config(
                    PolicyConfig(**config['policy'])):
                raise ValueError('Direct YAML policy differs from checkpoint')
        return cls(encoder, policy, num_envs, config, log_path)

    def reset(self, epoch, now):
        self.epoch = epoch
        self.worker.reset()
        self.camera_frames = None
        self.last_capture = -float('inf')
        self.accepted_plan = -1
        for i in range(self.batch):
            self.executor.reset_env(i, epoch, now)

    def submit_frame(self, rgb, depth, camera_frames, now, positions, quaternions):
        frames = np.asarray(camera_frames)
        changed = self.camera_frames is None or np.all(frames != self.camera_frames)
        if not changed or now-self.last_capture < self.feature_period-1e-6:
            return False
        self.frame_id += 1
        self.worker.submit(rgb, depth, self.frame_id, now, positions, quaternions)
        self.camera_frames = frames.copy()
        self.last_capture = now
        return True

    def publish(self, plan):
        if plan is None or plan['epoch'] != self.epoch or plan['plan_id'] <= self.accepted_plan:
            return False
        for i in range(self.batch):
            self.executor.publish_plan(i, plan['trajectory'][i], plan['positions'][i],
                plan['quaternions'][i], plan['capture_time'], plan['plan_id'], self.epoch)
        self.accepted_plan = plan['plan_id']
        return True

    @torch.no_grad()
    def step(self, positions, quaternions, velocity, gravity, goal_xy, now, dt):
        started = time.perf_counter()
        packet = self.worker.snapshot()
        result = self.executor.step(packet, positions, quaternions, velocity, gravity,
            goal_xy, now, dt)
        command = result['command'].cpu().numpy()
        # .cpu() above synchronizes the policy result before wall timing ends.
        total_s = time.perf_counter()-started
        self.last_result = result
        if self.log:
            for i in range(self.batch):
                row = dict(sim_time=now, env_id=i, epoch=self.epoch, plan_id=self.accepted_plan,
                    frame_id=-1 if packet is None else packet.frame_id,
                    rgbd_age_s=-1 if packet is None else now-packet.capture_time,
                    plan_age_s=float(result['plan_age'][i]),
                    control_wall_s=total_s, control_deadline_missed=total_s > dt,
                    feature_encode_wall_s=-1 if packet is None else packet.encode_wall_s,
                    feature_queue_wall_s=-1 if packet is None else packet.queue_wall_s,
                    command_v=float(command[i, 0]), command_w=float(command[i, 1]),
                    invalid_input=bool(result['invalid_input'][i]),
                    limited=bool(result['limited'][i]))
                if self.writer is None:
                    self.writer = csv.DictWriter(self.log, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
            self.log.flush()
        return command

    def close(self):
        try:
            self.worker.close()
        finally:
            if self.log:
                self.log.close()
