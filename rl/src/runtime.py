"""Nonblocking latest-frame RGB-D encoding and local residual execution.

The high-level HTTP planner is intentionally not called from this module.
Call submit_rgbd on new sensor frames, publish_plan on planner completion,
and step at the control rate with a freshly aligned nominal controller command.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from dataclasses import dataclass
import threading
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .actions import ActionComposer, DirectActionMapper
from .observation import make_observation, make_direct_observation
from .reference import ReferenceTracker


@dataclass(frozen=True)
class FeaturePacket:
    features: dict
    frame_id: int
    capture_time: float
    positions: np.ndarray
    quaternions: np.ndarray
    epoch: int
    encode_wall_s: float = 0.
    queue_wall_s: float = 0.
    published_wall_time: float = 0.


class LatestRGBDWorker:
    """One encoder consumer + one replaceable input slot; no unbounded backlog."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.condition = threading.Condition()
        self.pending = self.latest = None
        self.epoch = 0
        self.last_id = -1
        self.last_time = -float('inf')
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='reactive-rgbd')
        self.thread.start()

    def reset(self):
        with self.condition:
            self.epoch += 1
            self.pending = self.latest = None
            self.last_id = -1
            self.last_time = -float('inf')
            self.error = None

    def submit(self, rgb, depth, frame_id, capture_time, positions, quaternions):
        with self.condition:
            if self.closed:
                raise RuntimeError('RGB-D worker is closed')
            if frame_id <= self.last_id:
                return False
            if not np.isfinite(capture_time) or capture_time < self.last_time:
                raise ValueError('Capture time must be finite and monotonic within epoch')
            self.last_id, self.last_time = frame_id, capture_time
            self.pending = (np.array(rgb, copy=True), np.array(depth, copy=True), int(frame_id),
                float(capture_time), np.array(positions, copy=True), np.array(quaternions, copy=True), self.epoch, time.perf_counter())
            self.condition.notify()
            return True

    def snapshot(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('RGB-D encoding failed') from self.error
            return self.latest

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                rgb, depth, frame, timestamp, pos, quat, epoch, submitted = self.pending
                self.pending = None
            try:
                started = time.perf_counter()
                features = self.encoder(rgb, depth)
                # Publish only after this producer's GPU writes have completed.
                if features['rgb_tokens'].is_cuda:
                    torch.cuda.current_stream(features['rgb_tokens'].device).synchronize()
                completed = time.perf_counter()
                packet = FeaturePacket(features, frame, timestamp, pos, quat, epoch,
                    completed-started, started-submitted, completed)
                with self.condition:
                    if epoch == self.epoch and not self.closed:
                        self.latest, self.error = packet, None
            except Exception as exc:
                with self.condition:
                    if epoch == self.epoch:
                        self.error, self.latest = exc, None

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=30.)
        if self.thread.is_alive():
            raise RuntimeError('RGB-D worker did not exit; encoder may be stalled')


class ReactiveExecutor:
    def __init__(self, policy, num_envs, limits=None, max_plan_age=3.,
                 max_rgbd_age=.5, max_nominal_age=.3):
        if num_envs < 1 or min(max_plan_age, max_rgbd_age, max_nominal_age) <= 0:
            raise ValueError('Environment count and age limits must be positive')
        self.policy, self.num_envs = policy.eval(), num_envs
        self.device = next(policy.parameters()).device
        self.composer = ActionComposer(limits)
        self.max_rgbd_age, self.max_nominal_age = max_rgbd_age, max_nominal_age
        self.trackers = [ReferenceTracker(policy.config.path_points, max_plan_age) for _ in range(num_envs)]
        self.previous = torch.zeros(num_envs, 2, device=self.device)
        self.previous_residual = torch.zeros_like(self.previous)
        self.last_frame = [-1]*num_envs
        self.reset_time = np.full(num_envs, -float('inf'))
        self.last_step_time = None
        policy.initialize_hidden(num_envs)

    def reset_env(self, i, episode_id, now):
        self.trackers[i].reset(episode_id)
        self.previous[i] = self.previous_residual[i] = 0
        self.last_frame[i] = -1
        self.reset_time[i] = now
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[i] = True
        self.policy.reset(dones)

    def publish_plan(self, i, trajectory, position, quaternion, capture_time, plan_id, episode_id):
        return self.trackers[i].publish(trajectory, position, quaternion, capture_time, plan_id, episode_id)

    def reference(self, positions, quaternions, now):
        results = [tracker.sample(positions[i], quaternions[i], now) for i, tracker in enumerate(self.trackers)]
        return tuple(np.asarray([row[j] for row in results]) for j in range(4))

    @torch.no_grad()
    def step(self, packet, positions, quaternions, body_velocity, projected_gravity,
             goal_xy, nominal, nominal_times, nominal_plan_ids, now, dt,
             danger=None, emergency=None, observe_only=False):
        if not np.isfinite(now) or (self.last_step_time is not None and now <= self.last_step_time):
            raise ValueError('Control time must increase; create a new executor when restarting simulation')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Control dt must be positive')
        self.last_step_time = now
        batch = self.num_envs
        nominal = torch.as_tensor(nominal, device=self.device, dtype=torch.float32)
        danger = torch.zeros(batch, device=self.device, dtype=torch.bool) if danger is None else torch.as_tensor(danger, device=self.device).bool()
        emergency = torch.zeros(batch, device=self.device, dtype=torch.bool) if emergency is None else torch.as_tensor(emergency, device=self.device).bool()
        path, mask, plan_age, valid_path = self.reference(positions, quaternions, now)
        nominal_age = now-np.asarray(nominal_times)
        matching = np.asarray(nominal_plan_ids) == np.asarray([t.plan_id for t in self.trackers])
        valid_nominal = matching & (nominal_age >= 0) & (nominal_age <= self.max_nominal_age)
        if packet is None:
            valid_visual = np.zeros(batch, dtype=bool)
        else:
            valid_visual = ((now-packet.capture_time >= 0) & (now-packet.capture_time <= self.max_rgbd_age)
                            & (packet.capture_time > self.reset_time))
        invalid = ~(valid_path & valid_nominal & valid_visual)
        stop = emergency | torch.as_tensor(invalid, device=self.device)
        if packet is None:
            latent = torch.zeros_like(nominal)
            obs = None
        else:
            if packet.positions.shape != (batch, 3) or packet.quaternions.shape != (batch, 4):
                raise ValueError('Feature packet capture poses do not match environment batch')
            pose_delta = []
            for i in range(batch):
                old, current = Rotation.from_quat(packet.quaternions[i]), Rotation.from_quat(quaternions[i])
                shift = old.inv().apply(np.asarray(positions[i])-packet.positions[i])
                angle = (old.inv()*current).as_euler('xyz')[2]
                pose_delta.append([shift[0], shift[1], angle])
            fields = {'body_velocity': body_velocity, 'projected_gravity': projected_gravity,
                'pointgoal_xy': goal_xy, 'nominal': nominal, 'previous_command': self.previous,
                'previous_residual': self.previous_residual, 'plan_age': plan_age,
                'rgbd_age': np.full(batch, now-packet.capture_time),
                'depth_valid_fraction': packet.features['depth_valid_fraction'],
                'path_valid': valid_path, 'rgbd_valid': valid_visual, 'rgbd_pose_delta': pose_delta,
                'new_rgbd': [packet.frame_id != f for f in self.last_frame],
                'dt': np.full(batch, dt), 'danger': danger}
            features = {k: v.to(self.device) for k, v in packet.features.items()}
            obs = make_observation(features, path, mask, fields)
            if observe_only:
                latent = torch.zeros_like(nominal)
            else:
                self.policy.reset(torch.as_tensor(invalid, device=self.device))
                latent = self.policy.act_inference(obs)
                self.policy.reset(torch.as_tensor(invalid, device=self.device))
            for i in range(batch):
                if valid_visual[i]:
                    self.last_frame[i] = packet.frame_id
        if observe_only:
            return dict(observation=obs, nominal=nominal, stop=stop, danger=danger,
                        invalid_input=torch.as_tensor(invalid, device=self.device), plan_age=plan_age)
        result = self.composer(latent, nominal, self.previous, dt, danger, stop)
        self.previous.copy_(result['command'])
        self.previous_residual.copy_(result['applied_residual'])
        result.update(observation=obs, invalid_input=torch.as_tensor(invalid, device=self.device),
                      latent=latent, plan_age=plan_age)
        return result


def _pose_delta(old_position, old_quaternion, position, quaternion):
    """Planar capture->current pose delta [dx, dy, dyaw] in the capture frame."""
    old = Rotation.from_quat(old_quaternion)
    current = Rotation.from_quat(quaternion)
    shift = old.inv().apply(np.asarray(position, dtype=float) - np.asarray(old_position, dtype=float))
    angle = (old.inv()*current).as_euler('xyz')[2]
    return np.array([shift[0], shift[1], angle], dtype=np.float32)


class DirectReactiveExecutor:
    """MPC-free tracker: latent z -> bounded velocity, with latency-aware state.

    Same async contract as ReactiveExecutor (latest-frame RGB-D worker, timestamped
    reference path), but the policy directly commands [v, w]; there is no nominal
    controller, no MPC status, and no Acados dependency.
    """
    def __init__(self, policy, num_envs, limits=None, max_plan_age=3., max_rgbd_age=.5):
        if num_envs < 1 or min(max_plan_age, max_rgbd_age) <= 0:
            raise ValueError('Environment count and age limits must be positive')
        self.policy, self.num_envs = policy.eval(), num_envs
        self.device = next(policy.parameters()).device
        self.mapper = DirectActionMapper(limits)
        self.max_rgbd_age = max_rgbd_age
        self.trackers = [ReferenceTracker(policy.config.path_points, max_plan_age) for _ in range(num_envs)]
        self.previous = torch.zeros(num_envs, 2, device=self.device)
        self.previous_delta = torch.zeros_like(self.previous)
        self.last_frame = [-1]*num_envs
        self.last_plan_id = [-1]*num_envs
        self.reset_time = np.full(num_envs, -float('inf'))
        self.last_step_time = None
        policy.initialize_hidden(num_envs)

    def reset_env(self, i, episode_id, now):
        self.trackers[i].reset(episode_id)
        self.previous[i] = self.previous_delta[i] = 0
        self.last_frame[i] = -1
        self.last_plan_id[i] = -1
        self.reset_time[i] = now
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[i] = True
        self.policy.reset(dones)

    def publish_plan(self, i, trajectory, position, quaternion, capture_time, plan_id, episode_id):
        return self.trackers[i].publish(trajectory, position, quaternion, capture_time, plan_id, episode_id)

    def reference(self, positions, quaternions, now):
        results = [tracker.sample(positions[i], quaternions[i], now) for i, tracker in enumerate(self.trackers)]
        return tuple(np.asarray([row[j] for row in results]) for j in range(4))

    def apply_command(self, command):
        """Apply an externally-provided command (e.g., a BC teacher) and update previous."""
        command = torch.as_tensor(command, device=self.device, dtype=torch.float32)
        if command.shape != self.previous.shape:
            raise ValueError('Command must match the executor batch shape')
        old = self.previous.clone()
        self.previous_delta.copy_(command-old)
        self.previous.copy_(command)

    @torch.no_grad()
    def step(self, packet, positions, quaternions, body_velocity, projected_gravity,
             goal_xy, now, dt, emergency=None, observe_only=False):
        if not np.isfinite(now) or (self.last_step_time is not None and now <= self.last_step_time):
            raise ValueError('Control time must increase; create a new executor when restarting simulation')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Control dt must be positive')
        self.last_step_time = now
        batch = self.num_envs
        emergency = torch.zeros(batch, device=self.device, dtype=torch.bool) if emergency is None else torch.as_tensor(emergency, device=self.device).bool()
        if emergency.shape != (batch,):
            raise ValueError('Emergency mask must have shape B')
        path, mask, plan_age, valid_path = self.reference(positions, quaternions, now)
        if packet is None:
            valid_visual = np.zeros(batch, dtype=bool)
        else:
            valid_visual = ((now-packet.capture_time >= 0) & (now-packet.capture_time <= self.max_rgbd_age)
                            & (packet.capture_time > self.reset_time))
        invalid = ~(valid_path & valid_visual)
        stop = emergency | torch.as_tensor(invalid, device=self.device)
        if packet is None:
            latent = torch.zeros(batch, 2, device=self.device)
            obs = None
        else:
            if packet.positions.shape != (batch, 3) or packet.quaternions.shape != (batch, 4):
                raise ValueError('Feature packet capture poses do not match environment batch')
            rgbd_pose_delta = np.stack([_pose_delta(packet.positions[i], packet.quaternions[i],
                positions[i], quaternions[i]) for i in range(batch)])
            plan_pose_delta = np.zeros((batch, 3), dtype=np.float32)
            for i, tracker in enumerate(self.trackers):
                if tracker.world is not None:
                    plan_pose_delta[i] = _pose_delta(tracker.capture_position, tracker.capture_quaternion,
                        positions[i], quaternions[i])
            measured = np.asarray(body_velocity, dtype=np.float32)[:, [0, 5]]
            command_error = self.previous.cpu().numpy() - measured
            remaining_arc = np.array([t.arc[-1]-t.progress if t.world is not None else 0. for t in self.trackers], dtype=np.float32)
            new_plan = np.array([float(t.plan_id != self.last_plan_id[i]) for i, t in enumerate(self.trackers)], dtype=np.float32)
            new_rgbd = np.array([float(packet.frame_id != f) for f in self.last_frame], dtype=np.float32)
            fields = {'body_velocity': body_velocity, 'projected_gravity': projected_gravity,
                'pointgoal_xy': goal_xy, 'previous_command': self.previous,
                'previous_command_delta': self.previous_delta, 'command_velocity_error': command_error,
                'plan_age': plan_age, 'rgbd_age': np.full(batch, now-packet.capture_time),
                'plan_pose_delta': plan_pose_delta, 'rgbd_pose_delta': rgbd_pose_delta,
                'depth_valid_fraction': packet.features['depth_valid_fraction'],
                'path_valid': valid_path, 'rgbd_valid': valid_visual,
                'new_plan': new_plan, 'new_rgbd': new_rgbd,
                'dt': np.full(batch, dt), 'remaining_path_arc': remaining_arc}
            features = {k: v.to(self.device) for k, v in packet.features.items()}
            obs = make_direct_observation(features, path, mask, fields)
            if observe_only:
                latent = torch.zeros(batch, 2, device=self.device)
            else:
                self.policy.reset(torch.as_tensor(invalid, device=self.device))
                latent = self.policy.act_inference(obs)
                self.policy.reset(torch.as_tensor(invalid, device=self.device))
            for i in range(batch):
                if valid_visual[i]:
                    self.last_frame[i] = packet.frame_id
                self.last_plan_id[i] = self.trackers[i].plan_id
        if observe_only:
            return dict(observation=obs, stop=stop, emergency=emergency,
                        invalid_input=torch.as_tensor(invalid, device=self.device),
                        plan_age=plan_age, valid_path=valid_path, valid_visual=valid_visual)
        old_previous = self.previous.clone()
        result = self.mapper(latent, self.previous, dt, stop)
        self.previous_delta.copy_(result['command']-old_previous)
        self.previous.copy_(result['command'])
        result.update(observation=obs, invalid_input=torch.as_tensor(invalid, device=self.device),
                      latent=latent, plan_age=plan_age, valid_path=valid_path, valid_visual=valid_visual)
        return result
