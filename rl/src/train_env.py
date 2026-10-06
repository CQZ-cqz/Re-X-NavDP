"""One-control-step residual training, with asynchronous frozen RGB-D and planner.

Backend contract: reset()->snapshot; step(command)->(next_snapshot, terminated,
truncated, terminal_snapshot). Snapshot arrays are copied before Isaac auto-reset.
This first adapter explicitly supports one G1, rather than silently mixing resets.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from concurrent.futures import ThreadPoolExecutor
import time
import csv
import numpy as np
import torch
from .actions import ActionLimits, DirectLimits
from .policy import PolicyConfig, ReactiveActorCritic
from .reference import project_to_path
from .runtime import LatestRGBDWorker, FeaturePacket, ReactiveExecutor, DirectReactiveExecutor


class AsyncPlanner:
    def __init__(self, predict, reset):
        self.predict, self.reset_rpc = predict, reset
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reactive-planner')
        self.future = None
        self.epoch = 0
        self.plan_id = 0

    def reset(self):
        if self.future is not None:
            self.future.result(timeout=190.)
            self.future = None
        self.epoch += 1
        self.reset_rpc()

    def submit(self, state):
        if self.future is not None:
            return False
        self.plan_id += 1
        plan_id, epoch = self.plan_id, self.epoch
        def run():
            trajectory = self.predict(state)
            return dict(trajectory=trajectory, capture_time=state['now'],
                positions=state['position'], quaternions=state['quaternion'],
                plan_id=plan_id, epoch=epoch)
        self.future = self.pool.submit(run)
        return True

    def poll(self, wait=False):
        if self.future is None or (not wait and not self.future.done()):
            return None
        plan = self.future.result(timeout=190.)
        self.future = None
        return plan if plan['epoch'] == self.epoch else None

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def navigation_reward(before, after, result, previous, dt, cfg):
    """Rewards use pre-reset end state, never the next episode's goal/contact."""
    distance_before = np.linalg.norm(before['goal_xy'], axis=-1)
    distance_after = np.linalg.norm(after['goal_xy'], axis=-1)
    command = result['command'].detach().cpu().numpy()
    delta = result['raw_residual'].detach().cpu().numpy()
    parts = dict(progress=cfg['progress']*(distance_before-distance_after),
        collision=-cfg['collision']*np.asarray(after['collision'], dtype=float),
        fall=-cfg['fall']*np.asarray(after['fallen'], dtype=float),
        success=cfg['success']*np.asarray(after['success'], dtype=float),
        smoothness=-cfg['smoothness']*np.square(command-previous).sum(-1),
        residual=-cfg['residual']*np.square(delta).sum(-1)*dt,
        time=-np.full(len(command), cfg['time']*dt))
    return parts


class ReactiveTrainEnv:
    def __init__(self, backend, encoder, mpc, planner, config, device='cpu', log_path=None):
        self.backend, self.encoder, self.mpc, self.planner = backend, encoder, mpc, planner
        self.num_envs = backend.num_envs
        self.cfg, self.device = config, torch.device(device)
        self.dt = backend.dt
        self.worker = LatestRGBDWorker(encoder)
        # Observer owns no PPO recurrent state; terminal-observation assembly cannot reset critic memory.
        observer = ReactiveActorCritic(PolicyConfig(**config['policy'])).to(device)
        rt = config['runtime']
        self.control = ReactiveExecutor(observer, self.num_envs, ActionLimits(**config['actions']),
            rt['max_plan_age_s'], rt['max_rgbd_age_s'], rt['max_nominal_age_s'])
        self.frame_id = 0
        self.frame = None
        self.last_feature_time = self.last_plan_time = -float('inf')
        self.state = self.prepared = None
        self.steps = self.episodes = self.invalid_steps = self.mpc_failures = 0
        self.timeout_bootstraps = self.successes = self.collisions = 0
        self.reward_totals = {}
        self.log = open(log_path, "w", newline="") if log_path else None
        self.writer = None

    def _clear(self, now):
        self.worker.reset()
        self.planner.reset()
        for i in range(self.num_envs):
            self.control.reset_env(i, self.planner.epoch, now)
        self.control.last_step_time = None
        self.frame = None
        self.last_feature_time = self.last_plan_time = -float('inf')

    def _publish(self, plan):
        if plan:
            for i in range(self.num_envs):
                self.control.publish_plan(i, plan['trajectory'][i], plan['positions'][i],
                    plan['quaternions'][i], plan['capture_time'], plan['plan_id'], plan['epoch'])

    def _prepare(self, state, warm=False, terminal=False):
        now = state['now']
        fresh = self.frame != state['frame']
        if fresh and now-self.last_feature_time >= self.cfg['runtime']['feature_period_s']-1e-6:
            self.frame_id += 1
            self.worker.submit(state['rgb'], state['depth'], self.frame_id, now,
                state['position'], state['quaternion'])
            self.frame, self.last_feature_time = state['frame'], now
        if not terminal:
            self._publish(self.planner.poll())
            if fresh and now-self.last_plan_time >= self.cfg['navigation_training']['planner_period_s']:
                if self.planner.submit(state):
                    self.last_plan_time = now
        if warm:
            self._publish(self.planner.poll(wait=True))
            deadline = time.monotonic()+60.
            while self.worker.snapshot() is None and time.monotonic() < deadline:
                time.sleep(.005)
            if self.worker.snapshot() is None:
                raise TimeoutError('Initial RGB-D encoder did not complete')
        packet = self.worker.snapshot()
        if packet is None:
            pc = self.control.policy.config
            tokens = torch.zeros(self.num_envs, pc.grid_size**2, pc.token_dim, device=self.device)
            packet = FeaturePacket(dict(rgb_tokens=tokens, depth_tokens=tokens.clone(),
                depth_valid_fraction=torch.zeros(self.num_envs, device=self.device)), -1,
                now-self.cfg['runtime']['max_rgbd_age_s']-1., state['position'], state['quaternion'], self.worker.epoch)
        path, mask, _, valid = self.control.reference(state['position'], state['quaternion'], now)
        nominal = np.zeros((self.num_envs, 2), dtype=np.float32)
        failed = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if valid[i]:
                points = path[i, mask[i].astype(bool), :2]
                reference = np.zeros((1, len(points)+1, 3), dtype=np.float32)
                reference[0, 1:, :2] = points
                nominal[i] = self.mpc.solve(reference)[0][0, 0]
                failed[i] = int(self.mpc.last_statuses[0]) != 0 or not np.isfinite(nominal[i]).all()
                if failed[i]:
                    nominal[i] = 0
        if failed.any() and not terminal:
            self.mpc_failures += int(failed.sum())
        prepared = self.control.step(packet, state['position'], state['quaternion'],
            state['velocity'], state['gravity'], state['goal_xy'], nominal, np.full(self.num_envs, now),
            [t.plan_id for t in self.control.trackers], now, self.dt,
            emergency=failed, observe_only=True)
        return prepared

    def reset(self):
        self.state = self.backend.reset()
        self._clear(self.state['now'])
        # First reset has no prior-episode image; allow its freshly captured initial frame.
        self.control.reset_time[:] = -float('inf')
        self.prepared = self._prepare(self.state, warm=True)
        return self.prepared['observation']

    def step(self, latent):
        prepared = self.prepared
        previous = self.control.previous.detach().cpu().numpy().copy()
        result = self.control.composer(latent, prepared['nominal'], self.control.previous,
            self.dt, prepared['danger'], prepared['stop'])
        self.control.previous.copy_(result['command'])
        self.control.previous_residual.copy_(result['applied_residual'])
        following, terminated, truncated, terminal_state = self.backend.step(result['command'].detach().cpu().numpy())
        done_mask = (terminated | truncated).astype(bool)
        # Per-env end state: pre-reset terminal state for done envs, post-step for the rest.
        end = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in following.items()}
        if terminal_state is not None and done_mask.any():
            for key in end:
                ev, tv = end[key], terminal_state[key]
                if isinstance(ev, np.ndarray) and ev.ndim >= 1 and ev.shape[0] == self.num_envs:
                    ev[done_mask] = tv[done_mask]
        parts = navigation_reward(self.state, end, result, previous, self.dt, self.cfg['reward'])
        # Lateral deviation from each env's original world path, at end-of-step position.
        deviation = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            tracker = self.control.trackers[i]
            if tracker.world is not None and len(tracker.world) > 1:
                starts, vectors = tracker.world[:-1, :2], np.diff(tracker.world[:, :2], axis=0)
                point = end['position'][i, :2]
                t = np.clip(((point-starts)*vectors).sum(-1)/np.maximum((vectors**2).sum(-1), 1e-9), 0, 1)
                deviation[i] = np.linalg.norm(starts+t[:, None]*vectors-point, axis=-1).min()
        parts['tracking'] = -self.cfg['reward']['tracking']*deviation*self.dt
        reward = sum(parts.values())
        if not np.isfinite(reward).all():
            raise FloatingPointError('Nonfinite navigation reward')
        self.steps += 1
        self.invalid_steps += int(prepared['stop'].sum())
        for key, value in parts.items():
            self.reward_totals[key] = self.reward_totals.get(key, 0.)+float(np.asarray(value).sum())
        if self.log:
            commands = result['command'].detach().cpu().numpy()
            for i in range(self.num_envs):
                row = dict(step=self.steps, env=i, sim_time=end['now'],
                    plan_id=self.control.trackers[i].plan_id,
                    plan_age_s=float(prepared['plan_age'][i]),
                    stopped=bool(prepared['stop'][i]),
                    goal_before=float(np.linalg.norm(self.state['goal_xy'][i])),
                    goal_after=float(np.linalg.norm(end['goal_xy'][i])),
                    command_v=float(commands[i, 0]), command_w=float(commands[i, 1]),
                    desired_v=float(result['desired'][i, 0]), desired_w=float(result['desired'][i, 1]),
                    raw_desired_v=float(result['raw_desired'][i, 0]),
                    raw_desired_w=float(result['raw_desired'][i, 1]),
                    joint_limited=bool(result['joint_limited'][i]),
                    slew_limited=bool(result['limited'][i]),
                    terminated=bool(terminated[i]), truncated=bool(truncated[i]),
                    collision=bool(end['collision'][i]), fallen=bool(end['fallen'][i]),
                    success=bool(end['success'][i]), reward=float(reward[i]),
                    **{f'reward_{key}': float(np.asarray(value)[i]) for key, value in parts.items()})
                if self.writer is None:
                    self.writer = csv.DictWriter(self.log, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
            self.log.flush()
        extras = {}
        if done_mask.any():
            # Assemble before clearing paths/features/commands for the next episode.
            extras['terminal_observation'] = self._prepare(end, terminal=True)['observation'].clone()
            self.timeout_bootstraps += int(((truncated & ~terminated).astype(bool)).sum())
            self.episodes += int(done_mask.sum())
            self.successes += int(np.asarray(end['success'])[done_mask].sum())
            self.collisions += int(np.asarray(end['collision'])[done_mask].sum())
            self._clear(following['now'])
        self.state = following
        self.prepared = self._prepare(following)
        return self.prepared['observation'], torch.as_tensor(reward, device=self.device, dtype=torch.float32), \
            torch.as_tensor(terminated, device=self.device).bool(), torch.as_tensor(truncated, device=self.device).bool(), extras

    def metrics(self):
        return dict(steps=self.steps, episodes=self.episodes, successes=self.successes,
            collisions=self.collisions, timeout_bootstraps=self.timeout_bootstraps,
            invalid_fraction=self.invalid_steps/max(1, self.steps), mpc_failures=self.mpc_failures,
            reward_terms=self.reward_totals.copy())

    def close(self):
        try:
            self.worker.close()
        finally:
            try:
                self.planner.close()
                self.backend.close()
            finally:
                if self.log:
                    self.log.close()


def _huber(x):
    """Standard Huber loss (delta=1): quadratic inside |x|<=1, linear outside."""
    return np.where(np.abs(x) <= 1., .5*np.square(x), np.abs(x)-.5)


def direct_reward(before, after, command, previous, geometry, dt, cfg, stop=None,
                  return_info=False):
    """Phase-aware trajectory and terminal reward for the MPC-free tracker.

    ``geometry`` is the per-env ``ReferenceTracker.reward_geometry()`` snapshot
    captured when the action was computed; the reward for a_t is settled against
    that old path even if a new plan arrives before the transition completes.
    ``stop`` is the per-env safety/emergency mask; forced stops are exempt from
    action-dependent penalties. With ``return_info=True``, also return raw
    geometry/safety diagnostics for scale audits (never summed into reward).
    """
    command = np.asarray(command, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    batch = command.shape[0]
    pos_before = np.asarray(before['position'], dtype=np.float32)[:, :2]
    pos_after = np.asarray(after['position'], dtype=np.float32)[:, :2]
    goal_before = np.asarray(before['goal_xy'], dtype=np.float32)
    goal_after = np.asarray(after['goal_xy'], dtype=np.float32)
    vmax = float(cfg['max_v'])

    # Project both poses into the OLD path's bounded progress neighbourhood.
    # Arc progress handles curved paths more faithfully than a single tangent
    # dot product and cannot jump to an unrelated part of a loop.
    delta_s = np.zeros(batch, dtype=np.float32)
    e_before = np.zeros(batch, dtype=np.float32)
    e_after = np.zeros(batch, dtype=np.float32)
    remaining_arc = np.full(batch, np.inf, dtype=np.float32)
    path_valid = np.zeros(batch, dtype=np.float32)
    for i in range(batch):
        g = None if geometry is None else geometry[i]
        if g is None or len(g['world']) < 2:
            continue
        progress = float(g.get('progress', 0.))
        arc_min = progress-.05
        arc_max = progress+float(g.get('lookahead_m', 1.))
        _, e_before[i], s_before = project_to_path(
            g['world'], g['arc'], pos_before[i], arc_min, arc_max)
        _, e_after[i], s_after = project_to_path(
            g['world'], g['arc'], pos_after[i], arc_min, arc_max)
        path_valid[i] = 1.
        delta_s[i] = s_after-s_before
        remaining_arc[i] = max(0., float(g['arc'][-1])-s_after)
    delta_s = path_valid*np.clip(delta_s, -vmax*dt, vmax*dt)

    # Δg: goal distance decrease.
    g_before = np.linalg.norm(goal_before, axis=-1)
    g_after = np.linalg.norm(goal_after, axis=-1)
    delta_g = np.clip(g_before-g_after, -vmax*dt, vmax*dt)

    # Improvement in cross-track error supplies a directional rejoin signal;
    # the corridor cost prevents reward-neutral parallel tracking off the path.
    delta_e = path_valid*np.clip(e_before-e_after, -vmax*dt, vmax*dt)
    corridor = np.square(np.maximum(0.,
        (e_after-cfg['tracking_free_band'])/cfg['tracking_error_scale']))
    corridor = path_valid*np.clip(corridor, 0., 4.)

    clearance = np.asarray(after['clearance'], dtype=np.float32)
    ttc = np.asarray(after['ttc'], dtype=np.float32)
    closing = np.asarray(after['closing'], dtype=np.float32)
    danger = (clearance < cfg['danger_clearance']) | (ttc < cfg['danger_ttc'])
    g_safe = np.where(danger, cfg['g_safe'], 1.).astype(np.float32)
    clear_scale = max(cfg['clearance_ref']-cfg['clearance_hard'], 1e-6)
    c_clear = np.clip(np.square(np.maximum(0.,
        (cfg['clearance_ref']-clearance)/clear_scale)), 0., 4.)
    c_ttc = closing*np.square(np.maximum(0., (cfg['ttc_ref']-ttc)/cfg['ttc_ref']))

    dv = (command[:, 0]-previous[:, 0])/(cfg['smooth_dv']*dt)
    dw = (command[:, 1]-previous[:, 1])/(cfg['smooth_dw']*dt)
    c_smooth = np.clip(np.square(dv)+np.square(dw), 0., 4.)
    if stop is not None:
        c_smooth = np.where(np.asarray(stop, dtype=bool), 0., c_smooth)

    velocity_before = np.asarray(before['velocity'], dtype=np.float32)
    velocity = np.asarray(after['velocity'], dtype=np.float32)
    v_xy_before = np.linalg.norm(velocity_before[:, :2], axis=-1)
    yaw_rate_before = np.abs(velocity_before[:, 5])
    v_xy = np.linalg.norm(velocity[:, :2], axis=-1)
    yaw_rate = np.abs(velocity[:, 5])
    # Fade trajectory tracking into PointGoal approach before the stopping
    # radius. This prevents a delayed local path from rewarding motion through
    # the goal, while retaining X-NavDP's detours away from the approach zone.
    stop_distance = float(cfg['goal_stop_distance'])
    approach_distance = max(float(cfg['goal_approach_distance']), stop_distance+1e-6)
    path_reward_scale = np.clip(
        (g_after-stop_distance)/(approach_distance-stop_distance), 0., 1.)
    approach_zone = (g_after > stop_distance) & (g_after <= approach_distance)
    goal_progress_zone = g_after <= approach_distance
    final_zone = g_after <= stop_distance

    # Begin braking before entering the success radius. Only excess speed is
    # penalized, so slow obstacle avoidance and turns remain available.
    v_ref = vmax*path_reward_scale
    yaw_ref = float(cfg['goal_approach_yaw_rate_max'])*path_reward_scale
    c_approach_speed = approach_zone*(
        np.square(np.maximum(0., v_xy-v_ref)/max(vmax, 1e-6)) +
        np.square(np.maximum(0., yaw_rate-yaw_ref)/max(
            float(cfg['goal_approach_yaw_rate_max']), 1e-6)))
    # Use excess above the success thresholds, not the much larger action
    # bounds. This supplies gradient close to success without penalizing a
    # state that already satisfies the terminal velocity limits.
    stop_v_scale = max(float(cfg['goal_stop_speed_scale']), 1e-6)
    stop_w_scale = max(float(cfg['goal_stop_yaw_rate_scale']), 1e-6)
    motion_before = np.square(np.maximum(
        0., v_xy_before-float(cfg['goal_stop_speed']))/stop_v_scale) + np.square(
            np.maximum(0., yaw_rate_before-float(cfg['goal_stop_yaw_rate']))/stop_w_scale)
    motion_after = np.square(np.maximum(
        0., v_xy-float(cfg['goal_stop_speed']))/stop_v_scale) + np.square(
            np.maximum(0., yaw_rate-float(cfg['goal_stop_yaw_rate']))/stop_w_scale)
    motion_before = np.clip(motion_before, 0., 4.)
    motion_after = np.clip(motion_after, 0., 4.)
    c_goal_stop = final_zone*motion_after
    goal_brake = final_zone*np.clip(motion_before-motion_after,
        -cfg['goal_brake_delta_clip'], cfg['goal_brake_delta_clip'])
    capture_score = final_zone*np.exp(
        -np.square(g_after/max(float(cfg['goal_capture_distance_scale']), 1e-6))
        -np.square(v_xy/max(float(cfg['goal_capture_speed_scale']), 1e-6))
        -np.square(yaw_rate/max(float(cfg['goal_capture_yaw_rate_scale']), 1e-6)))
    command_stop_cost = final_zone*np.clip(
        np.square(command[:, 0]/max(float(cfg['goal_command_speed_scale']), 1e-6)) +
        np.square(command[:, 1]/max(float(cfg['goal_command_yaw_rate_scale']), 1e-6)),
        0., 4.)
    hold_ready = (final_zone & (v_xy <= cfg['goal_stop_speed']) &
        (yaw_rate <= cfg['goal_stop_yaw_rate']))

    stall = np.asarray(after['stall'], dtype=np.float32)
    contact = np.asarray(after['contact'], dtype=np.float32)
    success = np.asarray(after['success'], dtype=np.float32)
    collision = np.asarray(after['collision'], dtype=np.float32)
    fallen = np.asarray(after['fallen'], dtype=np.float32)
    oob = np.asarray(after['oob'], dtype=np.float32)

    stopped = np.zeros(batch, dtype=bool) if stop is None else np.asarray(stop, dtype=bool)
    active_stall = stall*(~danger)*(~stopped)
    parts = dict(
        progress_s=cfg['weight_progress_s']*path_reward_scale*delta_s,
        progress_g=cfg['weight_progress_g']*delta_g,
        tracking_recovery=cfg['weight_tracking_recovery']*g_safe*path_reward_scale*delta_e,
        tracking=-cfg['weight_tracking_corridor']*g_safe*path_reward_scale*corridor*dt,
        clearance=-cfg['weight_clearance']*c_clear*dt,
        ttc=-cfg['weight_ttc']*c_ttc*dt,
        smoothness=-cfg['weight_smoothness']*c_smooth*dt,
        goal_approach_progress=(cfg['weight_goal_approach_progress']*g_safe*
            goal_progress_zone*delta_g),
        goal_approach_speed=-cfg['weight_goal_approach_speed']*c_approach_speed*dt,
        goal_stop=-cfg['weight_goal_stop']*c_goal_stop*dt,
        goal_brake=cfg['weight_goal_brake']*goal_brake,
        goal_capture=cfg['weight_goal_capture']*capture_score*dt,
        goal_command_stop=-cfg['weight_goal_command_stop']*command_stop_cost*dt,
        goal_hold=cfg['weight_goal_hold']*hold_ready*dt,
        stall=-cfg['weight_stall']*active_stall*dt,
        contact=-cfg['weight_contact']*contact*dt,
        time=-np.full(batch, cfg['weight_time']*dt),
        success=cfg['weight_success']*success,
        collision=-cfg['weight_collision']*collision,
        fall=-cfg['weight_fall']*fallen,
        oob=-cfg['weight_oob']*oob)
    if not return_info:
        return parts
    info = dict(delta_s=delta_s, cross_track_before=e_before,
        cross_track_after=e_after, delta_cross_track=delta_e,
        remaining_path_arc=remaining_arc, clearance=clearance, ttc=ttc,
        danger=danger.astype(np.float32), final_zone=final_zone.astype(np.float32),
        approach_zone=approach_zone.astype(np.float32),
        goal_progress_zone=goal_progress_zone.astype(np.float32),
        path_reward_scale=path_reward_scale, reference_speed=v_ref,
        reference_yaw_rate=yaw_ref, linear_speed=v_xy, yaw_rate=yaw_rate,
        goal_stop_cost=c_goal_stop, goal_capture_score=capture_score,
        goal_command_stop_cost=command_stop_cost,
        hold_ready=hold_ready.astype(np.float32))
    return parts, info


class DirectReactiveTrainEnv:
    """MPC-free one-control-step RGB-D tracker environment (no Acados / no MPC).

    Mirrors ReactiveTrainEnv's async contract (frozen RGB-D worker + planner,
    latest-only feature buffer, pre-reset terminal capture) but the policy
    directly commands [v, w] and reward v1 uses a non-advancing path snapshot.
    """
    def __init__(self, backend, encoder, planner, config, device='cpu', log_path=None):
        self.backend, self.encoder, self.planner = backend, encoder, planner
        self.num_envs = backend.num_envs
        self.cfg, self.device = config, torch.device(device)
        self.dt = backend.dt
        self.worker = LatestRGBDWorker(encoder)
        observer = ReactiveActorCritic(PolicyConfig(**config['policy'])).to(device)
        rt = config['runtime']
        self.control = DirectReactiveExecutor(observer, self.num_envs, DirectLimits(**config['actions']),
            rt['max_plan_age_s'], rt['max_rgbd_age_s'])
        self.frame_id = 0
        self.frame = None
        self.last_feature_time = self.last_plan_time = -float('inf')
        self.state = self.prepared = self.geometry = None
        self.steps = self.episodes = self.invalid_steps = 0
        self.timeout_bootstraps = self.successes = self.collisions = self.falls = self.oob = 0
        self.reward_totals = {}
        self.log = open(log_path, "w", newline="") if log_path else None
        self.writer = None

    def _clear(self, now):
        self.worker.reset()
        self.planner.reset()
        for i in range(self.num_envs):
            self.control.reset_env(i, self.planner.epoch, now)
        self.control.last_step_time = None
        self.frame = None
        self.last_feature_time = self.last_plan_time = -float('inf')

    def _publish(self, plan):
        if plan:
            for i in range(self.num_envs):
                self.control.publish_plan(i, plan['trajectory'][i], plan['positions'][i],
                    plan['quaternions'][i], plan['capture_time'], plan['plan_id'], plan['epoch'])

    def _prepare(self, state, warm=False, terminal=False):
        now = state['now']
        fresh = self.frame != state['frame']
        if fresh and now-self.last_feature_time >= self.cfg['runtime']['feature_period_s']-1e-6:
            self.frame_id += 1
            self.worker.submit(state['rgb'], state['depth'], self.frame_id, now,
                state['position'], state['quaternion'])
            self.frame, self.last_feature_time = state['frame'], now
        if not terminal:
            self._publish(self.planner.poll())
            if fresh and now-self.last_plan_time >= self.cfg['navigation_training']['planner_period_s']:
                if self.planner.submit(state):
                    self.last_plan_time = now
        if warm:
            self._publish(self.planner.poll(wait=True))
            deadline = time.monotonic()+60.
            while self.worker.snapshot() is None and time.monotonic() < deadline:
                time.sleep(.005)
            if self.worker.snapshot() is None:
                raise TimeoutError('Initial RGB-D encoder did not complete')
        packet = self.worker.snapshot()
        if packet is None:
            pc = self.control.policy.config
            tokens = torch.zeros(self.num_envs, pc.grid_size**2, pc.token_dim, device=self.device)
            packet = FeaturePacket(dict(rgb_tokens=tokens, depth_tokens=tokens.clone(),
                depth_valid_fraction=torch.zeros(self.num_envs, device=self.device)), -1,
                now-self.cfg['runtime']['max_rgbd_age_s']-1., state['position'], state['quaternion'], self.worker.epoch)
        prepared = self.control.step(packet, state['position'], state['quaternion'],
            state['velocity'], state['gravity'], state['goal_xy'], now, self.dt, observe_only=True)
        self.geometry = [t.reward_geometry() for t in self.control.trackers]
        return prepared

    def reset(self):
        self.state = self.backend.reset()
        self._clear(self.state['now'])
        # First reset has no prior-episode image; allow its freshly captured initial frame.
        self.control.reset_time[:] = -float('inf')
        self.prepared = self._prepare(self.state, warm=True)
        return self.prepared['observation']

    def step(self, latent):
        prepared = self.prepared
        previous = self.control.previous.detach().cpu().numpy().copy()
        result = self.control.mapper(latent, self.control.previous, self.dt, prepared['stop'])
        old_previous = self.control.previous.clone()
        self.control.previous_delta.copy_(result['command']-old_previous)
        self.control.previous.copy_(result['command'])
        following, terminated, truncated, terminal_state = self.backend.step(result['command'].detach().cpu().numpy())
        done_mask = (terminated | truncated).astype(bool)
        end = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in following.items()}
        if terminal_state is not None and done_mask.any():
            for key in end:
                ev, tv = end[key], terminal_state[key]
                if isinstance(ev, np.ndarray) and ev.ndim >= 1 and ev.shape[0] == self.num_envs:
                    ev[done_mask] = tv[done_mask]
        parts, reward_info = direct_reward(self.state, end,
            result['command'].detach().cpu().numpy(), previous, self.geometry, self.dt,
            self.cfg['reward'], stop=prepared['stop'].detach().cpu().numpy(), return_info=True)
        reward = sum(parts.values())
        if not np.isfinite(reward).all():
            raise FloatingPointError('Nonfinite navigation reward')
        self.steps += 1
        self.invalid_steps += int(prepared['stop'].sum())
        for key, value in parts.items():
            self.reward_totals[key] = self.reward_totals.get(key, 0.)+float(np.asarray(value).sum())
        if self.log:
            commands = result['command'].detach().cpu().numpy()
            for i in range(self.num_envs):
                row = dict(step=self.steps, env=i, sim_time=end['now'],
                    plan_id=self.control.trackers[i].plan_id,
                    plan_age_s=float(prepared['plan_age'][i]),
                    stopped=bool(prepared['stop'][i]),
                    goal_before=float(np.linalg.norm(self.state['goal_xy'][i])),
                    goal_after=float(np.linalg.norm(end['goal_xy'][i])),
                    command_v=float(commands[i, 0]), command_w=float(commands[i, 1]),
                    terminated=bool(terminated[i]), truncated=bool(truncated[i]),
                    collision=bool(end['collision'][i]), fallen=bool(end['fallen'][i]),
                    oob=bool(end['oob'][i]), success=bool(end['success'][i]), reward=float(reward[i]),
                    **{key: float(np.asarray(value)[i]) for key, value in reward_info.items()},
                    **{f'reward_{key}': float(np.asarray(value)[i]) for key, value in parts.items()})
                if self.writer is None:
                    self.writer = csv.DictWriter(self.log, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
            self.log.flush()
        extras = {}
        if done_mask.any():
            extras['terminal_observation'] = self._prepare(end, terminal=True)['observation'].clone()
            self.timeout_bootstraps += int(((truncated & ~terminated).astype(bool)).sum())
            self.episodes += int(done_mask.sum())
            self.successes += int(np.asarray(end['success'])[done_mask].sum())
            self.collisions += int(np.asarray(end['collision'])[done_mask].sum())
            self.falls += int(np.asarray(end['fallen'])[done_mask].sum())
            self.oob += int(np.asarray(end['oob'])[done_mask].sum())
            self._clear(following['now'])
        self.state = following
        self.prepared = self._prepare(following)
        return self.prepared['observation'], torch.as_tensor(reward, device=self.device, dtype=torch.float32), \
            torch.as_tensor(terminated, device=self.device).bool(), torch.as_tensor(truncated, device=self.device).bool(), extras

    def metrics(self):
        return dict(steps=self.steps, episodes=self.episodes, successes=self.successes,
            collisions=self.collisions, falls=self.falls, oob=self.oob,
            timeout_bootstraps=self.timeout_bootstraps,
            invalid_fraction=self.invalid_steps/max(1, self.steps*self.num_envs),
            reward_terms=self.reward_totals.copy())

    def close(self):
        try:
            self.worker.close()
        finally:
            try:
                self.planner.close()
                self.backend.close()
            finally:
                if self.log:
                    self.log.close()
