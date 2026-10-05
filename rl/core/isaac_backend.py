"""Isaac G1 backend; captures physical terminal state before automatic reset.

Uses the existing scene/locomotion pipeline. Replaces termination terms only
inside this training instance; original evaluation SR definitions are untouched.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from copy import deepcopy
import math
import numpy as np
import torch


class IsaacReactiveBackend:
    def __init__(self, env, controller, config):
        self.wrapper, self.env, self.controller = env, env.unwrapped, controller
        self.num_envs, self.dt = env.num_envs, self.env.step_dt
        self.cfg = config
        self.obs = None
        self.active = False
        self.hold = np.zeros(self.num_envs)
        self.stall_hold = np.zeros(self.num_envs)
        self.collision_hold = np.zeros(self.num_envs)
        self.last_clearance = np.full(self.num_envs, 5., dtype=np.float32)
        self.last_clearance_time = np.full(self.num_envs, -np.inf, dtype=np.float64)
        self.last_ttc = np.full(self.num_envs, 10., dtype=np.float32)
        self.last_closing = np.zeros(self.num_envs, dtype=np.float32)
        self.outcome = dict(success=np.zeros(self.num_envs, bool), collision=np.zeros(self.num_envs, bool),
                            fallen=np.zeros(self.num_envs, bool), oob=np.zeros(self.num_envs, bool))
        self.terminal = None
        self.original_reset = self.env._reset_idx
        self.env._reset_idx = self._before_reset
        manager = self.env.termination_manager
        arrival = deepcopy(manager.get_term_cfg('arrive_goal'))
        arrival.func, arrival.params = self._terminate, {}
        manager.set_term_cfg('arrive_goal', arrival)
        timeout = deepcopy(manager.get_term_cfg('time_out'))
        timeout.func, timeout.params = self._timeout, {}
        manager.set_term_cfg('time_out', timeout)
        # Fail early if the intended non-foot collision sensor is absent.
        self.env.scene['contact_sensor']

    def _goal(self):
        from src.environment.tasks.observation_utils import oracle_imu_pose_data
        return oracle_imu_pose_data(self.env).detach().cpu().numpy().copy()

    def _stall_step(self, speed, goal):
        """Per-step stall: moving slowly while still away from the goal."""
        stall_speed = self.cfg.get('stall_speed_mps', self.cfg.get('success_speed_mps', .05))
        near_goal = np.linalg.norm(goal[:, :2], axis=-1) <= self.cfg['success_distance_m']
        return (speed < stall_speed) & ~near_goal

    def _terminate(self, env):
        data = env.scene['robot'].data
        elapsed = env.episode_length_buf.detach().cpu().numpy()*self.dt
        goal = self._goal()
        speed = torch.linalg.vector_norm(data.root_lin_vel_b[:, :2], dim=-1).cpu().numpy()
        force = env.scene['contact_sensor'].data.net_forces_w
        contact = (torch.linalg.vector_norm(force, dim=-1).amax(-1).cpu().numpy()
                   > self.cfg['collision_force_n']) & (elapsed >= self.cfg['contact_grace_s'])
        # A brief brush only penalises the reward; sustained contact terminates.
        hold_seconds = self.cfg.get('collision_hold_seconds', 0.0)
        if hold_seconds <= 0:
            collision = contact  # legacy: immediate termination on contact
        else:
            collision_hold = getattr(self, 'collision_hold', np.zeros_like(speed))
            self.collision_hold = np.where(contact, collision_hold+self.dt, 0.)
            collision = self.collision_hold >= hold_seconds
        fallen = ((data.root_pos_w[:, 2].cpu().numpy() < self.cfg['fall_height_m']) |
                  (data.projected_gravity_b[:, 2].cpu().numpy() > -.3)) & (elapsed >= self.cfg['contact_grace_s'])
        reached = (np.linalg.norm(goal[:, :2], axis=-1) <= self.cfg['success_distance_m']) & (speed <= self.cfg['success_speed_mps'])
        if self.cfg.get('success_yaw_rate') is not None:
            reached = reached & (np.abs(data.root_ang_vel_b[:, 2].cpu().numpy()) <= self.cfg['success_yaw_rate'])
        oob = self._oob(data.root_pos_w.cpu().numpy())
        stall_step = self._stall_step(speed, goal)
        stall_hold = getattr(self, 'stall_hold', np.zeros_like(speed))
        self.stall_hold = np.where(stall_step & ~collision & ~fallen, stall_hold+self.dt, 0.)
        stall = self.stall_hold >= self.cfg.get('stall_seconds', float('inf'))
        self.hold = np.where(reached & ~collision & ~fallen, self.hold+self.dt, 0.)
        success = self.hold >= self.cfg['success_hold_s']
        self.outcome = dict(success=success, collision=collision, fallen=fallen, oob=oob)
        return torch.as_tensor(success | collision | fallen | oob | stall, device=env.device)

    def _timeout(self, env):
        return env.episode_length_buf >= max(1, math.ceil(self.cfg['episode_seconds']/self.dt))

    def _oob(self, position):
        # Prefer an explicit navigable bounding box (from the scene ESDF); fall back
        # to a radius-from-origin only for legacy configs, else no OOB check.
        if 'oob_min' in self.cfg and 'oob_max' in self.cfg:
            min_xy = np.asarray(self.cfg['oob_min'], dtype=float)
            max_xy = np.asarray(self.cfg['oob_max'], dtype=float)
            xy = np.asarray(position, dtype=float)[:, :2]
            return ((xy < min_xy) | (xy > max_xy)).any(axis=-1)
        radius = self.cfg.get('oob_distance_m')
        if radius is None:
            return np.zeros(position.shape[0], dtype=bool)
        return np.linalg.norm(position[:, :2], axis=-1) > radius

    def _clearance(self, depth, far=5.0, bottom_floor=None, percentile=None):
        """Robust obstacle range inside G1's swept-body camera corridor.

        Back-projecting image columns lets the reward cover shoulders and
        swinging arms instead of protecting only the pelvis centerline. A low
        percentile remains robust to isolated invalid depth pixels.
        """
        if depth.ndim == 4:
            depth = depth[..., 0]  # B x H x W x 1 -> B x H x W
        if bottom_floor is None:
            bottom_floor = self.cfg.get('depth_floor_crop_fraction', .3)
        if percentile is None:
            percentile = self.cfg.get('clearance_percentile', 1.)
        batch, height, width = depth.shape
        cropped = depth[:, :int(height*(1-bottom_floor)), :]
        calibration_width = float(self.cfg.get('depth_calibration_width_px', width))
        width_scale = width/max(calibration_width, 1.)
        fx = float(self.cfg.get('depth_fx_px', 326.39856))*width_scale
        cx = float(self.cfg.get('depth_cx_px', (calibration_width-1.)*.5))*width_scale
        columns = np.arange(width, dtype=np.float32)[None, None, :]
        lateral = np.abs((columns-cx)*cropped/max(fx, 1e-6))
        half_width = float(self.cfg.get('clearance_half_width_m', float('inf')))
        valid_image = ((cropped > .1) & (cropped <= 5.) &
                       (lateral <= half_width))
        band = cropped.reshape(batch, -1)
        valid = valid_image.reshape(batch, -1)
        clearance = np.full(batch, far, dtype=np.float32)
        for i in range(depth.shape[0]):
            if valid[i].any():
                clearance[i] = np.percentile(band[i, valid[i]], percentile)
        return clearance

    def _ttc(self, clearance, now, far=10.0):
        """Camera-space TTC from measured clearance rate across simulation steps."""
        clearance = np.asarray(clearance, dtype=np.float32)
        dt = float(now)-self.last_clearance_time
        valid = np.isfinite(dt) & (dt > 1e-6)
        closing_speed = np.zeros(self.num_envs, dtype=np.float32)
        closing_speed[valid] = ((self.last_clearance[valid]-clearance[valid])/
            dt[valid]).astype(np.float32)
        closing = closing_speed > self.cfg.get('ttc_min_closing_speed_mps', .05)
        hard = self.cfg.get('clearance_hard_m', .25)
        ttc = np.full(self.num_envs, far, dtype=np.float32)
        ttc[closing] = (np.maximum(clearance[closing]-hard, 0.) /
            np.maximum(closing_speed[closing], 1e-3))
        advanced = valid | ~np.isfinite(self.last_clearance_time)
        self.last_clearance[advanced] = clearance[advanced]
        self.last_clearance_time[advanced] = float(now)
        self.last_ttc[advanced] = ttc[advanced]
        self.last_closing[advanced] = closing[advanced]
        return self.last_ttc.copy(), self.last_closing.copy()

    def snapshot(self):
        data = self.env.scene['robot'].data
        camera = self.env.scene['camera_sensor']
        # Raw camera access avoids observation_manager.compute(): that would mutate RGB history twice.
        output = camera.data.output
        def cpu(tensor):
            return tensor.detach().cpu().numpy().copy()
        position = cpu(data.root_pos_w)
        velocity = cpu(torch.cat((data.root_lin_vel_b, data.root_ang_vel_b), -1))
        goal = self._goal()
        depth = cpu(output['distance_to_image_plane'])
        clearance = self._clearance(depth)
        now = float(self.env.common_step_counter*self.dt)
        ttc, closing = self._ttc(clearance, now)
        stall = self._stall_step(np.linalg.norm(velocity[:, :2], axis=-1), goal)
        force = self.env.scene['contact_sensor'].data.net_forces_w
        contact = (torch.linalg.vector_norm(force, dim=-1).amax(-1).cpu().numpy()
                   > self.cfg['collision_force_n'])
        return dict(now=now,
            position=position, quaternion=cpu(data.root_quat_w[:, [1, 2, 3, 0]]),
            velocity=velocity, gravity=cpu(data.projected_gravity_b), goal_xy=goal[:, :2],
            rgb=cpu(output['rgb'][..., :3]), depth=depth, frame=int(camera.frame[0]),
            clearance=clearance, ttc=ttc, closing=closing, stall=stall, contact=contact,
            **{k: v.copy() for k, v in self.outcome.items()})

    def _before_reset(self, ids):
        if self.active:
            self.terminal = self.snapshot()
        self.original_reset(ids)
        indices = ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids)
        self.last_clearance[indices] = 5.
        self.last_clearance_time[indices] = -np.inf
        self.last_ttc[indices] = 10.
        self.last_closing[indices] = 0.
        self.hold[indices] = 0
        self.stall_hold[indices] = 0
        self.collision_hold[indices] = 0
        for value in self.outcome.values():
            value[indices] = False

    def reset(self):
        self.active = False
        self.obs, _ = self.env.reset()
        self.terminal = None
        self.active = True
        return self.snapshot()

    def step(self, command):
        self.terminal = None
        action = self.controller.forward_batch(self.obs['policy'], command)
        # Bypass RSL wrapper to retain separate terminated/truncated arrays.
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        return self.snapshot(), terminated.cpu().numpy().copy(), truncated.cpu().numpy().copy(), self.terminal

    def close(self):
        self.env._reset_idx = self.original_reset
        self.wrapper.close()
