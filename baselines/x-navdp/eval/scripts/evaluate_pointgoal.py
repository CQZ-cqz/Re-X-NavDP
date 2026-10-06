"""
NavDP Evaluation Script for Point-Goal Navigation.

This script runs evaluation of the NavDP diffusion-based navigation policy
using Isaac Lab as the simulation environment.
"""

import sys
import os
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

import cv2
import numpy as np
import imageio
import torch
import argparse
import time
import csv
import json
import yaml
import threading
import traceback
import random
from copy import deepcopy

from eval.src import navigator_reset, navigator_shutdown, pointgoal_step
from eval.config_utils import load_default_config
from eval.control_trace import ControlTraceRecorder
from bridge.recovery.execution import ExecutionHistory
from eval.environment import create_environment, BatchMPCNEWController, namespace_to_dict


input_obs = None
input_lock = threading.Lock()
output_action = None
output_action_version = 0
output_plan = None
output_lock = threading.Lock()
planning_rpc_lock = threading.Lock()
output_epoch = 0
stop_event = threading.Event()
simulation_app = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate; defaults to all samples in the scene.",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Evaluation scene index; defaults to environment.scene_index in the YAML config.",
    )
    parser.add_argument("--server_port", type=int, default=19999)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override environment.num_envs to evaluate this many episodes in parallel.",
    )
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop cleanly after this many simulation steps (useful for smoke tests).",
    )
    parser.add_argument(
        "--record_num",
        type=int,
        default=None,
        help="Max episodes to record video for per scene (random subset). Default: record all.",
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=3.0,
        help="Contact-force threshold (N) above which an episode is flagged as collided.",
    )
    parser.add_argument(
        "--keep_server",
        action="store_true",
        help="Do not shut down the policy server at the end (for multi-scene runs).",
    )
    parser.add_argument("--preserve_blanket_scale", action="store_true",
        help="Preserve authored blanket geometry; use for comparisons with reactive training.")
    exec_mode = parser.add_mutually_exclusive_group()
    exec_mode.add_argument("--reactive_zero", action="store_true",
        help="Enable aligned per-step MPC with a zero-initialized residual actor (integration baseline).")
    exec_mode.add_argument("--reactive_checkpoint", type=str,
        help="Trained residual actor checkpoint; enables reactive execution.")
    exec_mode.add_argument("--direct_zero", action="store_true",
        help="Enable MPC-free direct tracking with a zero-initialized actor (integration baseline).")
    exec_mode.add_argument("--direct_checkpoint", type=str,
        help="Trained direct tracker checkpoint; enables MPC-free execution.")
    parser.add_argument("--reactive_encoder_checkpoint", type=str,
        default=str(Path(_REPO_ROOT)/"../../checkpoints/x-navdp_posttrain.ckpt"))
    parser.add_argument("--reactive_config", default=str(Path(_REPO_ROOT)/"../../rl/config/reactive_rgbd_g1.yaml"))
    parser.add_argument("--direct_config", default=str(Path(_REPO_ROOT)/"../../rl/config/reactive_rgbd_direct_g1.yaml"))
    parser.add_argument("--reactive_device", default="cpu")
    parser.add_argument("--strict_pointgoal", action="store_true",
        help="Use continuous PointGoal distance/speed/yaw hold for success. Direct mode enables this automatically.")
    parser.add_argument("--strict_success_distance", type=float,
        help="Strict success radius in metres (default: direct YAML or 0.35).")
    parser.add_argument("--strict_success_speed", type=float,
        help="Strict planar-speed limit in m/s (default: direct YAML or 0.12).")
    parser.add_argument("--strict_success_yaw_rate", type=float,
        help="Strict absolute yaw-rate limit in rad/s (default: direct YAML or 0.40).")
    parser.add_argument("--strict_success_hold", type=float,
        help="Required continuous hold in seconds (default: direct YAML or 0.25).")
    args = parser.parse_args()
    if (args.reactive_zero or args.reactive_checkpoint or args.direct_zero or args.direct_checkpoint) and not args.reactive_encoder_checkpoint:
        parser.error("Reactive/direct execution requires --reactive_encoder_checkpoint")
    if args.reactive_zero or args.reactive_checkpoint:
        for candidate in (args.reactive_config, args.reactive_encoder_checkpoint, args.reactive_checkpoint):
            if candidate and not Path(candidate).is_file():
                parser.error(f"Reactive input file does not exist: {candidate}")
    if args.direct_zero or args.direct_checkpoint:
        for candidate in (args.direct_config, args.reactive_encoder_checkpoint, args.direct_checkpoint):
            if candidate and not Path(candidate).is_file():
                parser.error(f"Direct input file does not exist: {candidate}")
    return args


def parse_observations(observation, obs_mapping, return_tensor=False):
    """Parse observations according to the mapping configuration."""
    return_observations = {}
    if hasattr(obs_mapping, "__dict__"):
        obs_mapping = vars(obs_mapping)
    for key, value in obs_mapping.items():
        if value not in observation:
            continue
        if return_tensor:
            return_observations[key] = observation[value]
        else:
            return_observations[key] = observation[value].cpu().numpy()
    return return_observations


def add_robot_state(observation, env):
    """Attach world-frame robot pose used by temporal guidance and stuck detection."""
    observation = dict(observation)
    robot = env.unwrapped.scene["robot"]
    observation["robot_pose"] = robot.data.root_pos_w
    # Isaac Lab uses wxyz quaternions; SciPy guidance code expects xyzw.
    observation["robot_rot"] = robot.data.root_quat_w[:, [1, 2, 3, 0]]
    return observation


def write_metrics(metrics, path="exploration.csv"):
    """Write evaluation metrics to CSV file."""
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


class StrictPointGoalTermination:
    """Continuous PointGoal hold criterion shared with direct-policy training."""

    def __init__(self, num_envs, dt, distance=.35, speed=.12, yaw_rate=.40, hold=.25):
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.distance = float(distance)
        self.speed = float(speed)
        self.yaw_rate = float(yaw_rate)
        self.hold_seconds = float(hold)
        self.hold = None
        self.last_episode_step = None
        self.last_success = np.zeros(self.num_envs, dtype=bool)
        self.last_distance = np.full(self.num_envs, np.inf, dtype=np.float32)
        self.last_speed = np.full(self.num_envs, np.inf, dtype=np.float32)
        self.last_yaw_rate = np.full(self.num_envs, np.inf, dtype=np.float32)

    def update(self, distance, speed, yaw_rate, episode_step):
        """Advance the hold timer once for the current environment step."""
        distance = torch.as_tensor(distance)
        speed = torch.as_tensor(speed, device=distance.device)
        yaw_rate = torch.as_tensor(yaw_rate, device=distance.device)
        episode_step = torch.as_tensor(episode_step, device=distance.device)
        if self.hold is None:
            self.hold = torch.zeros_like(distance, dtype=torch.float32)
            self.last_episode_step = torch.full_like(episode_step, -1)
        restarted = episode_step < self.last_episode_step
        base = torch.where(restarted, torch.zeros_like(self.hold), self.hold)
        eligible = ((distance <= self.distance) & (speed <= self.speed) &
                    (yaw_rate <= self.yaw_rate))
        new_step = episode_step != self.last_episode_step
        increment = torch.where(new_step, torch.full_like(base, self.dt), torch.zeros_like(base))
        self.hold = torch.where(eligible, base+increment, torch.zeros_like(base))
        self.last_episode_step = episode_step.clone()
        success = self.hold+1e-7 >= self.hold_seconds
        self.last_success = success.detach().cpu().numpy().astype(bool)
        self.last_distance = distance.detach().cpu().numpy().astype(np.float32)
        self.last_speed = speed.detach().cpu().numpy().astype(np.float32)
        self.last_yaw_rate = yaw_rate.detach().cpu().numpy().astype(np.float32)
        return success

    def __call__(self, env):
        from src.environment.tasks.observation_utils import oracle_imu_pose_data
        robot = env.scene['robot'].data
        goal = oracle_imu_pose_data(env)[:, :2]
        distance = torch.linalg.vector_norm(goal, dim=-1)
        speed = torch.linalg.vector_norm(robot.root_lin_vel_b[:, :2], dim=-1)
        yaw_rate = torch.abs(robot.root_ang_vel_b[:, 2])
        return self.update(distance, speed, yaw_rate, env.episode_length_buf)


def strict_success_settings(args, direct_enabled):
    values = dict(distance=.35, speed=.12, yaw_rate=.40, hold=.25)
    if direct_enabled:
        with open(args.direct_config, encoding="utf-8") as handle:
            navigation = yaml.safe_load(handle)['navigation_training']
        values.update(distance=float(navigation['success_distance_m']),
            speed=float(navigation['success_speed_mps']),
            yaw_rate=float(navigation['success_yaw_rate']),
            hold=float(navigation['success_hold_s']))
    for key, argument in (
            ('distance', args.strict_success_distance),
            ('speed', args.strict_success_speed),
            ('yaw_rate', args.strict_success_yaw_rate),
            ('hold', args.strict_success_hold)):
        if argument is not None:
            values[key] = float(argument)
    if min(values.values()) <= 0:
        raise ValueError(f"Strict success thresholds must be positive: {values}")
    return values


def install_strict_pointgoal(env, settings):
    """Replace the legacy latched arrival term without changing the environment."""
    tracker = StrictPointGoalTermination(env.num_envs, env.unwrapped.step_dt, **settings)
    manager = env.unwrapped.termination_manager
    arrival = deepcopy(manager.get_term_cfg('arrive_goal'))
    arrival.func = tracker
    arrival.params = {}
    manager.set_term_cfg('arrive_goal', arrival)
    return tracker


def planning_thread(server_port, mpc_controller, reactive=False):
    """Thread for running navigation planning asynchronously."""
    global input_obs, output_action, output_action_version, output_plan
    next_plan_id = 0
    while not stop_event.is_set():
        obs_to_process = None
        with input_lock:
            if input_obs is not None:
                obs_to_process = input_obs
                input_obs = None

        if obs_to_process is not None:
            try:
                epoch = obs_to_process.pop("_epoch")
                capture = obs_to_process.pop("_capture", None)
                with planning_rpc_lock:
                    if epoch != output_epoch or stop_event.is_set():
                        continue
                    next_plan_id += 1
                    obs_to_process["execution_feedback"]["plan_id"] = next_plan_id
                    output_trajectory, _, _ = pointgoal_step(**obs_to_process, port=server_port)
                    if reactive:
                        if capture is None:
                            raise ValueError("Reactive planning requires capture pose/time")
                        with output_lock:
                            output_plan = dict(trajectory=output_trajectory.copy(), epoch=epoch,
                                plan_id=next_plan_id, **capture)
                            output_action_version = next_plan_id
                    else:
                        output_trajectory = np.concatenate((np.zeros_like(output_trajectory[:, 0:1]), output_trajectory), axis=1)
                        mpc_controls, _, _, _ = mpc_controller.solve(output_trajectory)
                        with output_lock:
                            output_action = mpc_controls
                            output_action_version = next_plan_id
            except Exception:
                print("[planning_thread] policy/MPC step failed:")
                print(traceback.format_exc(), flush=True)
                with output_lock:
                    output_action = None

        time.sleep(0.01)


def main():
    global input_obs, input_lock, output_action, output_lock, simulation_app, output_epoch, output_plan

    args = get_args()
    device = args.device
    reactive_enabled = bool(args.reactive_zero or args.reactive_checkpoint)
    direct_enabled = bool(args.direct_zero or args.direct_checkpoint)
    bridge_enabled = reactive_enabled or direct_enabled
    strict_success_enabled = bool(args.strict_pointgoal or direct_enabled)
    strict_settings = strict_success_settings(args, direct_enabled) if strict_success_enabled else None
    cfg = load_default_config(args.config_file)
    if args.preserve_blanket_scale:
        cfg.environment.preserve_blanket_scale = True
    if args.num_envs is not None:
        cfg.environment.num_envs = args.num_envs
    scene_dir = cfg.environment.scene_dir
    # Set the MDL material search path before Isaac starts (mirrors the training worker,
    # whose configure_mdl_system_path must run before AppLauncher or MDL parsing hangs).
    from src.training.worker import configure_mdl_system_path
    configure_mdl_system_path(scene_dir)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        device=device
    )
    simulation_app = app_launcher.app
    scene_index = (
        args.scene_index
        if args.scene_index is not None
        else getattr(cfg.environment, "scene_index", 0)
    )
    if direct_enabled:
        mpc_controller = None  # Direct mode never instantiates an Acados solver.
    else:
        mpc_controller = BatchMPCNEWController(batch=cfg.environment.num_envs, **namespace_to_dict(cfg.mpc))

    planning_thread_obj = threading.Thread(target=planning_thread, args=(args.server_port, mpc_controller, bridge_enabled))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    env, controller, house_id = create_environment(cfg, scene_index=scene_index, device=device)
    strict_termination = None
    if strict_success_enabled:
        strict_termination = install_strict_pointgoal(env, strict_settings)
        print("Strict PointGoal success enabled: "
              f"distance<={strict_settings['distance']:.2f} m, "
              f"speed<={strict_settings['speed']:.2f} m/s, "
              f"|yaw_rate|<={strict_settings['yaw_rate']:.2f} rad/s for "
              f"{strict_settings['hold']:.2f} s", flush=True)

    reset_outputs = env.reset()
    if isinstance(reset_outputs, tuple) and len(reset_outputs) == 2:
        raw_obs, infos = reset_outputs
        obs = infos.get("observations", raw_obs)
    else:
        obs, infos = reset_outputs, {}
    obs = add_robot_state(obs, env)

    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]
    sample_indices = [env.unwrapped._sample_idx[i].item() for i in range(env.num_envs)]
    server_algo = navigator_reset(
        camera_intrinsic.cpu().numpy(),
        batch_size=env.num_envs,
        port=args.server_port,
        sample_indices=sample_indices,
        scene_name=house_id
    )

    scene_split = scene_dir.split('/')[-1]
    save_prefix = cfg.run_root_dir
    embodiment_name = {
        "dingo": "wheeled",
        "unitree_g1": "humanoid",
        "unitree_go2": "quadruped",
    }.get(getattr(cfg.environment, "embodiment", ""), getattr(cfg.environment, "embodiment", "unknown"))
    scene_kind = getattr(cfg.environment, "scene_type", scene_split)
    output_group = f"{embodiment_name}_{scene_kind}"
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = os.path.join(save_prefix, output_group, server_algo + time_stamp, scene_split, house_id)
    os.makedirs(save_dir, exist_ok=True)
    reactive_bridge = None
    if direct_enabled:
        control_mode = "direct-zero" if args.direct_zero else "direct-learned"
    elif reactive_enabled:
        control_mode = "reactive-zero" if args.reactive_zero else "reactive-learned"
    else:
        control_mode = "mpc"
    if direct_enabled:
        from bridge.planner_executor_bridge import DirectReactiveEvalBridge
        reactive_bridge = DirectReactiveEvalBridge.from_args(args, env.num_envs,
            os.path.join(save_dir, "reactive_control.csv"))
        with open(os.path.join(save_dir, "reactive_run.json"), "w") as handle:
            json.dump(dict(mode="direct-zero" if args.direct_zero else "direct-learned",
                args=vars(args), control_dt=env.unwrapped.step_dt,
                encoder_fingerprint=reactive_bridge.worker.encoder.fingerprint,
                visual_encoder_metadata=reactive_bridge.worker.encoder.metadata,
                capture_clock="simulation control tick at first observed camera update",
                strict_success=strict_settings,
                config=namespace_to_dict(cfg)), handle, indent=2)
        print("Direct execution enabled: MPC-free [v,w] command, current RGB-D, "
              + ("zero actor" if args.direct_zero else "learned direct tracker"), flush=True)
    elif reactive_enabled:
        from bridge.planner_executor_bridge import ReactiveEvalBridge
        reactive_bridge = ReactiveEvalBridge.from_args(args, mpc_controller, env.num_envs,
            os.path.join(save_dir, "reactive_control.csv"))
        with open(os.path.join(save_dir, "reactive_run.json"), "w") as handle:
            json.dump(dict(mode="zero" if args.reactive_zero else "learned",
                args=vars(args), control_dt=env.unwrapped.step_dt,
                encoder_fingerprint=reactive_bridge.worker.encoder.fingerprint,
                visual_encoder_metadata=reactive_bridge.worker.encoder.metadata,
                capture_clock="simulation control tick at first observed camera update",
                config=namespace_to_dict(cfg)), handle, indent=2)
        print("Reactive execution enabled: per-step aligned MPC, current RGB-D, "
              + ("zero residual" if args.reactive_zero else "learned residual"), flush=True)


    done_sample_indices = set()
    total_episode_count = env.unwrapped._total_episode_count
    if args.num_episodes is not None:
        total_episode_count = min(args.num_episodes, total_episode_count)

    # Random video subset: record only `record_num` episodes (by ordinal) per scene.
    record_ordinals = set(range(total_episode_count))
    if args.record_num is not None and args.record_num < total_episode_count:
        record_ordinals = set(random.sample(range(total_episode_count), args.record_num))
    episode_ordinal = 0

    def make_writer(sample_idx):
        nonlocal episode_ordinal
        record = episode_ordinal in record_ordinals
        episode_ordinal += 1
        if not record:
            return None
        return imageio.get_writer(save_dir + f"/fps_{sample_idx}.mp4", fps=20)

    fps_writers = []
    current_episode_idx = []
    for i in range(env.num_envs):
        sample_idx = env.unwrapped._sample_idx[i].item()
        if sample_idx not in done_sample_indices:
            fps_writers.append(make_writer(sample_idx))
            current_episode_idx.append(sample_idx)
        else:
            print(f"Warning: sample_idx {sample_idx} already done, skipping env {i}")
            fps_writers.append(None)
            current_episode_idx.append(None)
    control_trace = ControlTraceRecorder(save_dir, control_mode)

    with input_lock:
        input_obs = parse_observations(obs, cfg.obs_mapping)
        input_obs["_epoch"] = output_epoch
        if bridge_enabled:
            input_obs["_capture"] = dict(capture_time=0.,
                positions=obs["robot_pose"].cpu().numpy().copy(),
                quaternions=obs["robot_rot"].cpu().numpy().copy())
        input_obs["execution_feedback"] = {
            "sim_time": 0.0, "executed_segments": [[] for _ in range(env.num_envs)]}
    planner_camera_frames = env.unwrapped.scene.sensors['camera_sensor'].frame.cpu().numpy().copy()
    euclidean = np.sqrt(np.square(obs['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))
    trajectory_length = np.zeros((env.num_envs))
    evaluation_metrics = []
    collided = np.zeros(env.num_envs, dtype=bool)
    contact_threshold = args.collision_threshold
    try:
        env.unwrapped.scene["contact_sensor"]
        has_contact = True
    except KeyError:
        has_contact = False

    execution_history = ExecutionHistory(env.num_envs)
    output_action_version_last = 0
    step_count = 0

    while simulation_app.is_running():
        with output_lock:
            if output_action_version > output_action_version_last:
                output_action_version_last = output_action_version
                print("action update")
            if output_action is not None and output_action.shape[1] != 0:
                applied_plan_id = output_action_version
                first_action = output_action[:, 0].copy()
                output_action = output_action[:, 1:]
            else:
                first_action = None
                applied_plan_id = None

        if reactive_bridge is not None:
            now = float(step_count * env.unwrapped.step_dt)
            positions = obs["robot_pose"].cpu().numpy()
            quaternions = obs["robot_rot"].cpu().numpy()
            camera = env.unwrapped.scene.sensors['camera_sensor']
            reactive_bridge.submit_frame(obs['raw_rgb'].cpu().numpy(), obs['raw_depth'].cpu().numpy(),
                camera.frame.cpu().numpy(), now, positions, quaternions)
            with output_lock:
                plan_snapshot = output_plan
            reactive_bridge.publish(plan_snapshot)
            robot_data = env.unwrapped.scene['robot'].data
            first_action = reactive_bridge.step(positions, quaternions,
                torch.cat((robot_data.root_lin_vel_b, robot_data.root_ang_vel_b), -1).cpu().numpy(),
                robot_data.projected_gravity_b.cpu().numpy(), obs['goal_pose'][:, :2].cpu().numpy(),
                now, env.unwrapped.step_dt)
            applied_plan_id = reactive_bridge.accepted_plan if reactive_bridge.accepted_plan >= 0 else None

        command_valid = first_action is not None
        if command_valid:
            command = np.asarray(first_action, dtype=np.float32)
        else:
            command = np.zeros((env.num_envs, 2), dtype=np.float32)
        robot_action = controller.forward_batch(obs['policy'], command)

        # Final distance-to-goal of the *current* episode (captured before the step that
        # may terminate it; the env auto-resets on termination so post-step goal_pose is
        # already the next episode's).
        goal_dist_before_step = np.sqrt(
            np.square(obs['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))

        robot_data = env.unwrapped.scene['robot'].data
        control_trace.write_batch(
            sim_time=float(step_count * env.unwrapped.step_dt),
            episode_time=(env.unwrapped.episode_length_buf.detach().cpu().numpy()
                          * env.unwrapped.step_dt),
            episode_indices=current_episode_idx,
            plan_id=applied_plan_id,
            command_valid=command_valid,
            command=command,
            linear_velocity=robot_data.root_lin_vel_b.detach().cpu().numpy(),
            angular_velocity=robot_data.root_ang_vel_b.detach().cpu().numpy(),
            goal_distance=goal_dist_before_step)

        planar_speed = torch.linalg.vector_norm(obs['policy'][:, :2], dim=-1)
        trajectory_length += (planar_speed * env.unwrapped.step_dt).cpu().numpy()

        start_positions = obs["robot_pose"].detach().cpu().numpy().copy()
        start_quaternions = obs["robot_rot"].detach().cpu().numpy().copy()
        step_outputs = env.step(robot_action)
        if len(step_outputs) == 5:
            obs, rewards, terminated, truncated, infos = step_outputs
            dones = terminated | truncated
        else:
            raw_obs, rewards, dones, infos = step_outputs
            obs = infos.get("observations", raw_obs)
        obs = add_robot_state(obs, env)
        execution_history.record(
            applied_plan_id, env.unwrapped.step_dt, start_positions, start_quaternions,
            obs["robot_pose"].detach().cpu().numpy(), obs["robot_rot"].detach().cpu().numpy(),
            dones.detach().cpu().numpy())
        step_count += 1

        # Collision detection from the contact sensor. Skip done envs: they have already
        # been auto-reset, so their contact forces are the next episode's (~0).
        if has_contact:
            try:
                net_forces = env.unwrapped.scene["contact_sensor"].data.net_forces_w
                if net_forces.ndim == 3 and net_forces.shape[1] > 1:
                    force_mag = torch.norm(net_forces, dim=-1).max(dim=1).values
                else:
                    force_mag = torch.norm(net_forces[:, 0], dim=-1)
                collision_now = (force_mag > contact_threshold).cpu().numpy()
                collided |= collision_now & (~dones.cpu().numpy())
            except Exception:
                pass

        for i in range(env.num_envs):
            if fps_writers[i] is None:
                continue
            resize_raw_image = cv2.resize(obs['raw_rgb'][i].detach().cpu().numpy(), (384, 384))
            if 'birdeye_rgb' in obs:
                bev_image = obs['birdeye_rgb'][i].detach().cpu().numpy()
            else:
                bev_image = np.zeros_like(obs['raw_rgb'][i].detach().cpu().numpy())
            resize_bev_image = cv2.resize(bev_image, (384, 384))
            fps_writers[i].append_data(np.concatenate([resize_raw_image, resize_bev_image], axis=1))

        for i in range(env.num_envs):
            if dones[i] == True:
                current_sample_idx = current_episode_idx[i]
                new_sample_idx = env.unwrapped._sample_idx[i].item()
                # Serialize reset with inference/MPC; discard pre-reset queued requests.
                with planning_rpc_lock:
                    output_epoch += 1
                    with input_lock:
                        input_obs = None
                    with output_lock:
                        output_action = None
                        output_plan = None
                    if reactive_bridge is not None:
                        reactive_bridge.reset(output_epoch, float(step_count * env.unwrapped.step_dt))
                    execution_history.reset_env(i)
                    navigator_reset(env_id=i, port=args.server_port, sample_idx=new_sample_idx)

                if strict_termination is not None:
                    success_flag = float(strict_termination.last_success[i])
                    final_ne = float(strict_termination.last_distance[i])
                    terminal_speed = float(strict_termination.last_speed[i])
                    terminal_yaw_rate = float(strict_termination.last_yaw_rate[i])
                else:
                    success_flag = (1 - float(infos["time_outs"][i].item()))
                    final_ne = float(goal_dist_before_step[i])
                    terminal_speed = float('nan')
                    terminal_yaw_rate = float('nan')
                if fps_writers[i] is not None:
                    fps_writers[i].close()
                    fps_writers[i] = None

                if current_sample_idx is not None and current_sample_idx not in done_sample_indices:
                    evaluation_metrics.append({
                        'success': success_flag,
                        'spl': (
                            euclidean[i] / max(trajectory_length[i], euclidean[i], 1e-8)
                        ) * success_flag,
                        'distance': euclidean[i],
                        'ne': final_ne,
                        'collision': float(collided[i]),
                        'terminal_speed': terminal_speed,
                        'terminal_yaw_rate': terminal_yaw_rate,
                        'episode_idx': current_sample_idx
                    })
                    done_sample_indices.add(current_sample_idx)
                    write_metrics(evaluation_metrics, save_dir + "/metric.csv")
                elif current_sample_idx is not None:
                    print(f"Warning: sample_idx {current_sample_idx} already done, skipping metrics")

                if len(done_sample_indices) >= total_episode_count:
                    print(f"All {total_episode_count} episodes completed!")
                    break

                if new_sample_idx not in done_sample_indices:
                    fps_writers[i] = make_writer(new_sample_idx)
                    current_episode_idx[i] = new_sample_idx
                else:
                    print(f"Warning: new sample_idx {new_sample_idx} already done, skipping env {i}")
                    fps_writers[i] = None
                    current_episode_idx[i] = None

                euclidean[i] = np.sqrt(np.square(obs['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))[i]
                trajectory_length[i] = 0.0
                collided[i] = False

        if len(done_sample_indices) >= total_episode_count:
            print(f"All {total_episode_count} episodes completed!")
            break
        if args.max_steps is not None and step_count >= args.max_steps:
            print(f"Reached --max_steps={args.max_steps}; stopping evaluation cleanly")
            break

        camera_frames_now = env.unwrapped.scene.sensors['camera_sensor'].frame.cpu().numpy()
        # Do not assign a later capture pose/time to an unchanged image pair.
        fresh_planner_frame = np.all(camera_frames_now != planner_camera_frames)
        if not bridge_enabled or fresh_planner_frame:
            with input_lock:
                input_obs = parse_observations(obs, cfg.obs_mapping)
                input_obs["_epoch"] = output_epoch
                if bridge_enabled:
                    input_obs["_capture"] = dict(capture_time=float(step_count * env.unwrapped.step_dt),
                        positions=obs["robot_pose"].cpu().numpy().copy(),
                        quaternions=obs["robot_rot"].cpu().numpy().copy())
                input_obs["execution_feedback"] = {
                    "sim_time": float(step_count * env.unwrapped.step_dt),
                    "executed_segments": execution_history.snapshot(),
                }
            planner_camera_frames = camera_frames_now.copy()

    if reactive_bridge is not None:
        reactive_bridge.close()
    control_trace.close()
    print(f"Control trace: {control_trace.csv_path}", flush=True)
    print(f"Control plot:  {control_trace.figure_path}", flush=True)
    stop_event.set()
    planning_thread_obj.join(timeout=10.0)
    if not args.keep_server:
        navigator_shutdown(port=args.server_port)
    for w in fps_writers:
        if w is not None:
            try:
                w.close()
            except (RuntimeError, ValueError, OSError):
                pass
    try:
        env.close()
    except (RuntimeError, AttributeError, OSError):
        pass
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
