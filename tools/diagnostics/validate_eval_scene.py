"""Isolate the home/commercial eval render deadlock.

Replicates `eval/environment.create_environment` for a single home/commercial
scene, but exposes the two config differences vs. the (working) training env
so we can bisect which one triggers the Hydra render deadlock:

  * `--robot-invisible`  -> robot_visible=False (training hides the robot)
  * `--no-birdeye`       -> drop the bird-eye camera + observation

Run with the same GPU-1 env vars as training/eval:
    CUDA_VISIBLE_DEVICES=1 ISAAC_ACTIVE_GPU=1 ISAAC_PHYSICS_GPU=0 \
    python ../tools/diagnostics/validate_eval_scene.py --scene-type home --robot-invisible --no-birdeye
"""
import argparse
import faulthandler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Bisect the home/commercial eval render deadlock.")
    parser.add_argument("--config_file", type=str,
                        default="eval/config/eval_pointgoal/wheeled_internscene_home.yaml")
    parser.add_argument("--scene-type", type=str, default=None, choices=("home", "commercial"))
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--embodiment", type=str, default="dingo")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--step-timeout", type=int, default=120)
    parser.add_argument("--robot-invisible", action="store_true",
                        help="Hide the robot from the camera (robot_visible=False, like training).")
    parser.add_argument("--no-birdeye", action="store_true",
                        help="Drop the bird-eye camera and observation.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher
    from src.training.worker import (
        configure_mdl_system_path,
        disable_app_control_on_stop,
        disable_async_rendering,
    )
    from eval.config_utils import load_default_config
    from eval.environment import _resolve_scene, namespace_to_dict

    cfg = load_default_config(args.config_file)
    scene_dir = cfg.environment.scene_dir
    if args.scene_type is not None:
        cfg.environment.scene_type = args.scene_type
    configure_mdl_system_path(scene_dir)

    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        device=args.device,
    )
    simulation_app = app_launcher.app
    disable_async_rendering()
    print("[validate_eval] AppLauncher ready; arming hang watchdog", flush=True)
    faulthandler.dump_traceback_later(args.step_timeout, exit=True)
    env = None
    try:
        from src.environment import create_dingoeval_environment

        scene_index = args.scene_index if args.scene_index is not None else cfg.environment.scene_index
        scene_data, scene_name = _resolve_scene(cfg, scene_index)
        controller_config = namespace_to_dict(cfg.controller) if hasattr(cfg, "controller") else None
        print(f"[validate_eval] scene_index={scene_index} scene_name={scene_name} "
              f"scene_type={scene_data.get('scene_type')}", flush=True)
        print(f"[validate_eval] robot_visible={not args.robot_invisible} "
              f"enable_birdeye={not args.no_birdeye}", flush=True)

        print("[validate_eval] creating env ...", flush=True)
        env, controller = create_dingoeval_environment(
            scene_dir=scene_dir,
            scene_index=0,
            num_envs=args.num_envs,
            scene_scale=getattr(cfg.environment, "scene_scale", None),
            device=args.device,
            embodiment=args.embodiment,
            scene_data=scene_data,
            controller_config=controller_config,
        )
        print("[validate_eval] env created; resetting ...", flush=True)
        obs, _ = env.reset()
        print(f"[validate_eval] reset ok; obs keys={sorted(obs.keys())}", flush=True)

        try:
            import torch
            for step in range(args.num_steps):
                print(f"[validate_eval] step {step} START", flush=True)
                commands = torch.zeros((args.num_envs, 2), dtype=torch.float32, device=args.device)
                obs, rew, done, info = env.step(commands)
                print(f"[validate_eval] step {step} DONE", flush=True)
            env.close()
            print(f"[validate_eval] OK scene={scene_name}", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("[validate_eval] STEP FAILED (see traceback above)", flush=True)
            raise
    finally:
        disable_app_control_on_stop(env)
        simulation_app.close()


if __name__ == "__main__":
    main()
