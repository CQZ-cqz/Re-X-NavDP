"""Load one scene in headless Isaac and run a few reset/step cycles.

This validates that a scene's USD (and its referenced models/materials),
navigation ESDF, and point-goal data all load correctly through the same
`create_dingonav_environment` path used by distributed training. Useful for
spot-checking the eval InternScenes home/commercial scenes before a long run.

Example (must run with the same GPU-1 env vars as training):
    SCENE_DIR=/mnt/data3/cqz/nav/scenes \
    CUDA_VISIBLE_DEVICES=1 ISAAC_ACTIVE_GPU=1 ISAAC_PHYSICS_GPU=0 \
    python ../tools/diagnostics/validate_scene.py --scene-type commercial --num-steps 5
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
    parser = argparse.ArgumentParser(description="Validate one scene in headless Isaac.")
    parser.add_argument("--scene_dir", type=str, default=os.environ.get("SCENE_DIR", "/mnt/data3/cqz/nav/scenes"))
    parser.add_argument("--scene-type", type=str, default=None, choices=("cluttered_hard", "cluttered_easy", "commercial", "home"))
    parser.add_argument("--split", type=str, default="eval", choices=("train", "eval", "all"))
    parser.add_argument("--scene-index", type=int, default=None, help="Explicit scene order index; overrides --scene-type.")
    parser.add_argument("--embodiment", type=str, default="dingo", choices=("dingo", "unitree_g1", "unitree_go2"))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--step-timeout", type=int, default=120, help="Dump traceback + exit if a step hangs longer than this (seconds).")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    # Isaac Sim imports must happen after the app launcher wraps SimulationApp.
    from isaaclab.app import AppLauncher

    from src.training.scene_assets import load_and_mix_scene_data
    from src.training.worker import configure_mdl_system_path, disable_app_control_on_stop

    configure_mdl_system_path(args.scene_dir)

    app_launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
    simulation_app = app_launcher.app
    try:
        from src.environment import create_dingonav_environment

        scenes = load_and_mix_scene_data(
            args.scene_dir,
            scene_split_file=os.path.join(args.scene_dir, "scene_split.json"),
            split=args.split,
            usd_variant="navigation",
        )
        if args.scene_index is not None:
            index = args.scene_index
        else:
            match = [s for s in scenes if s["scene_type"] == args.scene_type]
            if not match:
                raise SystemExit(f"no scene of type {args.scene_type!r} in the loaded list")
            index = scenes.index(match[0])

        scene = scenes[index]
        print(f"[validate] scene_index={index} scene_type={scene['scene_type']} "
              f"scene_name={scene['scene_name']} embodiment={args.embodiment}", flush=True)

        env, controller = create_dingonav_environment(
            scenes, index, num_envs=args.num_envs, device=args.device, embodiment=args.embodiment
        )
        print("[validate] env created; resetting", flush=True)
        obs, _ = env.reset()
        print(f"[validate] reset ok; obs keys={sorted(obs.keys())}", flush=True)
        # If a step hangs (e.g. physics deadlock on a specific scene), dump the
        # stack so we can see exactly where instead of guessing.
        faulthandler.dump_traceback_later(args.step_timeout, exit=True)
        try:
            for step in range(args.num_steps):
                commands = np.zeros((args.num_envs, 2), dtype=np.float32)
                control = controller.forward_batch(None, commands)
                print(f"[validate] step {step} START", flush=True)
                obs, rew, done, info = env.step(control)
                # rewards/dones are CUDA tensors; move to CPU before numpy conversion.
                rew_np = np.asarray(rew.detach().cpu()).round(3)
                done_np = np.asarray(done.detach().cpu())
                print(f"[validate] step {step} DONE: rewards={rew_np} done={done_np}", flush=True)
            env.close()
            print(f"[validate] OK scene={scene['scene_name']}", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("[validate] STEP FAILED (see traceback above)", flush=True)
            raise
    finally:
        disable_app_control_on_stop(locals().get("env"))
        simulation_app.close()


if __name__ == "__main__":
    main()
