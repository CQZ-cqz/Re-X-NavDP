#!/usr/bin/env bash
# Stage-1 curriculum: lightweight clutter-easy PPO initialized from the
# well-fitted home MPC behavior-cloning policy. Defaults yield 262,144 control
# transitions: 8 train scenes * 2 epochs * 8 updates * 256 steps * 8 environments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../baselines/x-navdp" && pwd)"
cd "$REPO_ROOT"

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  INITIAL=(--resume-checkpoint "$RESUME_CHECKPOINT")
else
  INITIAL=(--bc-init "${BC_INIT:-outputs/direct_bc.pt}")
fi

exec bash ../../rl/scripts/launch_direct_full_train.sh \
  --scene-config eval/config/eval_pointgoal/humanoid_clutter_easy.yaml \
  --checkpoint ../checkpoints/x-navdp_posttrain.ckpt \
  "${INITIAL[@]}" \
  --epochs "${EPOCHS:-2}" \
  --iterations-per-scene "${ITERATIONS_PER_SCENE:-8}" \
  --rollout-steps "${ROLLOUT_STEPS:-256}" \
  --num-envs "${NUM_ENVS:-8}" \
  --success-distance "${SUCCESS_DISTANCE:-0.30}" \
  --output-root "${OUTPUT_ROOT:-outputs/direct_rgbd_clutter_easy_full}" \
  "$@"
