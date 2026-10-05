#!/usr/bin/env bash
# Direct-policy (ReX-NavDP, MPC-free) evaluation on clutter scenes with video.
#
# Two-stage: the X-NavDP policy server produces trajectories (pretrained
# checkpoint for clutter), and the eval client runs Isaac + the direct tracker.
# Direct mode never instantiates an Acados solver, so no acados lib is needed.
#
# Tunables (env vars):
#   DIRECT_CHECKPOINT  direct policy checkpoint (default: ../checkpoints/rl_direct_tracker.pt,
#                      then newest reward-v* run, then the clutter epoch-1 checkpoint)
#   SCENE_CONFIG       scene config (default: clutter_easy)
#   SCENE_INDEX        scene index (default: 0)
#   NUM_EPISODES       episode cap (default: 10)
#   NUM_ENVS           parallel envs (default: 1)
#   RECORD_NUM         max episodes to record video (default: 3, random subset)
#   POLICY_VISUALIZATION  planner debug panels: 1=on, 0=off (default: 1)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../x-navdp" && pwd)"
cd "$REPO_ROOT"

# GPU-1 workaround (memory: GPU 0 is usually occupied).
export CUDA_VISIBLE_DEVICES=1
export ISAAC_ACTIVE_GPU=1
export ISAAC_PHYSICS_GPU=0
export OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY

# The policy-server subprocess imports bridge/ddim from the repo root (x-navdp/..).
export PYTHONPATH="$(cd "$REPO_ROOT/.." && pwd):${PYTHONPATH:-}"

PY="${PY:-python}"
PORT=19999
PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT:-../checkpoints/x-navdp_posttrain.ckpt}
ENCODER_CHECKPOINT=${ENCODER_CHECKPOINT:-../checkpoints/x-navdp_posttrain.ckpt}     # matches direct policy encoder fingerprint

DIRECT_CHECKPOINT=${DIRECT_CHECKPOINT:-}
if [[ -z "$DIRECT_CHECKPOINT" ]]; then
    if [[ -f ../checkpoints/rl_direct_tracker.pt ]]; then
        # Unified checkpoint index: the designated in-use direct tracker weight.
        DIRECT_CHECKPOINT=../checkpoints/rl_direct_tracker.pt
    else
        shopt -s nullglob
        reward_candidates=(outputs/direct_rgbd_clutter_easy_reward_v*_*/latest.pt)
        shopt -u nullglob
        for candidate in "${reward_candidates[@]}"; do
            if [[ -z "$DIRECT_CHECKPOINT" || "$candidate" -nt "$DIRECT_CHECKPOINT" ]]; then
                DIRECT_CHECKPOINT="$candidate"
            fi
        done
        if [[ -z "$DIRECT_CHECKPOINT" ]]; then
            DIRECT_CHECKPOINT=outputs/direct_rgbd_clutter_easy_epoch1.pt
        fi
    fi
fi
[[ -f "$DIRECT_CHECKPOINT" ]] || {
    echo "direct checkpoint not found: $DIRECT_CHECKPOINT"; exit 1; }
SCENE_CONFIG=${SCENE_CONFIG:-eval/config/eval_pointgoal/humanoid_clutter_easy.yaml}
SCENE_INDEX=${SCENE_INDEX:-0}
NUM_EPISODES=${NUM_EPISODES:-10}
NUM_ENVS=${NUM_ENVS:-1}
RECORD_NUM=${RECORD_NUM:-3}
POLICY_VISUALIZATION=${POLICY_VISUALIZATION:-1}

SERVER_VIS_ARGS=()
if [[ "$POLICY_VISUALIZATION" == "0" ]]; then
    SERVER_VIS_ARGS+=(--no-visualization)
fi

echo "Evaluation only (no PPO updates)"
echo "Planner checkpoint: $PLANNER_CHECKPOINT"
echo "Direct checkpoint:  $DIRECT_CHECKPOINT"
echo "Held-out scene index: $SCENE_INDEX"

# --- start policy server (planner) ---
nohup "$PY" -m eval.src.policy_server --port "$PORT" --embodiment humanoid \
    --checkpoint "$PLANNER_CHECKPOINT" --device cuda:0 \
    "${SERVER_VIS_ARGS[@]}" \
    > /tmp/policy_server_direct.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do
    grep -q "Running on" /tmp/policy_server_direct.log 2>/dev/null && break
    sleep 2
done
grep -q "Running on" /tmp/policy_server_direct.log 2>/dev/null || {
    echo "policy server failed to start"; tail -20 /tmp/policy_server_direct.log; exit 1; }
echo "policy server ready (PID $SERVER_PID)"

# --- run direct eval (records metric.csv + reactive_control.csv + mp4 videos) ---
"$PY" -m eval.scripts.evaluate_pointgoal \
    --config_file "$SCENE_CONFIG" --scene_index "$SCENE_INDEX" --device cuda:0 \
    --direct_checkpoint "$DIRECT_CHECKPOINT" \
    --direct_config ../rl/config/reactive_rgbd_direct_g1.yaml \
    --reactive_encoder_checkpoint "$ENCODER_CHECKPOINT" \
    --reactive_device cuda:0 \
    --strict_pointgoal \
    --num_episodes "$NUM_EPISODES" --num_envs "$NUM_ENVS" --record_num "$RECORD_NUM" \
    --server_port "$PORT" --keep_server
