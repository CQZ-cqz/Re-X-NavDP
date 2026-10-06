#!/usr/bin/env bash
# Capture preprocessed observations for the offline DDIM/DDPM comparison.
#
# Runs ONE episode of X-NavDP (post-trained planner) + Acados MPC on a clutter
# scene, with the policy server dumping --save-observations NPZ files at every
# decision point. Those NPZs are then fed to ddim.cli compare (no Isaac).
#
# Tunables (env vars):
#   OBS_DIR       where NPZ observations are written (default below)
#   SCENE_CONFIG  eval config (default: humanoid_clutter_easy.yaml)
#   SCENE_INDEX   held-out scene index (default 0 -> easy_6 for clutter easy)
#   CHECKPOINT    planner checkpoint (default: post-trained x-navdp_posttrain.ckpt)
#   NUM_EPISODES  episode cap (default 1)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../baselines/x-navdp" && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export ISAAC_ACTIVE_GPU="${ISAAC_ACTIVE_GPU:-1}"
export ISAAC_PHYSICS_GPU="${ISAAC_PHYSICS_GPU:-0}"
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-${HOME}/acados}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
unset DISPLAY

PY=${PY:-python}
PORT=${PORT:-20005}
CHECKPOINT=${CHECKPOINT:-../checkpoints/x-navdp_posttrain.ckpt}
SCENE_CONFIG=${SCENE_CONFIG:-eval/config/eval_pointgoal/humanoid_clutter_easy.yaml}
SCENE_INDEX=${SCENE_INDEX:-0}
NUM_EPISODES=${NUM_EPISODES:-1}
NUM_ENVS=${NUM_ENVS:-1}
RECORD_NUM=${RECORD_NUM:-1}
OBS_DIR=${OBS_DIR:-outputs/ddim_ddpm_observations/cluttered_easy}

[[ -f "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT"; exit 1; }
mkdir -p "$OBS_DIR"

echo "Capturing observations -> $OBS_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Scene: $SCENE_CONFIG idx=$SCENE_INDEX   Episodes: $NUM_EPISODES"

SERVER_LOG=/tmp/policy_server_capture.log
nohup "$PY" -m eval.src.policy_server --port "$PORT" --embodiment humanoid \
    --checkpoint "$CHECKPOINT" --device cuda:0 --no-visualization \
    --save-observations "$OBS_DIR" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
    if grep -q "Running on" "$SERVER_LOG" 2>/dev/null; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "policy server exited during startup"; tail -40 "$SERVER_LOG"; exit 1
    fi
    sleep 2
done
grep -q "Running on" "$SERVER_LOG" 2>/dev/null || {
    echo "policy server did not become ready"; tail -40 "$SERVER_LOG"; exit 1; }
echo "policy server ready (PID $SERVER_PID)"

"$PY" -m eval.scripts.evaluate_pointgoal \
    --config_file "$SCENE_CONFIG" --scene_index "$SCENE_INDEX" --device cuda:0 \
    --num_episodes "$NUM_EPISODES" --num_envs "$NUM_ENVS" \
    --record_num "$RECORD_NUM" --server_port "$PORT" --strict_pointgoal --keep_server

echo "capture done. observations: $(find "$OBS_DIR" -name '*.npz' | wc -l)"
