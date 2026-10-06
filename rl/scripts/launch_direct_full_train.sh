#!/usr/bin/env bash
# Run resumable direct RGB-D PPO across the complete home train split.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../baselines/x-navdp" && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export ISAAC_ACTIVE_GPU="${ISAAC_ACTIVE_GPU:-1}"
export ISAAC_PHYSICS_GPU="${ISAAC_PHYSICS_GPU:-0}"
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-${HOME}/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
unset DISPLAY

PYTHON_BIN="${PYTHON_BIN:-python}"
# Keep the nohup/background PID identical to the scheduler PID. The scheduler
# forwards TERM/INT to its active per-scene process group.
exec "${PYTHON_BIN}" -u ../../rl/cli.py train-full "$@"
