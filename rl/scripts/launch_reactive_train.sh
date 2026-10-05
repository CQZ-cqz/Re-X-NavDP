#!/usr/bin/env bash
# Launch vectorized reactive residual training (G1) on GPU 1.
#
# Usage:
#   bash ../rl/scripts/launch_reactive_train.sh [train_reactive.py args...]
#   bash ../rl/scripts/launch_reactive_train.sh --num-envs 8 --iterations 20
#
# The shell wrapper sets LD_LIBRARY_PATH for acados (libhpipm.so) — this cannot be
# done from inside the Python process because the dynamic linker caches
# LD_LIBRARY_PATH at startup. train_reactive.py still applies CUDA/Isaac/DISPLAY
# defaults internally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../x-navdp" && pwd)"
cd "$REPO_ROOT"

export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-${HOME}/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" ../rl/cli.py train-reactive "$@"
