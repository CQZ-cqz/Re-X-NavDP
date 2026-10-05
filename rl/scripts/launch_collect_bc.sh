#!/usr/bin/env bash
# Collect MPC-teacher BC data for the direct tracker (plan P4).
#
# The teacher IS the MPC solver, so this wrapper sets LD_LIBRARY_PATH for acados
# (libhpipm.so) before launching collect_direct_bc.py — this cannot be done from
# inside the Python process (the dynamic linker caches LD_LIBRARY_PATH at startup).
#
# Usage:
#   bash ../rl/scripts/launch_collect_bc.sh [collect_direct_bc.py args...]
#   bash ../rl/scripts/launch_collect_bc.sh --steps 20000 --scene-index 0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../x-navdp" && pwd)"
cd "$REPO_ROOT"

export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-${HOME}/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" ../rl/cli.py collect "$@"
