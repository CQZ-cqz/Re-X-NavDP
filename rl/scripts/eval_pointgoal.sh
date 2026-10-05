#!/usr/bin/env bash
# Unified pointgoal evaluation: switch between the MPC baseline and the RL direct tracker.
#
# The repository has two independent execution modes for the same pointgoal task:
#   * baseline — the original X-NavDP planner + MPC teacher (x-navdp/eval).
#   * direct   — the distilled RL direct tracker that outputs (v, omega) without MPC.
#
# Usage:
#   EVAL_MODE=baseline bash rl/scripts/eval_pointgoal.sh --config_file ... --scene_index 0
#   EVAL_MODE=direct   bash rl/scripts/eval_pointgoal.sh            # env-var driven (see run_direct_eval.sh)
#
# Both write an identical metric.csv, so the two runs can be compared directly
# (see rl/scripts/compare_eval.sh).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${EVAL_MODE:-baseline}"

case "$MODE" in
  baseline)
    exec bash "$DIR/../../x-navdp/eval/scripts/run_evaluation.sh" "$@"
    ;;
  direct)
    exec bash "$DIR/run_direct_eval.sh" "$@"
    ;;
  *)
    echo "usage: EVAL_MODE=baseline|direct bash eval_pointgoal.sh [args...]" >&2
    exit 2
    ;;
esac
