#!/usr/bin/env bash
# Compare SR / SPL / NE / CR between two pointgoal evaluation output directories.
#
# Typical use: compare the MPC baseline against the RL direct tracker. Both eval
# runs must cover the same scene split to be comparable (see eval_pointgoal.sh).
#
# Usage:
#   bash rl/scripts/compare_eval.sh outputs/evaluation/humanoid_clutter_baseline \
#                                     outputs/evaluation/humanoid_clutter_direct
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT="$DIR/../../baselines/x-navdp/eval/scripts/report_eval_metrics.py"

if [[ $# -lt 2 ]]; then
  echo "usage: bash compare_eval.sh <baseline_eval_dir> <direct_eval_dir>" >&2
  exit 2
fi

echo "=== baseline: $1 ==="
python "$REPORT" "$1"
echo
echo "=== direct: $2 ==="
python "$REPORT" "$2"
