#!/usr/bin/env python3
"""Aggregate X-NavDP evaluation metrics into a SR / SPL / NE / CR report.

Input is one evaluation group directory, for example:

    outputs/evaluation/humanoid_home

The script recursively finds ``metric.csv`` files. Each metric.csv is expected to
contain the columns written by ``eval.scripts.evaluate_pointgoal``::

    success, spl, distance, ne, collision, episode_idx

For backwards compatibility, missing ``ne``/``collision`` columns are treated as
0 (so older runs can still be summarised, but NE/CR will be 0).

Definitions (computed per episode, averaged over all episodes):
  * SR  = success rate              = mean(success)
  * SPL = success weighted by path  = mean(spl)
  * NE  = navigation error          = mean(ne)     (final distance-to-goal, metres)
  * CR  = collision rate            = mean(collision)

Usage:
    python eval/scripts/report_eval_metrics.py outputs/evaluation/humanoid_home [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def usd_name(metric_path: Path) -> str:
    """``metric.csv`` parent directory is usually ``<scene>_usd``."""
    scene_dir = metric_path.parent.name
    if scene_dir.endswith("_usd"):
        return f"{scene_dir[:-4]}.usd"
    if scene_dir.endswith(".usd"):
        return scene_dir
    return scene_dir


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for key in ("success", "spl", "ne", "collision", "episode_idx"):
                try:
                    row[key] = float(raw.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    row[key] = 0.0
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"episodes": 0, "success": 0, "sr": 0.0, "spl": 0.0, "ne": 0.0, "cr": 0.0}
    sr = sum(r["success"] for r in rows) / n
    spl = sum(r["spl"] for r in rows) / n
    ne = sum(r["ne"] for r in rows) / n
    cr = sum(r["collision"] for r in rows) / n
    success_count = sum(1 for r in rows if r["success"] >= 0.5)
    return {
        "episodes": n,
        "success": success_count,
        "sr": sr,
        "spl": spl,
        "ne": ne,
        "cr": cr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Evaluation group directory, e.g. outputs/evaluation/humanoid_home")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-scene summary to this CSV")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"Input directory does not exist: {root}")
        return 2

    buckets: dict[str, list[dict]] = defaultdict(list)
    for metric_path in sorted(root.rglob("metric.csv")):
        buckets[usd_name(metric_path)].extend(read_rows(metric_path))

    if not buckets:
        print(f"No metric.csv files found under: {root}")
        return 2

    print(f"Input: {root}")
    print()
    print(f"{'scene':<40} {'ep':>5} {'SR':>8} {'SPL':>8} {'NE(m)':>8} {'CR':>8}")

    all_rows: list[dict] = []
    per_scene: list[tuple[str, dict]] = []
    for usd in sorted(buckets):
        s = summarize(buckets[usd])
        all_rows.extend(buckets[usd])
        per_scene.append((usd, s))
        print(
            f"{usd:<40} {s['episodes']:>5} {s['sr']:>7.2%} {s['spl']:>8.4f} "
            f"{s['ne']:>8.3f} {s['cr']:>7.2%}"
        )

    total = summarize(all_rows)
    print("-" * 82)
    print(
        f"{'OVERALL':<40} {total['episodes']:>5} {total['sr']:>7.2%} {total['spl']:>8.4f} "
        f"{total['ne']:>8.3f} {total['cr']:>7.2%}"
    )
    print()
    print(
        f"Overall: SR={total['sr']:.2%}  SPL={total['spl']:.4f}  "
        f"NE={total['ne']:.3f}m  CR={total['cr']:.2%}  "
        f"episodes={total['episodes']}  success={total['success']}"
    )

    if args.csv is not None:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scene", "episodes", "success", "sr", "spl", "ne", "cr"])
            for usd, s in per_scene:
                writer.writerow([usd, s["episodes"], s["success"], s["sr"], s["spl"], s["ne"], s["cr"]])
            writer.writerow(["OVERALL", total["episodes"], total["success"],
                             total["sr"], total["spl"], total["ne"], total["cr"]])
        print(f"\nWrote per-scene summary to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
