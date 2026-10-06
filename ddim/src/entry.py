"""Shared helpers for the ddim CLI (observation loading, model build, metrics, plots)."""

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ddim.src.ddim_ddpm_metrics import compare_sets
from ddim.src.diffusion_sampling import sampling_timesteps
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""Offline DDPM vs DDIM comparison on identical preprocessed observations.

For each saved observation (fixed RGB history, depth, goal, embodiment) this
script draws the same initial noise and generates 8 candidate trajectories per
sampler, with RTC disabled, repeated over several random seeds. It then answers
the question "does DDIM preserve DDPM's trajectory distribution, and does the
frozen critic still rank its candidates as highly?" without needing Isaac Sim.

Comparison groups (reference is always the DDPM-10 baseline):
  * ``ddpm_self``    DDPM-10 vs DDPM-10 (independent noise): the policy's own
                     run-to-run variance, so not all difference is blamed on DDIM.
  * ``ddim10_vs_ddpm`` DDIM-10 vs DDPM-10 (same initial noise): sampler swap.
  * ``ddim5_vs_ddpm``  DDIM-5  vs DDPM-10 (same initial noise): sampler + fewer steps.

Metrics (per observation, aggregated over seeds):
  * matched ADE/FDE      -- Hungarian min-cost pairing between the 8-candidate sets.
  * preferred ADE/FDE    -- distance between the two Q-argmax trajectories.
  * Q stats + diffs      -- frozen critic: mean/max/min, max-diff, win rate.
  * diversity            -- mean pairwise ADE and endpoint dispersion per set.

Observation NPZ format (written by ``policy_server --save-observations``):
  rgb [8,224,224,3], depth [224,224,1], pointgoal [3], embodiment (int),
  plus optional robot_pos/robot_quat/scene/sample_idx.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


from ddim.src.ddim_ddpm_metrics import compare_sets  # noqa: E402
from ddim.src.diffusion_sampling import sampling_timesteps  # noqa: E402
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe pair: blue (DDPM) vs orange (DDIM).
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#8C8C8C"

# Base configs; extra DDIM grids may be appended via --ddim-grids NAME=9,6,4,2,0.
# Each spec is (name, sampler, steps_or_grid, noise_ref) with noise_ref in "a"/"b".
BASE_SPECS = (
    ("ddpm_a", "ddpm", 10, "a"),
    ("ddpm_b", "ddpm", 10, "b"),
    ("ddim10", "ddim", 10, "a"),
    ("ddim5", "ddim", 5, "a"),
)


def resolve_grid(sampler, steps_or_grid):
    """Return a descending long tensor of sampling timesteps.

    ``steps_or_grid`` is either an int step count (uniform DDIM/DDPM grid via
    ``sampling_timesteps``) or an explicit descending list of timesteps in 0..9.
    """
    if isinstance(steps_or_grid, (list, tuple)):
        grid = [int(t) for t in steps_or_grid]
        if not (2 <= len(grid) <= 10) or any(not (0 <= t <= 9) for t in grid):
            raise ValueError(f"explicit grid {grid} must be 2..10 timesteps in 0..9")
        if any(a <= b for a, b in zip(grid, grid[1:])):
            raise ValueError(f"explicit grid {grid} must be strictly descending")
        return torch.tensor(grid, dtype=torch.long)
    return sampling_timesteps(sampler, int(steps_or_grid))


def run_compare_config(model, sampler, steps_or_grid, initial_noise, goal, rgb, depth,
               embodiment, seed, device):
    """Run one sampler config over the whole observation batch."""
    model.sampler = sampler
    model.sampling_timesteps = resolve_grid(sampler, steps_or_grid)
    torch.manual_seed(seed)
    n = goal.shape[0]
    valid = np.zeros(n, dtype=np.int32)
    prev = np.zeros((n, 24, 3), dtype=np.float32)
    guidance = np.array([0.5] * 6 + [0.05] * 2, dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    paths, scores, _, _ = model.predict_pointgoal_action_with_guidance(
        goal, rgb, depth, 8, valid, prev, 0, 8, guidance,
        guidance_step=5, embodiment=embodiment, initial_noise=initial_noise)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not (np.isfinite(paths).all() and np.isfinite(scores).all()):
        raise RuntimeError(f"Non-finite output for {sampler}/{steps_or_grid}")
    return paths, scores, elapsed


def plot_overlay(record, paths_seed0, q_seed0, out_path):
    """Plot DDPM (blue) vs DDIM (orange) candidate sets for one observation."""
    rgb = record["rgb"]
    ddpm_a = paths_seed0["ddpm_a"]  # [8,24,3]
    ddpm_b = paths_seed0["ddpm_b"]
    q_a = q_seed0["ddpm_a"]
    title = f"{record['scene']}  sample {record['sample_idx']}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110)
    fig.suptitle(title, fontsize=11)
    for ax, (name, cand, q_cand) in zip(
            axes, (("DDIM-10", paths_seed0["ddim10"], q_seed0["ddim10"]),
                   ("DDIM-5", paths_seed0["ddim5"], q_seed0["ddim5"]))):
        draw_set(ax, ddpm_b, None, GREY, "DDPM-10 (self)", linestyle="--", alpha=0.25)
        draw_set(ax, ddpm_a, q_a, BLUE, "DDPM-10")
        draw_set(ax, cand, q_cand, ORANGE, name)
        ax.set_title(name + " vs DDPM-10", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def draw_set(ax, paths, q, color, label, linestyle="-", alpha=0.6):
    """Plot one candidate set; thick line for the Q-argmax, tiny Q labels."""
    paths = paths[..., :2]
    if q is None:
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], color=color, linestyle=linestyle,
                    linewidth=1.0, alpha=alpha)
        return
    argmax = int(np.argmax(q))
    for i, p in enumerate(paths):
        if i == argmax:
            ax.plot(p[:, 0], p[:, 1], color=color, linestyle=linestyle,
                    linewidth=2.5, alpha=1.0, label=label)
            ax.plot(p[0, 0], p[0, 1], "o", color=color, markersize=3)
            ax.text(p[-1, 0], p[-1, 1], f"{q[i]:.2f}", color=color,
                    fontsize=7, va="bottom", ha="left")
        else:
            ax.plot(p[:, 0], p[:, 1], color=color, linestyle=linestyle,
                    linewidth=1.0, alpha=alpha)
            ax.text(p[-1, 0], p[-1, 1], f"{q[i]:.2f}", color=color,
                    fontsize=5, va="bottom", ha="left", alpha=0.7)

"""Sweep samplers, step counts, and RTC guidance on identical observations.

Runs the full offline matrix in one shot and writes a single JSON report plus a
console summary, so the fidelity/speed/guidance tradeoffs can be read off
directly. No Isaac Sim needed.

Matrix
------
* samplers: DDPM (10-step only, the baseline) and DDIM at 2/3/4/5/6/8/10 steps.
* RTC: off (no guidance) and on (receding-horizon correction toward the real
  previously-executed trajectory captured with each observation).

Groups (candidate vs reference):
* ``ddpm_self``             DDPM-10 vs DDPM-10 independent noise: baseline variance.
* ``ddim{k}_vs_ddpm``       DDIM-k vs DDPM-10, RTC off: sampler/steps fidelity.
* ``ddim{k}_rtc_vs_ddpm_rtc`` DDIM-k vs DDPM-10, RTC on.
* ``{name}_rtc_effect``     RTC on vs off for the same sampler/steps.

Observation NPZ keys (from ``policy_server --save-observations``):
  rgb [8,224,224,3], depth [224,224,1], pointgoal [3], embodiment (int),
  prev_action [24,3], valid_segment_len (int), plus scene/sample_idx.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


from ddim.src.ddim_ddpm_metrics import compare_sets  # noqa: E402
from ddim.src.diffusion_sampling import sampling_timesteps  # noqa: E402
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment  # noqa: E402

# (name, sampler, steps, rtc, noise_ref)
CONFIGS = [
    ("ddpm10",     "ddpm", 10, False, "a"),
    ("ddpm10_b",   "ddpm", 10, False, "b"),
    ("ddim10",     "ddim", 10, False, "a"),
    ("ddim8",      "ddim", 8,  False, "a"),
    ("ddim6",      "ddim", 6,  False, "a"),
    ("ddim5",      "ddim", 5,  False, "a"),
    ("ddim4",      "ddim", 4,  False, "a"),
    ("ddim3",      "ddim", 3,  False, "a"),
    ("ddim2",      "ddim", 2,  False, "a"),
    ("ddpm10_rtc", "ddpm", 10, True,  "a"),
    ("ddim10_rtc", "ddim", 10, True,  "a"),
    ("ddim6_rtc",  "ddim", 6,  True,  "a"),
    ("ddim5_rtc",  "ddim", 5,  True,  "a"),
    ("ddim4_rtc",  "ddim", 4,  True,  "a"),
]

# (group_name, candidate_name, reference_name)
GROUPS = [
    ("ddpm_self", "ddpm10_b", "ddpm10"),
    ("ddim10_vs_ddpm", "ddim10", "ddpm10"),
    ("ddim8_vs_ddpm",  "ddim8",  "ddpm10"),
    ("ddim6_vs_ddpm",  "ddim6",  "ddpm10"),
    ("ddim5_vs_ddpm",  "ddim5",  "ddpm10"),
    ("ddim4_vs_ddpm",  "ddim4",  "ddpm10"),
    ("ddim3_vs_ddpm",  "ddim3",  "ddpm10"),
    ("ddim2_vs_ddpm",  "ddim2",  "ddpm10"),
    ("ddim10_rtc_vs_ddpm_rtc", "ddim10_rtc", "ddpm10_rtc"),
    ("ddim6_rtc_vs_ddpm_rtc",  "ddim6_rtc",  "ddpm10_rtc"),
    ("ddim5_rtc_vs_ddpm_rtc",  "ddim5_rtc",  "ddpm10_rtc"),
    ("ddim4_rtc_vs_ddpm_rtc",  "ddim4_rtc",  "ddpm10_rtc"),
    ("ddpm10_rtc_effect", "ddpm10_rtc", "ddpm10"),
    ("ddim10_rtc_effect", "ddim10_rtc", "ddim10"),
    ("ddim6_rtc_effect",  "ddim6_rtc",  "ddim6"),
    ("ddim5_rtc_effect",  "ddim5_rtc",  "ddim5"),
    ("ddim4_rtc_effect",  "ddim4_rtc",  "ddim4"),
]


def load_observations(args):
    paths = []
    for item in args.observations:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.npz"))))
        else:
            paths.append(item)
    if not paths:
        raise SystemExit(f"No NPZ observations under {args.observations}")
    records = []
    for p in paths:
        with np.load(p, allow_pickle=True) as data:
            records.append({
                "rgb": data["rgb"].astype(np.float32),
                "depth": data["depth"].astype(np.float32),
                "pointgoal": data["pointgoal"].astype(np.float32),
                "embodiment": int(data["embodiment"]),
                "prev_action": data["prev_action"].astype(np.float32)
                    if "prev_action" in data else None,
                "valid_segment_len": int(data["valid_segment_len"])
                    if "valid_segment_len" in data else 0,
                "sample_idx": int(data["sample_idx"]) if "sample_idx" in data else -1,
                "scene": str(data["scene"]) if "scene" in data else "",
                "path": str(p),
            })
    if args.max_observations and len(records) > args.max_observations:
        idx = np.linspace(0, len(records) - 1, args.max_observations).astype(int)
        records = [records[i] for i in idx]
    return records


def build_model(checkpoint, device):
    model = NavDP_Policy_Embodiment(temporal_depth=16, device=str(device)).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing checkpoint weights: {incompatible.missing_keys}")
    del state
    model.eval()
    model.rtc_enabled = False
    return model


def run_sweep_config(model, sampler, steps, rtc, initial_noise, goal, rgb, depth,
               prev_action, valid_segment_len, embodiment, seed, device):
    model.sampler = sampler
    model.sampling_timesteps = sampling_timesteps(sampler, steps)
    model.rtc_enabled = rtc
    torch.manual_seed(seed)
    n = goal.shape[0]
    guidance = np.array([0.5] * 6 + [0.05] * 2, dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    paths, scores, _, _ = model.predict_pointgoal_action_with_guidance(
        goal, rgb, depth, 8, valid_segment_len.copy(), prev_action.copy(),
        0, 8, guidance, guidance_step=5, embodiment=embodiment,
        initial_noise=initial_noise)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not (np.isfinite(paths).all() and np.isfinite(scores).all()):
        raise RuntimeError(f"Non-finite output for {sampler}/{steps} rtc={rtc}")
    return paths, scores, elapsed


def mean_of_dicts(dicts):
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def aggregate_across_obs(per_obs_means):
    keys = per_obs_means[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_obs_means], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out

