"""Re-X-NavDP DDIM / DDPM comparison — unified command line interface."""

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_BASE = _ROOT / "baselines/x-navdp"
for _p in (_ROOT, _BASE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ddim.src.diffusion_sampling import sampling_timesteps
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment
from ddim.src.entry import (BASE_SPECS, BLUE, CONFIGS, GREY, GROUPS, ORANGE,
                             aggregate_across_obs, build_model, load_observations,
                             mean_of_dicts, plot_overlay, resolve_grid,
                             run_compare_config, run_sweep_config)

def cmd_benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", help="Preprocessed NPZ; omit for synthetic smoke test")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--embodiment", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtc", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", help="Write benchmark metadata and measurements to JSON")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("runs must be positive and warmup nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; CPU smoke test: --device cpu --runs 1 --warmup 0")
    if args.observations:
        with np.load(args.observations) as data:
            rgb, depth, goal = (data[key].copy() for key in ("rgb", "depth", "pointgoal"))
            batch = len(goal)
            if args.rtc and not {"prev_action", "valid_segment_len"}.issubset(data.files):
                parser.error("RTC replay needs prev_action and valid_segment_len")
            previous = data["prev_action"].copy() if "prev_action" in data else np.zeros((batch, 24, 3))
            valid = data["valid_segment_len"].copy() if "valid_segment_len" in data else np.zeros(batch, dtype=int)
    else:
        batch = 1
        rgb = np.zeros((batch, 8, 224, 224, 3), dtype=np.float32)
        depth = np.ones((batch, 224, 224, 1), dtype=np.float32)
        goal = np.array([[2., 0., 0.]], dtype=np.float32)
        previous, valid = np.zeros((batch, 24, 3)), np.full(batch, 6, dtype=int)
    model = NavDP_Policy_Embodiment(temporal_depth=16, device=str(device)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing checkpoint weights: {incompatible.missing_keys}")
    del state
    model.eval()
    model.rtc_enabled = args.rtc
    metadata = {"input": args.observations or "synthetic_smoke_only", "device": str(device),
                      "rtc": args.rtc, "candidates": 8,
                      "checkpoint": args.checkpoint, "runs": args.runs, "warmup": args.warmup,
                      "seed": args.seed, "threads": args.threads,
                      "unexpected_checkpoint_key_count": len(incompatible.unexpected_keys)}
    print(json.dumps(metadata), flush=True)
    measurements = []
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    reference = None
    for sampler, steps in (("ddpm", 10), ("ddim", 10), ("ddim", 5), ("ddim", 4)):
        model.sampler = sampler
        model.sampling_timesteps = sampling_timesteps(sampler, steps)
        times, all_paths = [], []
        for run in range(-args.warmup, args.runs):
            torch.manual_seed(args.seed + max(run, 0))
            sync()
            start = time.perf_counter()
            paths, scores, best, _ = model.predict_pointgoal_action_with_guidance(
                goal, rgb, depth, 8, valid.copy(), previous, 0, 8,
                np.array([.5]*6 + [.05]*2), guidance_step=5, embodiment=args.embodiment)
            sync()
            elapsed = time.perf_counter() - start
            if not (np.isfinite(paths).all() and np.isfinite(scores).all()):
                raise RuntimeError(f"Nonfinite output: {sampler}/{steps}")
            if run >= 0:
                times.append(elapsed)
                all_paths.append(paths)
        all_paths = np.stack(all_paths)
        if reference is None:
            reference = all_paths
        measurement = {"sampler": sampler, "steps": steps,
                          "timesteps": model.sampling_timesteps.tolist(),
                          "mean_ms": 1000*float(np.mean(times)), "p95_ms": 1000*float(np.percentile(times, 95)),
                          "paired_path_rmse_vs_ddpm": float(np.sqrt(np.mean((all_paths-reference)**2))),
                          "note": "Paired RMSE is descriptive, not a quality metric; DDPM uses extra randomness."}
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump({"metadata": metadata, "measurements": measurements}, handle, indent=2)


def cmd_compare():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", nargs="+", required=True,
        help="Directory (all *.npz inside) or one or more NPZ files.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--embodiment", type=int, choices=(0, 1, 2), default=None,
        help="Override embodiment; default read from the observations.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds.")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--ddim-grids", action="append", default=[],
        metavar="NAME=9,6,4,2,0",
        help="Extra DDIM config with an explicit descending timestep grid "
             "(repeatable). Compared against the DDPM-10 reference.")
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--plot-observations", type=int, default=8,
        help="Number of observations to render overlay plots for (0 = none).")
    parser.add_argument("--output-dir", default="outputs/ddim_ddpm_comparison")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")

    records = load_observations(args)
    embodiments = {r["embodiment"] for r in records}
    if len(embodiments) > 1:
        parser.error(f"Observations span multiple embodiments {embodiments}; "
                     "pass --embodiment and split into homogeneous batches.")
    embodiment = args.embodiment if args.embodiment is not None else records[0]["embodiment"]
    if args.embodiment is not None and args.embodiment != records[0]["embodiment"]:
        print(f"WARNING: --embodiment {args.embodiment} overrides observation "
              f"embodiment {records[0]['embodiment']}", flush=True)

    n = len(records)
    rgb = np.stack([r["rgb"] for r in records])          # [n,8,224,224,3]
    depth = np.stack([r["depth"] for r in records])      # [n,224,224,1]
    goal = np.stack([r["pointgoal"] for r in records])   # [n,3]

    model = build_model(args.checkpoint, device)

    config_specs = list(BASE_SPECS)
    for spec in args.ddim_grids:
        if "=" not in spec:
            parser.error(f"--ddim-grids expects NAME=9,6,4,2,0, got {spec!r}")
        name, grid_str = spec.split("=", 1)
        try:
            grid = [int(t) for t in grid_str.split(",")]
        except ValueError:
            parser.error(f"invalid timestep grid {spec!r}")
        # validate eagerly so a bad grid fails before the long model loop
        try:
            resolve_grid("ddim", grid)
        except ValueError as exc:
            parser.error(str(exc))
        config_specs.append((name, "ddim", grid, "a"))
    configs = tuple(s[0] for s in config_specs)
    groups = ("ddpm_self", "ddim10_vs_ddpm", "ddim5_vs_ddpm") + tuple(
        f"{s[0]}_vs_ddpm" for s in config_specs[len(BASE_SPECS):])
    grid_meta = {s[0]: (s[2] if isinstance(s[2], list) else
                        resolve_grid(s[1], s[2]).tolist()) for s in config_specs}

    per_obs = [{"groups": {g: [] for g in groups},
                "latency": {c: [] for c in configs},
                "paths_seed0": None, "q_seed0": None} for _ in range(n)]

    for s in range(args.seeds):
        g_a = torch.Generator(device=device).manual_seed(args.seed_base + s)
        g_b = torch.Generator(device=device).manual_seed(args.seed_base + 10_000_000 + s)
        noise_a = torch.randn((n * args.candidates, 24, 3), generator=g_a, device=device)
        noise_b = torch.randn((n * args.candidates, 24, 3), generator=g_b, device=device)

        out = {}
        for cfg_idx, (name, sampler, steps_or_grid, noise_ref) in enumerate(config_specs):
            noise = noise_a if noise_ref == "a" else noise_b
            paths, scores, elapsed = run_compare_config(
                model, sampler, steps_or_grid, noise, goal, rgb, depth, embodiment,
                args.seed_base * len(config_specs) + s * len(config_specs) + cfg_idx, device)
            out[name] = (paths, scores)
            per_obs_latency = elapsed / n
            for i in range(n):
                per_obs[i]["latency"][name].append(per_obs_latency)

        for i in range(n):
            pa, qa = out["ddpm_a"][0][i], out["ddpm_a"][1][i]
            pb, qb = out["ddpm_b"][0][i], out["ddpm_b"][1][i]
            per_obs[i]["groups"]["ddpm_self"].append(compare_sets(pa, pb, qa, qb))
            for name, _, _, _ in config_specs:
                if name in ("ddpm_a", "ddpm_b"):
                    continue
                pn, qn = out[name][0][i], out[name][1][i]
                per_obs[i]["groups"][f"{name}_vs_ddpm"].append(compare_sets(pa, pn, qa, qn))
            if s == 0:
                per_obs[i]["paths_seed0"] = {c: out[c][0][i] for c in configs}
                per_obs[i]["q_seed0"] = {c: out[c][1][i] for c in configs}

    # Reduce seeds -> per-observation means, then aggregate across observations.
    report_obs = []
    group_means_by_obs = {g: [] for g in groups}
    latency_by_obs = {c: [] for c in configs}
    for i, obs in enumerate(per_obs):
        entry = {"scene": records[i]["scene"], "sample_idx": records[i]["sample_idx"],
                 "path": records[i]["path"], "groups": {}, "latency_ms": {}}
        for g in groups:
            means = mean_of_dicts(obs["groups"][g])
            entry["groups"][g] = means
            group_means_by_obs[g].append(means)
        for c in configs:
            entry["latency_ms"][c] = 1000.0 * float(np.mean(obs["latency"][c]))
            latency_by_obs[c].append(entry["latency_ms"][c])
        report_obs.append(entry)

    aggregate = {g: aggregate_across_obs(group_means_by_obs[g]) for g in groups}
    latency = {c: {"mean_ms": float(np.mean(latency_by_obs[c]))} for c in configs}

    metadata = {"checkpoint": args.checkpoint, "device": str(device),
                "embodiment": embodiment, "observations": n,
                "candidates": args.candidates, "seeds": args.seeds,
                "seed_base": args.seed_base,
                "timestep_grids": grid_meta,
                "rtc": False, "note": "ADE/FDE are 2D (x,y) metres; "
                "q_win is fraction of observations where candidate max-Q > reference max-Q."}

    os.makedirs(args.output_dir, exist_ok=True)
    report = {"metadata": metadata, "aggregate": aggregate,
              "latency_ms": latency, "per_observation": report_obs}
    report_path = os.path.join(args.output_dir, "comparison.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    # Console summary.
    print(json.dumps(metadata), flush=True)
    for g in groups:
        print(f"\n[{g}]", flush=True)
        for k, v in aggregate[g].items():
            print(f"  {k:32s} {v['mean']: .4f}  ± {v['std']: .4f}", flush=True)
    print("\nLatency (ms per observation, per config):", flush=True)
    for c in configs:
        print(f"  {c:12s} {latency[c]['mean_ms']: .2f}", flush=True)

    if args.plot_observations:
        plot_dir = os.path.join(args.output_dir, "overlays")
        os.makedirs(plot_dir, exist_ok=True)
        for i in range(min(args.plot_observations, n)):
            plot_overlay(records[i], per_obs[i]["paths_seed0"],
                         per_obs[i]["q_seed0"],
                         os.path.join(plot_dir, f"obs_{i:03d}.png"))
        print(f"\nOverlay plots: {plot_dir}/obs_*.png", flush=True)
    print(f"\nReport: {report_path}", flush=True)


def cmd_sweep():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", nargs="+", required=True,
        help="Directory (all *.npz inside) or one or more NPZ files.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--embodiment", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/ddim_rtc_sweep")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")

    records = load_observations(args)
    embodiments = {r["embodiment"] for r in records}
    if len(embodiments) > 1:
        parser.error(f"Observations span multiple embodiments {embodiments}; "
                     "pass --embodiment and split into homogeneous batches.")
    embodiment = args.embodiment if args.embodiment is not None else records[0]["embodiment"]

    rtc_requested = any(rtc for _, _, _, rtc, _ in CONFIGS)
    if rtc_requested and any(r["prev_action"] is None for r in records):
        parser.error("RTC configs require prev_action/valid_segment_len in the "
                     "observations; re-capture with the updated "
                     "capture_ddim_ddpm_obs.sh (saves guidance inputs).")

    n = len(records)
    rgb = np.stack([r["rgb"] for r in records])
    depth = np.stack([r["depth"] for r in records])
    goal = np.stack([r["pointgoal"] for r in records])
    prev_action = np.stack([r["prev_action"] for r in records])  # [n,24,3]
    valid_segment_len = np.array([r["valid_segment_len"] for r in records], dtype=np.int32)

    model = build_model(args.checkpoint, device)

    configs = [c[0] for c in CONFIGS]
    per_obs = [{"groups": {g: [] for g, _, _ in GROUPS},
                "latency": {c: [] for c in configs}} for _ in range(n)]

    for s in range(args.seeds):
        g_a = torch.Generator(device=device).manual_seed(args.seed_base + s)
        g_b = torch.Generator(device=device).manual_seed(args.seed_base + 10_000_000 + s)
        noise_a = torch.randn((n * args.candidates, 24, 3), generator=g_a, device=device)
        noise_b = torch.randn((n * args.candidates, 24, 3), generator=g_b, device=device)

        out = {}
        for cfg_idx, (name, sampler, steps, rtc, noise_ref) in enumerate(CONFIGS):
            noise = noise_a if noise_ref == "a" else noise_b
            paths, scores, elapsed = run_sweep_config(
                model, sampler, steps, rtc, noise, goal, rgb, depth,
                prev_action, valid_segment_len, embodiment,
                args.seed_base * len(CONFIGS) + s * len(CONFIGS) + cfg_idx, device)
            out[name] = (paths, scores)
            per_obs_latency = elapsed / n
            for i in range(n):
                per_obs[i]["latency"][name].append(per_obs_latency)

        for i in range(n):
            for gname, cand, ref in GROUPS:
                pc, qc = out[cand][0][i], out[cand][1][i]
                pr, qr = out[ref][0][i], out[ref][1][i]
                per_obs[i]["groups"][gname].append(compare_sets(pr, pc, qr, qc))

    report_obs = []
    group_means = {g: [] for g, _, _ in GROUPS}
    latency_by_obs = {c: [] for c in configs}
    for i, obs in enumerate(per_obs):
        entry = {"scene": records[i]["scene"], "sample_idx": records[i]["sample_idx"],
                 "groups": {}, "latency_ms": {}}
        for gname, _, _ in GROUPS:
            means = mean_of_dicts(obs["groups"][gname])
            entry["groups"][gname] = means
            group_means[gname].append(means)
        for c in configs:
            entry["latency_ms"][c] = 1000.0 * float(np.mean(obs["latency"][c]))
            latency_by_obs[c].append(entry["latency_ms"][c])
        report_obs.append(entry)

    aggregate = {g: aggregate_across_obs(group_means[g]) for g, _, _ in GROUPS}
    latency = {c: {"mean_ms": float(np.mean(latency_by_obs[c]))} for c in configs}
    config_meta = {name: {"sampler": sampler, "steps": steps, "rtc": rtc,
                          "grid": sampling_timesteps(sampler, steps).tolist()}
                   for name, sampler, steps, rtc, _ in CONFIGS}

    metadata = {"checkpoint": args.checkpoint, "device": str(device),
                "embodiment": embodiment, "observations": n,
                "candidates": args.candidates, "seeds": args.seeds,
                "seed_base": args.seed_base, "configs": config_meta,
                "note": "ADE/FDE are 2D (x,y) metres; q_win is the fraction of "
                "observations where candidate max-Q > reference max-Q."}

    os.makedirs(args.output_dir, exist_ok=True)
    report = {"metadata": metadata, "aggregate": aggregate,
              "latency_ms": latency, "per_observation": report_obs}
    report_path = os.path.join(args.output_dir, "sweep.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    # Console summary: compact tables.
    def row(gname):
        a = aggregate[gname]
        return (f"{a['matched_ade']['mean']:.3f}±{a['matched_ade']['std']:.3f}  "
                f"{a['preferred_ade']['mean']:.3f}±{a['preferred_ade']['std']:.3f}  "
                f"{a['q_max_diff']['mean']:+.3f}  {a['q_win']['mean']:.2f}")

    print(json.dumps(metadata, default=str), flush=True)
    print("\n=== RTC OFF: DDIM-k vs DDPM-10 (matched_ade  pref_ade  q_max_diff  q_win) ===")
    print(f"  {'ddpm_self':20s} {row('ddpm_self')}")
    for k in (10, 8, 6, 5, 4, 3, 2):
        g = f"ddim{k}_vs_ddpm"
        print(f"  {g:20s} {row(g)}   ({latency[f'ddim{k}']['mean_ms']:.1f} ms)")
    print("\n=== RTC ON: DDIM-k vs DDPM-10 (both guided) ===")
    print(f"  {'ddpm10_rtc':20s} {latency['ddpm10_rtc']['mean_ms']:.1f} ms")
    for k in (10, 6, 5, 4):
        g = f"ddim{k}_rtc_vs_ddpm_rtc"
        print(f"  {g:20s} {row(g)}   ({latency[f'ddim{k}_rtc']['mean_ms']:.1f} ms)")
    print("\n=== RTC effect (on vs off, same sampler/steps) ===")
    for name in ("ddpm10", "ddim10", "ddim6", "ddim5", "ddim4"):
        g = f"{name}_rtc_effect"
        print(f"  {g:20s} {row(g)}")
    print(f"\nReport: {report_path}", flush=True)


STAGES = {
    "benchmark": cmd_benchmark,
    "compare": cmd_compare,
    "sweep": cmd_sweep,
}

_HELP = """\
usage: python -m ddim.cli <stage> [args]

  benchmark  full-network latency on identical preprocessed observations
  compare    offline DDPM vs DDIM trajectory-distribution comparison
  sweep      sampler/step/RTC matrix sweep
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return
    stage = argv[0]
    if stage not in STAGES:
        print(f"unknown stage: {stage}\n\n{_HELP}", file=sys.stderr)
        raise SystemExit(2)
    sys.argv = argv
    STAGES[stage]()


if __name__ == "__main__":
    main()
