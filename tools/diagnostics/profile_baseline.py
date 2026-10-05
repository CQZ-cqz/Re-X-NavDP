"""Replay real observations; measure uninstrumented latency and diagnostic stages.

Scope: preprocessed numpy observations through candidate/Q/top-2 numpy outputs.
Excludes preprocessing, RPC, recovery state machine, MPC and simulator. Stage
timings synchronize CUDA and include CPU time; they perturb normal scheduling.
Normal-mode guidance is assumed because NPZ files do not record recovery state.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "x-navdp")))
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment


def stats(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


@contextmanager
def instrument(model, sync, totals, counts):
    enc = model.rgbd_encoder
    targets = [(enc, "forward", "encoder_total_inclusive"),
               (enc.rgb_model, "get_intermediate_layers", "rgb_vit"),
               (enc.depth_model, "get_intermediate_layers", "depth_vit"),
               (enc.former_net, "forward", "fusion_attention"),
               (model, "predict_noise", "base_denoiser"),
               (model, "predict_noise_ft", "ft_denoiser"),
               (model.noise_scheduler, "step", "scheduler"),
               (model, "pinv_corrected_velocity", "rtc_gradient"),
               (model, "smooth_trajectory", "smooth_actions"),
               (model, "smooth_cumulative_trajectory", "smooth_q_path"),
               (model, "predict_pointgoal_q", "q_scoring")]
    saved = []
    def wrap(original, label):
        def timed(*args, **kwargs):
            sync()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            sync()
            totals[label] += (time.perf_counter() - start) * 1000
            counts[label] += 1
            return result
        return timed
    try:
        for obj, attr, label in targets:
            saved.append((obj, attr, attr in obj.__dict__, obj.__dict__.get(attr)))
            setattr(obj, attr, wrap(getattr(obj, attr), label))
        yield
    finally:
        for obj, attr, existed, original in reversed(saved):
            if existed:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-observations", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-synthetic-prefix", action="store_true",
                        help="Missing RTC history: use zero actions and prefix length 6; timing only")
    args = parser.parse_args()
    if min(args.runs, args.max_observations, args.threads) < 1 or args.warmup < 0:
        parser.error("positive counts and nonnegative warmup required")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    def sync():
        torch.cuda.synchronize(device)
    root = Path(args.observations)
    files = sorted(root.rglob("*.npz")) if root.is_dir() else [root]
    if not files:
        parser.error("no observations found")
    files = [files[i] for i in np.linspace(0, len(files)-1,
             min(len(files), args.max_observations), dtype=int)]
    records = []
    synthetic_prefix_files = []
    for path in files:
        with np.load(path) as data:
            rec = {key: data[key].copy() for key in
                   ("rgb", "depth", "pointgoal", "embodiment")}
            if {"prev_action", "valid_segment_len"}.issubset(data.files):
                rec.update({key: data[key].copy() for key in ("prev_action", "valid_segment_len")})
            elif args.allow_synthetic_prefix:
                rec["prev_action"] = np.zeros((24, 3), dtype=np.float32)
                rec["valid_segment_len"] = np.array([6], dtype=np.int64)
                synthetic_prefix_files.append(str(path))
            else:
                parser.error(f"{path} lacks RTC history; explicit --allow-synthetic-prefix for timing only")
        if rec["pointgoal"].ndim == 1:
            for key in ("rgb", "depth", "pointgoal", "prev_action"):
                rec[key] = rec[key][None]
        rec["valid_segment_len"] = rec["valid_segment_len"].reshape(-1)
        rec["embodiment"] = int(rec["embodiment"].item())
        if rec["pointgoal"].shape != (1, 3):
            raise ValueError("benchmark expects batch=1")
        records.append(rec)
    model = NavDP_Policy_Embodiment(temporal_depth=16, device=str(device)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(str(incompatible.missing_keys))
    del state
    model.eval()
    def predict(i):
        rec = records[i % len(records)]
        torch.manual_seed(1234 + i)
        return model.predict_pointgoal_action_with_guidance(
            rec["pointgoal"], rec["rgb"], rec["depth"], 8,
            rec["valid_segment_len"].copy(), rec["prev_action"], 0, 23,
            np.array([.5]*6 + [.05]*2), guidance_step=5,
            embodiment=rec["embodiment"])
    report = {"metadata": {**vars(args), "gpu": torch.cuda.get_device_name(device),
              "torch": torch.__version__, "precision": "float32", "batch": 1,
              "candidates": 8, "sampler": model.sampler,
              "timesteps": model.sampling_timesteps.tolist(), "scope": __doc__,
              "observations_used": [str(p) for p in files],
              "synthetic_rtc_prefix_files": synthetic_prefix_files,
              "valid_prefix_lengths": [r["valid_segment_len"].tolist() for r in records],
              "unexpected_checkpoint_keys": len(incompatible.unexpected_keys)},
              "measurements": []}
    for rtc in (True, False):
        model.rtc_enabled = rtc
        for i in range(args.warmup):
            predict(i)
        sync()
        resident = torch.cuda.memory_allocated(device) / 2**20
        torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for i in range(args.runs):
            sync()
            start = time.perf_counter()
            output = predict(i)
            sync()
            latencies.append((time.perf_counter()-start)*1000)
            if not all(np.isfinite(x).all() for x in output[:3]):
                raise RuntimeError("nonfinite prediction")
        memory = {"resident_allocated_mib": resident,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20}
        reference = predict(0)
        stages, counts = defaultdict(float), defaultdict(int)
        rows, instrumented = [], []
        with instrument(model, sync, stages, counts):
            for i in range(args.runs):
                stages.clear()
                counts.clear()
                sync()
                start = time.perf_counter()
                output = predict(i)
                sync()
                instrumented.append((time.perf_counter()-start)*1000)
                if i == 0:
                    for a, b in zip(reference[:3], output[:3]):
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
                stages["encoder_other"] = stages["encoder_total_inclusive"] - sum(
                    stages[k] for k in ("rgb_vit", "depth_vit", "fusion_attention"))
                exclusive = sum(v for k, v in stages.items() if k != "encoder_total_inclusive")
                stages["other"] = instrumented[-1] - exclusive
                rows.append(dict(stages))
        result = {"rtc": rtc, "latency_ms": stats(latencies), "memory": memory,
                  "instrumented_latency_ms": stats(instrumented),
                  "stage_ms": {k: stats([r[k] for r in rows]) for k in rows[0]},
                  "stage_calls_per_prediction": dict(counts),
                  "instrumentation_output_equivalence": "passed"}
        report["measurements"].append(result)
        print(json.dumps(result), flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
