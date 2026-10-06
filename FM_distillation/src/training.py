"""Consolidated FM_distillation library module (entry points live in cli.py)."""

from rexnavdp import BASE, ROOT

"""Offline FM pipeline: freeze -> resumable label -> train -> original-Q evaluate.

Never starts the simulator or collection. GPU subcommands must be run explicitly.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time

from FM_distillation.src.storage import (atomic_json, digest, freeze, LabelCache,
    read_json, sample_rows, validate_splits, verify_code, writer_lock)


def load_teacher(snapshot, device):
    from bridge.teacher_adapter import load_checkpoint
    verify_code(snapshot)
    if digest(snapshot["checkpoint"]) != snapshot["teacher_sha256"]:
        raise ValueError("teacher checkpoint changed")
    return load_checkpoint(snapshot["checkpoint"], device, rtc_enabled=False)


def label_one(teacher, row, snapshot, seed):
    import numpy as np
    import torch
    from FM_distillation.src.fm_data import load_record, teacher_tap, validate_observation, validate_label
    if digest(row["source"]) != row["sha256"]:
        raise ValueError("observation changed since snapshot")
    obs, src = load_record(row["source"])
    validate_observation(obs, src)
    for name in ("scene", "split", "run_id", "episode_id", "step"):
        if src[name] != row[name]:
            raise ValueError(f"source provenance mismatch: {name}")
    torch.manual_seed(seed)
    initial = torch.randn(8, 24, 3, device="cuda:0")
    cuda_rng, cpu_rng = torch.cuda.get_rng_state(0), torch.get_rng_state()

    def predict():
        torch.cuda.set_rng_state(cuda_rng, 0)
        torch.set_rng_state(cpu_rng)
        # RTC is explicitly off. Gradients are unnecessary for offline targets.
        with torch.no_grad():
            return teacher.predict_pointgoal_action_with_guidance(
                obs["pointgoal"][None], obs["rgb"][None], obs["depth"][None], 8,
                np.array([src["valid_segment_len"]]), obs["prev_action"][None], 0, 23,
                np.array(src["guidance_factor"]), guidance_step=5,
                embodiment=src["embodiment"], initial_noise=initial)
    reference = predict()
    with teacher_tap(teacher) as tapped:
        actual = predict()
    for a, b in zip(reference[:3], actual[:3]):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        kwargs = dict(num_points=25, smooth_factor=.5, weight=teacher.weight)
        smooth = teacher.smooth_trajectory(tapped["raw"], **kwargs)
        qpath = teacher.smooth_cumulative_trajectory(tapped["raw"].div(4).cumsum(1), **kwargs)
        torch.testing.assert_close(smooth, tapped["smoothed_actions"])
        torch.testing.assert_close(qpath, tapped["q_path"])
        q1, q2 = teacher.predict_pointgoal_q(qpath, tapped["rgbd"], tapped["goal"],
                                           is_target=False, embodiment=src["embodiment"])
        torch.testing.assert_close(q1, tapped["q1"])
        torch.testing.assert_close(q2, tapped["q2"])
    def cpu(x):
        return x.detach().cpu().numpy()
    arrays = dict(raw_action_deltas=cpu(tapped["raw"]), smoothed_actions=cpu(smooth),
                  q_path=cpu(qpath), trajectories=actual[0][0], scores=actual[1][0],
                  q1=cpu(q1), q2=cpu(q2), top_indices=cpu((-(q1+q2)/2).argsort()[:2]),
                  top_trajectories=actual[2][0], rgbd_embed=cpu(tapped["rgbd"])[0],
                  goal_embed=cpu(tapped["goal"])[0], initial_noise=cpu(initial),
                  sampler_cuda_rng_state=cpu(cuda_rng), sampler_cpu_rng_state=cpu(cpu_rng))
    meta = {**src, "kind": "label", "rtc_enabled": False, "candidates": 8,
            "teacher_sha256": snapshot["teacher_sha256"], "observation": row["file"],
            "observation_sha256": row["sha256"], "seed": seed,
            "target_space": "raw_pre_smoothing_action_deltas", "tap_equivalence": True,
            "q_recompute_verified": True}
    validate_label(arrays, meta)
    return arrays, meta


def label(args):
    from FM_distillation.src.fm_data import load_record, save_record, validate_label
    snapshot = read_json(args.snapshot)
    validate_splits(snapshot["records"], require_validation=False)
    verify_code(snapshot)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    signature = dict(snapshot_sha256=digest(args.snapshot), seed=args.seed,
                     teacher_sha256=snapshot["teacher_sha256"], candidates=8, rtc_enabled=False,
                     pipeline_sha256=source_hashes())
    donor = Path(args.reuse_labels).resolve() if getattr(args, "reuse_labels", None) else None
    if donor:
        if donor == root.resolve():
            raise ValueError("reuse source must differ from output")
        previous = read_json(donor / "label_manifest.json")
        for key in ("seed", "teacher_sha256", "candidates", "rtc_enabled", "pipeline_sha256"):
            if previous[key] != signature[key]:
                raise ValueError(f"reuse label configuration mismatch: {key}")
    with writer_lock(root):
        manifest = root / "label_manifest.json"
        if manifest.exists():
            if read_json(manifest) != signature:
                raise ValueError("label resume configuration changed; use a new output")
        else:
            if any(root.glob("*/*.npz")) or (root / "COMPLETE.json").exists():
                raise ValueError("unrecognized label output")
            atomic_json(manifest, signature)
        teacher, completed, new_count = None, [], 0
        started, labeling_seconds, new_bytes = time.perf_counter(), 0., 0
        def report(status):
            result = dict(status=status, processed=len(completed), total=len(snapshot["records"]),
                          new_labels=new_count, new_npz_bytes=new_bytes,
                          elapsed_s=time.perf_counter()-started, teacher_labeling_s=labeling_seconds,
                          mean_labeling_s=labeling_seconds/new_count if new_count else None,
                          note="labeling includes two teacher passes and Q verification; not inference latency")
            if teacher is not None:
                import torch
                if torch.cuda.is_initialized():
                    result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                                  peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
            atomic_json(root/"label_report.json", result)
        for i, row in enumerate(snapshot["records"]):
            path = root / row["id"]
            receipt = path.with_suffix(".json")
            record_seed = int(hashlib.sha256(f'{args.seed}:{row["id"]}:{row["sha256"]}'.encode()).hexdigest()[:8], 16)
            if not path.exists() and donor and (donor / row["id"]).exists():
                old = donor / row["id"]
                checksum = read_json(old.with_suffix(".json"))["sha256"]
                _, metadata = LabelCache(donor, snapshot, 0).get({**row, "label_sha256": checksum})
                if metadata["seed"] != record_seed:
                    raise ValueError("reused label seed mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                # Same filesystem: immutable hard link, no extra NPZ disk cost.
                # Cross-filesystem: stage a copy before atomic publication.
                try:
                    os.link(old, path)
                except OSError as exc:
                    import errno
                    if exc.errno != errno.EXDEV:
                        raise
                    temporary = path.with_suffix(".pending")
                    shutil.copyfile(old, temporary)
                    if digest(temporary) != checksum:
                        raise ValueError("reused label copy checksum mismatch")
                    os.replace(temporary, path)
                atomic_json(receipt, {"sha256": checksum})
            if path.exists():
                # A crash between sample commit and receipt commit is recoverable.
                checksum = digest(path)
                if receipt.exists() and read_json(receipt) != {"sha256": checksum}:
                    raise ValueError(f"corrupted label: {path}")
                arrays, meta = load_record(path)
                validate_label(arrays, meta)
                if meta["seed"] != record_seed:
                    raise ValueError("label seed mismatch")
                cached = {**row, "label_sha256": checksum}
                LabelCache(root, snapshot, 0).get(cached)
            else:
                if receipt.exists():
                    raise ValueError(f"label missing despite receipt: {path}")
                if args.max_new_records and new_count >= args.max_new_records:
                    report("pilot_stopped")
                    print(f"Stopped after {new_count} new labels for a pilot check; no COMPLETE marker. "
                          "Resume the same command without --max-new-records to finish.", flush=True)
                    return
                if teacher is None:
                    teacher = load_teacher(snapshot, "cuda:0")
                label_started = time.perf_counter()
                arrays, meta = label_one(teacher, row, snapshot, record_seed)
                labeling_seconds += time.perf_counter()-label_started
                path.parent.mkdir(parents=True, exist_ok=True)
                # An interrupted write never appears as a completed .npz sample.
                temporary = path.with_suffix(".pending")
                if temporary.exists():
                    temporary.unlink()  # only this pipeline's uncommitted temporary
                save_record(temporary, arrays, meta)
                os.replace(temporary, path)
                checksum = digest(path)
                new_count += 1
                new_bytes += path.stat().st_size
            if not receipt.exists():
                atomic_json(receipt, {"sha256": checksum})
            completed.append({**row, "label_sha256": checksum})
            if (i+1) % 100 == 0 or i == 0 or i+1 == len(snapshot["records"]):
                print(f'label {i+1}/{len(snapshot["records"])}: {row["id"]}', flush=True)
        atomic_json(root / "COMPLETE.json", {**signature, "status": "complete", "records": completed})
        report("complete")


def source_hashes():
    names = ("../../FM_distillation/src/training.py", "../../FM_distillation/src/storage.py",
             "../../FM_distillation/src/flow_generator.py", "../../FM_distillation/src/fm_learning.py", "../../FM_distillation/src/fm_data.py")
    result = {name: digest(BASE / name) for name in names}
    for part in ("FM_distillation", "bridge", "ddim/src"):
        for path in sorted((ROOT / part).rglob("*.py")):
            result[os.path.relpath(path, BASE)] = digest(path)
    return result


def labeled_data(args):
    snapshot = read_json(args.snapshot)
    complete = read_json(Path(args.labels) / "COMPLETE.json")
    if (complete["status"] != "complete" or complete["snapshot_sha256"] != digest(args.snapshot)
            or complete["teacher_sha256"] != snapshot["teacher_sha256"]):
        raise ValueError("incomplete/mismatched labeling")
    original = [{k: v for k, v in row.items() if k != "label_sha256"} for row in complete["records"]]
    if original != snapshot["records"]:
        raise ValueError("labeled observation set differs from frozen snapshot")
    groups = validate_splits(complete["records"])
    return snapshot, groups, LabelCache(args.labels, snapshot)


def labeled_mixed_data(dataset):
    """Load a merge-success mixed dataset (failure-filtered, absolute label_path)."""
    from FM_distillation.src.merging import load_mixed_dataset
    return load_mixed_dataset(dataset)


def batch(cache, rows, candidates, device):
    import numpy as np
    import torch
    loaded = [cache.get(row) for row in rows]
    def tensor(values):
        return torch.from_numpy(np.stack(values)).to(device)
    result = dict(action_deltas=tensor([a["raw_action_deltas"][c] for (a, _), c in zip(loaded, candidates)]),
                  goal_embed=tensor([a["goal_embed"] for a, _ in loaded]),
                  rgbd_embed=tensor([a["rgbd_embed"] for a, _ in loaded]),
                  embodiment=torch.tensor([m["embodiment"] for _, m in loaded], device=device))
    if all("initial_noise" in a for a, _ in loaded):
        result["initial_noise"] = tensor([a["initial_noise"][c] for (a, _), c in zip(loaded, candidates)])
    return result


def validation_rows(groups, count):
    import numpy as np
    # Spaced deterministically over the scene, never split adjacent frames into train/val.
    return {scene: [rows[i] for i in np.linspace(0, len(rows)-1, min(count, len(rows)), dtype=int)]
            for (split, scene), rows in sorted(groups.items()) if split == "validation"}


def evaluate(model, cache, selected, device, seed, teacher=None):
    import numpy as np
    import torch
    from FM_distillation.src.fm_learning import candidate_metrics
    from FM_distillation.src.flow_generator import generate_and_rank
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    scenes = {}
    with torch.no_grad():
        for scene, rows in selected.items():
            metrics = []
            for row in rows:
                inputs = batch(cache, [row]*8, list(range(8)), device)
                noise = inputs.pop("initial_noise", None)
                if noise is None:
                    noise = torch.randn(8,24,3,generator=rng).to(device)
                loss = model.flow_loss(**inputs, noise=noise,
                                       t=torch.rand(8,generator=rng).to(device)).item()
                goal, rgbd, embodiment = (inputs[k][:1] for k in ("goal_embed", "rgbd_embed", "embodiment"))
                noise = torch.randn(1,8,24,3,generator=rng).to(device)
                if teacher is None:
                    prediction = model.sample(goal, rgbd, embodiment, candidates=8, steps=4, initial_noise=noise)
                else:
                    ranked = generate_and_rank(teacher, model, goal, rgbd, embodiment,
                                               candidates=8, steps=4, initial_noise=noise)
                    prediction = ranked["raw_action_deltas"]
                arrays, _ = cache.get(row)
                target = torch.from_numpy(arrays["raw_action_deltas"])[None]
                scores = torch.from_numpy(arrays["scores"])[None]
                result = candidate_metrics(prediction.cpu(), target, scores)
                result["fm_loss"] = loss
                if teacher is not None:
                    result.update(student_q_mean=ranked["scores"].mean().item(),
                                  student_q_top2=ranked["scores"].topk(2).values.mean().item(),
                                  teacher_q_top2=scores.topk(2).values.mean().item())
                    teacher_top = torch.from_numpy(arrays["top_trajectories"])[None].to(device)
                    distances = (ranked["top_trajectories"][:,:,None,:,:2] - teacher_top[:,None,:,:,:2]).norm(dim=-1).mean(-1)
                    result["selected_top2_nearest_teacher_ade_m"] = distances.min(-1).values.mean().item()
                metrics.append(result)
            scenes[scene] = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    aggregate = {k: float(np.mean([m[k] for m in scenes.values()])) for k in next(iter(scenes.values()))}
    return dict(per_scene=scenes, scene_balanced=aggregate, steps=4, candidates=8,
                observations={s: len(r) for s, r in selected.items()},
                note="offline held-out-scene subset; Q is not collision/success evidence; RTC disabled")


def save_checkpoint(path, value):
    import torch
    temporary = Path(path).with_suffix(".pending")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(args):
    import torch
    from FM_distillation.src.flow_generator import CompactFlowGenerator
    if getattr(args, "dataset", None):
        snapshot, groups, cache = labeled_mixed_data(args.dataset)
        identity = dict(dataset_sha256=digest(Path(args.dataset) / "snapshot.json"),
                        dataset_complete_sha256=digest(Path(args.dataset) / "COMPLETE.json"),
                        teacher_sha256=snapshot["teacher_sha256"])
    else:
        snapshot, groups, cache = labeled_data(args)
        identity = dict(snapshot_sha256=digest(args.snapshot),
                        labels_sha256=digest(Path(args.labels) / "COMPLETE.json"),
                        teacher_sha256=snapshot["teacher_sha256"])
    torch.manual_seed(args.seed)
    teacher = load_teacher(snapshot, "cpu")
    model = CompactFlowGenerator(teacher, depth=4)
    from FM_distillation.src.fm_backbone import backbone_meta
    teacher_state, teacher_meta = teacher.state_dict(), backbone_meta(teacher)
    del teacher  # cached-feature training: no teacher actor/encoder/Q on GPU
    model = model.to("cuda:0")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(args.seed)
    selected = validation_rows(groups, args.val_per_scene)
    signature = dict(**identity, source_sha256=source_hashes(),
                     depth=4, width=384, time_scale=9., precision="float32",
                     microbatch=args.microbatch, accumulate=args.accumulate, lr=args.lr,
                     seed=args.seed, val_per_scene=args.val_per_scene, eval_every=args.eval_every)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output):
        start, best = 0, float("inf")
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            if state["signature"] != signature:
                raise ValueError("resume data/model/optimizer/source configuration changed")
            # Resume in a fresh directory: cannot mix abandoned log tails or overwrite checkpoints.
            if any(p.name != ".writer.lock" for p in output.iterdir()):
                raise ValueError("resume output must be a new directory")
            model.load_state_dict(state["student"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            rng.set_state(state["batch_rng"])
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], 0)
            start, best = state["step"], state["best_val_loss"]
        elif any(p.name != ".writer.lock" for p in output.iterdir()):
            raise ValueError("training output not empty; use --resume with a new output")
        if args.steps <= start:
            raise ValueError("--steps is total optimizer updates and must exceed resumed step")
        atomic_json(output/"config.json", {**signature, **vars(args)})
        atomic_json(output/"validation_rows.json", {s: [r["id"] for r in rows] for s, rows in selected.items()})
        torch.cuda.reset_peak_memory_stats(0)
        wall = time.perf_counter()
        with (output/"metrics.jsonl").open("x") as log:
            for step in range(start+1, args.steps+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running = 0.
                for _ in range(args.accumulate):
                    rows = sample_rows(groups, args.microbatch, rng)
                    candidates = torch.randint(8, (args.microbatch,), generator=rng).tolist()
                    inputs = batch(cache, rows, candidates, "cuda:0")
                    noise = inputs.pop("initial_noise", None)
                    if noise is None:
                        noise = torch.randn(args.microbatch, 24, 3, generator=rng).to("cuda:0")
                    loss = model.flow_loss(**inputs,
                        noise=noise,
                        t=torch.rand(args.microbatch,generator=rng).to("cuda:0"))
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite loss")
                    (loss/args.accumulate).backward()
                    running += loss.item()/args.accumulate
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, train_loss=running, grad_norm=float(norm))
                if step % args.eval_every == 0 or step == args.steps:
                    record["validation"] = evaluate(model, cache, selected, "cuda:0", args.seed+1000)
                    val = record["validation"]["scene_balanced"]["fm_loss"]
                    improved = val < best
                    best = min(best, val)
                    checkpoint = output / f"step_{step:08d}.pt"
                    save_checkpoint(checkpoint, dict(student=model.state_dict(), optimizer=optimizer.state_dict(),
                        signature=signature, step=step, best_val_loss=best, batch_rng=rng.get_state(),
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0)))
                    atomic_json(output/"latest.json", {"checkpoint": checkpoint.name, "step": step})
                    if improved:
                        atomic_json(output/"best.json", {"checkpoint": checkpoint.name, "val_loss": val})
                if step == start+1 or step % args.log_every == 0 or "validation" in record:
                    record.update(elapsed_s=time.perf_counter()-wall,
                                  peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                                  peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
                    print(json.dumps(record), flush=True)
                log.write(json.dumps(record, allow_nan=False)+"\n")
                log.flush()
        from FM_distillation.src.fm_backbone import export_deploy_checkpoint
        export_deploy_checkpoint(model, teacher_state, teacher_meta, signature, output/"deploy.pt")
        print(f"Deploy checkpoint written: {output/'deploy.pt'}", flush=True)


def rank(args):
    import torch
    from FM_distillation.src.flow_generator import CompactFlowGenerator
    snapshot, groups, cache = labeled_data(args)
    state = torch.load(args.student, map_location="cpu", weights_only=True)
    if (state["signature"]["snapshot_sha256"] != digest(args.snapshot) or
            state["signature"]["labels_sha256"] != digest(Path(args.labels)/"COMPLETE.json") or
            state["signature"]["teacher_sha256"] != snapshot["teacher_sha256"]):
        raise ValueError("student provenance mismatch")
    teacher = load_teacher(snapshot, "cuda:0")
    model = CompactFlowGenerator(teacher, depth=4).to("cuda:0")
    model.load_state_dict(state["student"], strict=True)
    report = evaluate(model, cache, validation_rows(groups, args.val_per_scene), "cuda:0", args.seed+1000, teacher)
    report.update(student_sha256=digest(args.student), step=state["step"], seed=args.seed,
                  source_sha256=source_hashes(), snapshot_sha256=digest(args.snapshot))
    with Path(args.output).open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


"""Formal offline FM stage: EVERY optimizer update includes every train scene.

Explicit stage transition from the legacy checkpoint; no legacy source edits.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from FM_distillation.src.storage import atomic_json, digest, writer_lock


def all_scene_rows(groups, per_scene, rng):
    import torch
    rows = []
    for key in sorted(groups):
        if key[0] != "train":
            continue
        pool = groups[key]
        rows.extend(pool[i] for i in torch.randint(len(pool),(per_scene,),generator=rng).tolist())
    order = torch.randperm(len(rows),generator=rng).tolist()
    return [rows[i] for i in order]


"""All-scene, all-eight-candidate CFM; frozen conditions, no RTC/set/Q loss.

New entry point preserves legacy source fingerprints and checkpoint compatibility.
--steps is the TOTAL optimizer step, including the initialization checkpoint.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from FM_distillation.src.storage import atomic_json, digest, writer_lock


def all_candidate_batch(cache, rows):
    """Load each observation once, retaining CPU tensors [B,K,...]."""
    import numpy as np
    import torch
    loaded = [cache.get(row) for row in rows]
    def stack(key):
        return torch.from_numpy(np.stack([a[key] for a, _ in loaded]))
    actions = stack("raw_action_deltas")
    if actions.shape != (len(rows), 8, 24, 3):
        raise ValueError("expected teacher raw deltas [B,8,24,3]")
    return dict(action_deltas=actions, goal_embed=stack("goal_embed"),
                rgbd_embed=stack("rgbd_embed"), initial_noise=stack("initial_noise"),
                embodiment=torch.tensor([m["embodiment"] for _, m in loaded]))


def candidate_errors(model, inputs, noise, t):
    """Unreduced legacy CFM objective, one scalar per flattened candidate."""
    target = inputs["action_deltas"].detach()
    x = (1-t[:, None, None])*noise + t[:, None, None]*target
    predicted = model(x, t, inputs["goal_embed"].detach(),
                      inputs["rgbd_embed"].detach(), inputs["embodiment"])
    return (predicted-(target-noise)).square().mean(dim=(-1, -2))


def all_candidate_loss(model, data, device, rng, chunk_size, *, backward=False):
    """Exact mean over B*K, including a short final chunk; one optimizer step outside.

    All random draws happen before chunking so changing chunk size preserves pairs.
    Only each chunk's repeated conditions are moved to the GPU.
    """
    import torch
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    b, k, h, d = data["action_deltas"].shape
    n = b*k
    targets = data["action_deltas"].reshape(n, h, d)
    if "initial_noise" in data:
        # Latent alignment: pair each teacher trajectory with the exact initial
        # noise that generated it (stored in the label), preserving mode identity.
        noise = data["initial_noise"].reshape(n, h, d)
    else:
        noise = torch.randn(n, h, d, generator=rng)
    t = torch.rand(n, generator=rng)
    errors = []
    for start in range(0, n, chunk_size):
        end = min(start+chunk_size, n)
        owners = torch.arange(start, end)//k
        inputs = {key: value[owners].to(device) for key, value in data.items()
                  if key != "action_deltas"}
        inputs["action_deltas"] = targets[start:end].to(device)
        per_candidate = candidate_errors(model, inputs, noise[start:end].to(device),
                                         t[start:end].to(device))
        if not torch.isfinite(per_candidate).all():
            raise ValueError("nonfinite candidate loss")
        if backward:
            (per_candidate.sum()/n).backward()
        errors.append(per_candidate.detach().cpu())
    return torch.cat(errors).reshape(b, k), t.reshape(b, k)


def pairwise_diversity(deltas):
    """Mean pairwise cumulative-XY distance across candidates. [B,K,24,3] -> [B]."""
    import torch
    path = deltas.cumsum(-2) / 4
    matrix = (path[:, :, None, :, :2] - path[:, None, :, :, :2]).norm(dim=-1).mean(-1)
    k = deltas.shape[1]
    mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=deltas.device), diagonal=1)
    return matrix[:, mask].mean(-1)


def distribution_loss(model, data, device, *, candidates=8, steps=4,
                      lambda_mu=0.1, lambda_sigma=0.5):
    """Symmetric 2nd-order distribution matching on the cumulative path.

    Matches the per-candidate mean (location) and per-waypoint standard deviation
    (spread), both symmetric so the student is pulled back from over-spreading as
    well as from mode collapse. Differentiable through ``model.sample_with_grad``.
    """
    import torch
    samples = model.sample_with_grad(data["goal_embed"].to(device), data["rgbd_embed"].to(device),
                                     data["embodiment"].to(device), candidates=candidates, steps=steps)
    p_s = samples.cumsum(-2) / 4
    p_t = data["action_deltas"].to(device).cumsum(-2) / 4
    mu_s, mu_t = p_s.mean(1), p_t.mean(1)
    std_s, std_t = p_s.std(1, unbiased=False), p_t.std(1, unbiased=False)
    return lambda_mu * (mu_s - mu_t).square().mean() + lambda_sigma * (std_s - std_t).square().mean()


def sinkhorn_distance(C, eps, n_iter=20):
    """Entropy-regularized OT cost for [B, Ks, Kt] cost matrices, uniform marginals.

    Log-domain Sinkhorn-Knopp; differentiable through ``C``. ``eps`` controls the
    softness of the transport plan (small = hard one-to-one, large = soft/mass-
    spreading). Returns a per-batch scalar in the same units as ``C``.
    """
    import torch
    B, Ks, Kt = C.shape
    if eps <= 0 or n_iter < 1:
        raise ValueError("eps must be positive and n_iter positive")
    log_K = -C / eps
    log_a = -torch.log(torch.tensor(Ks, dtype=C.dtype, device=C.device))
    log_b = -torch.log(torch.tensor(Kt, dtype=C.dtype, device=C.device))
    log_u = torch.zeros(B, Ks, dtype=C.dtype, device=C.device)
    log_v = torch.zeros(B, Kt, dtype=C.dtype, device=C.device)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v[:, None, :], dim=2)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, :, None], dim=1)
    P = torch.exp(log_u[:, :, None] + log_K + log_v[:, None, :])
    return (P * C).sum(dim=(1, 2))


def sinkhorn_loss(model, data, device, *, candidates=8, steps=4, eps=0.1, n_iter=20):
    """Symmetric full-distribution matching between on-policy student samples and
    the teacher candidates, via Sinkhorn OT on the cumulative XY trajectory ADE.

    Unlike mean+std matching this sees the whole multimodal structure, and unlike
    a one-sided hinge it penalizes both under- and over-spread. Training-only;
    inference is unchanged. Differentiable through ``model.sample_with_grad``.
    """
    import torch
    samples = model.sample_with_grad(data["goal_embed"].to(device), data["rgbd_embed"].to(device),
                                     data["embodiment"].to(device), candidates=candidates, steps=steps)
    s = samples.cumsum(-2)[..., :2] / 4
    t = data["action_deltas"].to(device).cumsum(-2)[..., :2] / 4
    C = (s[:, :, None] - t[:, None]).norm(dim=-1).mean(dim=-1)  # [B, Ks, Kt] trajectory ADE
    return sinkhorn_distance(C, eps, n_iter).mean()


def loss_diagnostics(errors, times, actions):
    """Distribution and overlapping motion/time groups, NOT fixed-slot modes.

    Motion groups use teacher cumulative XY endpoint, not collision/recovery labels.
    Empty groups carry count=0 and mean=null, never an artificial zero loss.
    """
    import torch
    values = errors.detach().float().cpu()
    endpoint = actions.detach().cpu().sum(dim=-2)[..., :2]/4
    angle = torch.atan2(endpoint[..., 1], endpoint[..., 0]).abs()
    masks = {"backward_endpoint": endpoint[..., 0] < 0,
             "nonbackward_endpoint": endpoint[..., 0] >= 0,
             "large_endpoint_angle_gt60deg": (angle > math.pi/3) & (endpoint.norm(dim=-1) > .1)}
    for i in range(4):
        masks[f"time_quartile_{i}"] = (times >= i/4) & (times < (i+1)/4)
    groups = {}
    for name, mask in masks.items():
        subset = values[mask]
        groups[name] = dict(count=subset.numel(), mean=subset.mean().item() if subset.numel() else None)
    return dict(count=values.numel(), mean=values.mean().item(),
                p50=values.quantile(.5).item(), p90=values.quantile(.9).item(),
                maximum=values.max().item(), mean_observation_max=values.max(dim=1).values.mean().item(),
                groups=groups)


def evaluate_candidates(model, cache, selected, device, seed, chunk_size):
    """Keep legacy generated-set metrics; add fixed-seed per-candidate diagnostics."""
    import torch
    result = evaluate(model, cache, selected, device, seed)
    rng = torch.Generator().manual_seed(seed)
    diagnostics = {}
    with torch.no_grad():
        for scene, rows in selected.items():
            errors, times, actions = [], [], []
            for row in rows:
                data = all_candidate_batch(cache, [row])
                e, t = all_candidate_loss(model, data, device, rng, chunk_size)
                errors.append(e)
                times.append(t)
                actions.append(data["action_deltas"])
            diagnostics[scene] = loss_diagnostics(torch.cat(errors), torch.cat(times), torch.cat(actions))
    result["candidate_diagnostics"] = diagnostics
    result["diagnostics_note"] = "Separate fixed RNG stream; motion proxies are not recovery/collision labels."
    return result


"""Label validation scenes independently without changing the live train labeler."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from FM_distillation.src.storage import atomic_json, digest, freeze, read_json, LabelCache, verify_code, writer_lock


def validation_snapshot(collection, destination):
    destination = Path(destination)
    if destination.exists():
        snapshot = read_json(destination)
        if snapshot["collection"] != str(Path(collection).resolve()):
            raise ValueError("snapshot collection mismatch")
    else:
        with tempfile.TemporaryDirectory(prefix="fm-val-") as temporary:
            snapshot = freeze(collection, Path(temporary)/"snapshot.json", completed_only=True)
        snapshot["records"] = [r for r in snapshot["records"] if r["split"] == "validation"]
        snapshot["scenes"] = [r for r in snapshot["scenes"] if r["split"] == "validation"]
        snapshot["skipped_incomplete"] = [s for s in snapshot["skipped_incomplete"] if s.startswith("validation/")]
        check_snapshot(snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as f:
            json.dump(snapshot, f, allow_nan=False)
    check_snapshot(snapshot)
    return snapshot


def check_snapshot(snapshot):
    if (snapshot["skipped_incomplete"] or not snapshot["records"] or
            {r["scene"] for r in snapshot["scenes"]} != {"easy_5", "hard_5"} or
            {r["scene"] for r in snapshot["records"]} != {"easy_5", "hard_5"} or
            any(r["split"] != "validation" for r in snapshot["records"]+snapshot["scenes"])):
        raise ValueError("requires completed easy_5 and hard_5 validation scenes only")
    if len({r["id"] for r in snapshot["records"]}) != len(snapshot["records"]):
        raise ValueError("duplicate validation record")

