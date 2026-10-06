"""Dual-branch FM training via condition dropout (CFG-style regularizer).

Opt-in experimental variant; it does NOT change the v1 eight-pointgoal protocol
or require new labels. The 8 pointgoal candidates are split into a pointgoal
branch (candidates 0-3, real goal token) and a nogoal branch (candidates 4-7,
zero goal token); both regress the SAME pointgoal teacher deltas. This is
condition dropout: it keeps a goal-independent prior so the student does not
collapse onto a single goal-directed mode (the observed "单峰化" at long steps).

Loss: ``beta * (mild Q-weighted pointgoal) + alpha * (equal nogoal)``; the
pointgoal quality weight never backprops into Q. The earlier ``label-dual``
4-pointgoal + 4-zero-token-target path remains available as a separate variant
via ``validate_dual_label``/``label_dual_one``.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from FM_distillation.src.fm_data import load_record, save_record, teacher_tap, validate_observation
from FM_distillation.src.storage import (atomic_json, digest, read_json, validate_splits,
    verify_code, writer_lock)
from FM_distillation.src.training import (labeled_data, labeled_mixed_data, load_teacher,
    sample_rows, save_checkpoint, source_hashes, validation_rows)

PG = 1  # branch marker: pointgoal candidate
NG = 0  # branch marker: nogoal (zero-token) candidate
BRANCHES = np.array([PG, PG, PG, PG, NG, NG, NG, NG], dtype=np.int64)


def validate_dual_label(arrays, meta):
    """v1 shape/algebra/provenance plus the 4+4 branch contract."""
    from FM_distillation.src.fm_data import validate_label
    validate_label(arrays, meta)
    if meta.get("dual_branches") is not True:
        raise ValueError("not a dual-branch label")
    branches = np.asarray(meta.get("branches"))
    if branches.shape != (8,) or branches.dtype.kind not in "iu" or not np.array_equal(branches, BRANCHES):
        raise ValueError("dual label must be exactly 4 pointgoal then 4 nogoal")
    if meta.get("branch_scoring") != "q_on_real_goal":
        raise ValueError("dual Q must be scored against the real goal")


def label_dual_one(teacher, row, snapshot, seed, device="cuda:0"):
    """Label one observation with 4 pointgoal + 4 nogoal teacher candidates."""
    if digest(row["source"]) != row["sha256"]:
        raise ValueError("observation changed since snapshot")
    obs, src = load_record(row["source"])
    validate_observation(obs, src)
    for name in ("scene", "split", "run_id", "episode_id", "step"):
        if src[name] != row[name]:
            raise ValueError(f"source provenance mismatch: {name}")
    torch.manual_seed(seed)
    pg_noise = torch.randn(4, 24, 3, device=device)
    ng_noise = torch.randn(4, 24, 3, device=device)
    cuda_rng, cpu_rng = torch.cuda.get_rng_state(0), torch.get_rng_state()
    guidance = np.asarray(src["guidance_factor"])

    def predict_pg():
        torch.cuda.set_rng_state(cuda_rng, 0)
        torch.set_rng_state(cpu_rng)
        with torch.no_grad():
            return teacher.predict_pointgoal_action_with_guidance(
                obs["pointgoal"][None], obs["rgb"][None], obs["depth"][None], 4,
                np.array([src["valid_segment_len"]]), obs["prev_action"][None], 0, 23,
                guidance[:4], guidance_step=5, embodiment=src["embodiment"],
                initial_noise=pg_noise)

    def predict_ng():
        torch.cuda.set_rng_state(cuda_rng, 0)
        torch.set_rng_state(cpu_rng)
        zero = torch.zeros(1, 1, teacher.token_dim, device=device)
        with torch.no_grad():
            return teacher.predict_pointgoal_action_with_guidance(
                obs["pointgoal"][None], obs["rgb"][None], obs["depth"][None], 4,
                np.array([src["valid_segment_len"]]), obs["prev_action"][None], 0, 23,
                guidance[4:], guidance_step=5, embodiment=src["embodiment"],
                initial_noise=ng_noise, goal_embed_override=zero)

    pg_ref = predict_pg()
    with teacher_tap(teacher) as pg_tap:
        pg_act = predict_pg()
    ng_ref = predict_ng()
    with teacher_tap(teacher) as ng_tap:
        ng_act = predict_ng()
    for a, b in zip(pg_ref[:3], pg_act[:3]):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(ng_ref[:3], ng_act[:3]):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)

    with torch.no_grad():
        kwargs = dict(num_points=25, smooth_factor=.5, weight=teacher.weight)
        for tap in (pg_tap, ng_tap):
            smooth = teacher.smooth_trajectory(tap["raw"], **kwargs)
            qpath = teacher.smooth_cumulative_trajectory(tap["raw"].div(4).cumsum(1), **kwargs)
            torch.testing.assert_close(smooth, tap["smoothed_actions"])
            torch.testing.assert_close(qpath, tap["q_path"])
            q1, q2 = teacher.predict_pointgoal_q(qpath, tap["rgbd"], tap["goal"],
                                                 is_target=False, embodiment=src["embodiment"])
            torch.testing.assert_close(q1, tap["q1"])
            torch.testing.assert_close(q2, tap["q2"])
        # The nogoal pass must still be Q-scored on the REAL goal, not the zero token.
        torch.testing.assert_close(ng_tap["goal"][0], pg_tap["goal"][0])

    def cpu(x):
        return x.detach().cpu().numpy()

    q1 = np.concatenate([cpu(pg_tap["q1"]), cpu(ng_tap["q1"])])
    q2 = np.concatenate([cpu(pg_tap["q2"]), cpu(ng_tap["q2"])])
    scores = (q1 + q2) / 2
    top_indices = (-scores).argsort()[:2]
    arrays = dict(
        raw_action_deltas=np.concatenate([cpu(pg_tap["raw"]), cpu(ng_tap["raw"])]),
        smoothed_actions=np.concatenate([cpu(pg_tap["smoothed_actions"]), cpu(ng_tap["smoothed_actions"])]),
        q_path=np.concatenate([cpu(pg_tap["q_path"]), cpu(ng_tap["q_path"])]),
        trajectories=np.concatenate([pg_act[0][0], ng_act[0][0]]),
        scores=scores, q1=q1, q2=q2,
        top_indices=top_indices,
        top_trajectories=np.concatenate([pg_act[0][0], ng_act[0][0]])[top_indices],
        rgbd_embed=cpu(pg_tap["rgbd"])[0], goal_embed=cpu(pg_tap["goal"])[0],
        initial_noise=np.concatenate([cpu(pg_noise), cpu(ng_noise)]),
        sampler_cuda_rng_state=cpu(cuda_rng), sampler_cpu_rng_state=cpu(cpu_rng))
    meta = {**src, "kind": "label", "rtc_enabled": False, "candidates": 8,
            "dual_branches": True, "branches": BRANCHES.tolist(),
            "branch_scoring": "q_on_real_goal",
            "teacher_sha256": snapshot["teacher_sha256"], "observation": row["file"],
            "observation_sha256": row["sha256"], "seed": seed,
            "target_space": "raw_pre_smoothing_action_deltas", "tap_equivalence": True,
            "q_recompute_verified": True}
    validate_dual_label(arrays, meta)
    return arrays, meta


def label_dual(args):
    """Offline dual labeler: fresh, non-resumable experimental variant."""
    snapshot = read_json(args.snapshot)
    validate_splits(snapshot["records"], require_validation=False)
    verify_code(snapshot)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    signature = dict(snapshot_sha256=digest(args.snapshot), seed=args.seed,
                     teacher_sha256=snapshot["teacher_sha256"], candidates=8,
                     dual_branches=True, branch_scoring="q_on_real_goal", rtc_enabled=False,
                     pipeline_sha256=source_hashes())
    with writer_lock(root):
        manifest = root / "label_manifest.json"
        if manifest.exists():
            if read_json(manifest) != signature:
                raise ValueError("dual label resume configuration changed; use a new output")
        else:
            if any(root.glob("*/*.npz")) or (root / "COMPLETE.json").exists():
                raise ValueError("unrecognized dual label output")
            atomic_json(manifest, signature)
        teacher, completed, new_count = None, [], 0
        for i, row in enumerate(snapshot["records"]):
            path = root / row["id"]
            receipt = path.with_suffix(".json")
            record_seed = int(hashlib.sha256(
                f'{args.seed}:{row["id"]}:{row["sha256"]}'.encode()).hexdigest()[:8], 16)
            if path.exists():
                checksum = digest(path)
                arrays, meta = load_record(path)
                validate_dual_label(arrays, meta)
                if meta["seed"] != record_seed:
                    raise ValueError("dual label seed mismatch")
            else:
                if teacher is None:
                    teacher = load_teacher(snapshot, "cuda:0")
                arrays, meta = label_dual_one(teacher, row, snapshot, record_seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".pending")
                save_record(temporary, arrays, meta)
                os.replace(temporary, path)
                checksum = digest(path)
                new_count += 1
            atomic_json(receipt, {"sha256": checksum})
            completed.append({**row, "label_sha256": checksum})
            if (i + 1) % 100 == 0 or i == 0 or i + 1 == len(snapshot["records"]):
                print(f'dual-label {i + 1}/{len(snapshot["records"])}: {row["id"]}', flush=True)
        atomic_json(root / "COMPLETE.json", {**signature, "status": "complete", "records": completed})
        print(f"dual labels complete: {new_count} new, {len(completed)} total", flush=True)


def dual_batch(cache, rows):
    """Split one row's 8 candidates into a 4+4 condition-dropout batch.

    Candidates 0-3 are the pointgoal branch (real goal token) and 4-7 the
    nogoal branch (zero goal token). Both branches regress the SAME pointgoal
    teacher deltas: this is condition dropout, not zero-token targets. Works on
    the standard v1 labels, so no dedicated dual labeling is required.
    """
    loaded = [cache.get(row) for row in rows]

    def stack(key, idx=None):
        vals = [a[key] if idx is None else a[key][idx] for a, _ in loaded]
        return torch.from_numpy(np.stack(vals))

    goal = stack("goal_embed")
    return dict(
        pg_deltas=stack("raw_action_deltas", slice(0, 4)),
        ng_deltas=stack("raw_action_deltas", slice(4, 8)),
        pg_goal=goal,
        ng_goal=torch.zeros_like(goal),
        rgbd=stack("rgbd_embed"),
        embodiment=torch.tensor([m["embodiment"] for _, m in loaded]),
        q1=stack("q1", slice(0, 4)), q2=stack("q2", slice(0, 4)))


def pointgoal_weights(q, q_lambda, temperature):
    """Mild Q weighting: w_k = (1-lambda)/K + lambda*softmax(q̃_k/T). q: [B,K]."""
    K = q.shape[-1]
    qn = (q - q.mean(-1, keepdim=True)) / q.std(-1, keepdim=True).clamp_min(1e-6)
    return (1 - q_lambda) / K + q_lambda * torch.softmax(qn / temperature, dim=-1)


def dual_flow_loss(model, data, device, rng, chunk_size, *, beta=0.7, alpha=0.3,
                   q_lambda=0.2, temperature=0.2):
    """Dual-branch CFM: beta*(mild Q-weighted pointgoal) + alpha*(equal nogoal).

    ``data`` holds CPU tensors; only chunked slices move to the GPU. Noise/time
    are drawn up front so changing ``chunk_size`` preserves the same pairs.
    Returns ``(loss, diagnostics)`` with ``loss`` graph-connected for backward.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    pg, ng = data["pg_deltas"], data["ng_deltas"]
    B, K, H, D = pg.shape
    if ng.shape != (B, K, H, D):
        raise ValueError("dual branches must have equal candidate counts")
    if (data["q1"].shape != (B, K) or data["q2"].shape != (B, K)):
        raise ValueError("pointgoal Q must align with the pointgoal candidates")
    n = B * K

    def branch_errors(deltas, goal, rgbd, emb):
        targets = deltas.reshape(n, H, D)
        noise = torch.randn(n, H, D, generator=rng)
        t = torch.rand(n, generator=rng)
        errs = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            owners = torch.arange(start, end) // K
            tg = targets[start:end].to(device)
            no = noise[start:end].to(device)
            tt = t[start:end].to(device)
            x = (1 - tt[:, None, None]) * no + tt[:, None, None] * tg
            pred = model(x, tt, goal[owners].to(device).detach(),
                         rgbd[owners].to(device).detach(), emb[owners].to(device))
            errs.append((pred - (tg - no)).square().mean(dim=(-1, -2)))
        return torch.cat(errs).reshape(B, K)

    pg_err = branch_errors(pg, data["pg_goal"], data["rgbd"], data["embodiment"])
    ng_err = branch_errors(ng, data["ng_goal"], data["rgbd"], data["embodiment"])

    pg_loss = pg_err.mean()
    ng_loss = ng_err.mean()
    if q_lambda > 0:
        q = ((data["q1"] + data["q2"]) / 2).to(device).float()
        w = pointgoal_weights(q, q_lambda, temperature)
        pg_loss = (w * pg_err).sum(dim=1).mean()
    loss = beta * pg_loss + alpha * ng_loss
    return loss, dict(pg=pg_loss.detach(), ng=ng_loss.detach(), total=loss.detach())


def evaluate_dual(model, cache, selected, device, seed, chunk_size, *, beta, alpha,
                  q_lambda, temperature):
    import torch
    from FM_distillation.src.fm_learning import candidate_metrics
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    scenes = {}
    with torch.no_grad():
        for scene, rows in selected.items():
            metrics = []
            for row in rows:
                data = dual_batch(cache, [row])
                loss, parts = dual_flow_loss(model, data, device, rng, chunk_size,
                                             beta=beta, alpha=alpha, q_lambda=q_lambda,
                                             temperature=temperature)
                pg_pred = model.sample(data["pg_goal"].to(device), data["rgbd"].to(device),
                                       data["embodiment"].to(device), candidates=4, steps=4)
                ng_pred = model.sample(data["ng_goal"].to(device), data["rgbd"].to(device),
                                       data["embodiment"].to(device), candidates=4, steps=4)
                pg_metrics = candidate_metrics(pg_pred.cpu(), data["pg_deltas"],
                                               (data["q1"] + data["q2"]).float())
                ng_metrics = candidate_metrics(ng_pred.cpu(), data["ng_deltas"],
                                               torch.zeros_like(data["q1"], dtype=torch.float32))
                metrics.append(dict(
                    fm_loss=float(loss), pg_loss=float(parts["pg"]), ng_loss=float(parts["ng"]),
                    pg_ade_m=pg_metrics["student_to_teacher_nearest_ade_m"],
                    ng_ade_m=ng_metrics["student_to_teacher_nearest_ade_m"]))
            scenes[scene] = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    aggregate = {k: float(np.mean([m[k] for m in scenes.values()])) for k in next(iter(scenes.values()))}
    return dict(per_scene=scenes, scene_balanced=aggregate, steps=4,
                branches=["4_pointgoal", "4_nogoal"],
                observations={s: len(r) for s, r in selected.items()},
                note="dual-branch offline subset; Q is not collision/success evidence; RTC disabled")


def train_dual(args):
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
    del teacher
    model = model.to("cuda:0")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(args.seed)
    selected = validation_rows(groups, args.val_per_scene)
    signature = dict(**identity, source_sha256=source_hashes(),
                     depth=4, width=384, time_scale=9., precision="float32",
                     sampler="dual_branch_4pg_4ng_v1",
                     microbatch=args.microbatch, candidate_microbatch=args.candidate_microbatch,
                     accumulate=args.accumulate, lr=args.lr, seed=args.seed,
                     beta=args.beta, alpha=args.alpha, q_lambda=args.q_lambda,
                     temperature=args.temperature,
                     val_per_scene=args.val_per_scene, eval_every=args.eval_every)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output):
        if any(p.name != ".writer.lock" for p in output.iterdir()):
            raise ValueError("training output not empty; use a new output directory")
        atomic_json(output / "config.json", {**signature, **vars(args)})
        atomic_json(output / "validation_rows.json",
                    {s: [r["id"] for r in rows] for s, rows in selected.items()})
        torch.cuda.reset_peak_memory_stats(0)
        wall = time.perf_counter()
        best = float("inf")
        with (output / "metrics.jsonl").open("x") as log:
            for step in range(1, args.steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running = 0.
                for _ in range(args.accumulate):
                    rows = sample_rows(groups, args.microbatch, rng)
                    data = dual_batch(cache, rows)
                    loss, _ = dual_flow_loss(model, data, "cuda:0", rng, args.candidate_microbatch,
                                             beta=args.beta, alpha=args.alpha,
                                             q_lambda=args.q_lambda, temperature=args.temperature)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite loss")
                    (loss / args.accumulate).backward()
                    running += loss.item() / args.accumulate
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, train_loss=running, grad_norm=float(norm))
                if step % args.eval_every == 0 or step == args.steps:
                    record["validation"] = evaluate_dual(
                        model, cache, selected, "cuda:0", args.seed + 1000, args.candidate_microbatch,
                        beta=args.beta, alpha=args.alpha, q_lambda=args.q_lambda,
                        temperature=args.temperature)
                    val = record["validation"]["scene_balanced"]["fm_loss"]
                    improved = val < best
                    best = min(best, val)
                    checkpoint = output / f"step_{step:08d}.pt"
                    save_checkpoint(checkpoint, dict(student=model.state_dict(), optimizer=optimizer.state_dict(),
                        signature=signature, step=step, best_val_loss=best, batch_rng=rng.get_state(),
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0)))
                    atomic_json(output / "latest.json", {"checkpoint": checkpoint.name, "step": step})
                    if improved:
                        atomic_json(output / "best.json", {"checkpoint": checkpoint.name, "val_loss": val})
                if step == 1 or step % args.log_every == 0 or "validation" in record:
                    record.update(elapsed_s=time.perf_counter() - wall,
                                  peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                                  peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
                    print(json.dumps(record), flush=True)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
        from FM_distillation.src.fm_backbone import export_deploy_checkpoint
        export_deploy_checkpoint(model, teacher_state, teacher_meta, signature, output / "deploy.pt")
        print(f"Deploy checkpoint written: {output / 'deploy.pt'}", flush=True)
