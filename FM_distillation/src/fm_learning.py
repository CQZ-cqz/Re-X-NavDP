"""Cached teacher dataset and diagnostic metrics for P2, not navigation eval."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import json
from pathlib import Path

import numpy as np
import torch

from .fm_data import load_record, sha256, validate_label, validate_observation


def load_cache(run, name="labels_smoke"):
    run = Path(run)
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest["split"] != "train":
        raise ValueError("P2 learning smoke accepts train data only; never test/validation/debug")
    root = run/name
    completed = json.loads((root/"COMPLETE.json").read_text())
    label_manifest = json.loads((root/"label_manifest.json").read_text())
    paths = sorted(root.glob("*.npz"))
    if completed["status"] != "complete" or len(paths) != completed["count"] or not paths:
        raise ValueError("incomplete dataset")
    names = {p.name for p in paths}
    if names != set(completed["sha256"]) or names != set(label_manifest["sources"]):
        raise ValueError("dataset file set changed")
    if label_manifest["teacher_sha256"] != manifest["teacher_sha256"]:
        raise ValueError("teacher identity mismatch")
    rows, metadata = [], []
    for path in paths:
        if sha256(path) != completed["sha256"][path.name]:
            raise ValueError(f"label checksum mismatch: {path.name}")
        row, meta = load_record(path)
        validate_label(row, meta)
        if Path(meta["observation"]).name != meta["observation"]:
            raise ValueError("invalid source filename")
        source = run/"observations"/meta["observation"]
        digest = sha256(source)
        if digest != meta["observation_sha256"] or digest != label_manifest["sources"][source.name]:
            raise ValueError("source checksum mismatch")
        obs, src = load_record(source)
        validate_observation(obs, src)
        for key in ("run_id", "scene", "split"):
            if meta[key] != manifest[key] or src[key] != manifest[key]:
                raise ValueError("data provenance mismatch")
        for key in ("episode_id", "step", "embodiment", "stuck"):
            if meta[key] != src[key]:
                raise ValueError(f"label/source mismatch: {key}")
        if meta["teacher_sha256"] != manifest["teacher_sha256"]:
            raise ValueError("label teacher mismatch")
        rows.append(row)
        metadata.append(meta)
    def tensor(key):
        return torch.from_numpy(np.stack([r[key] for r in rows])).float()
    data = dict(actions=tensor("raw_action_deltas"), goal=tensor("goal_embed"),
                rgbd=tensor("rgbd_embed"), scores=tensor("scores"),
                embodiment=torch.tensor([m["embodiment"] for m in metadata]))
    return data, metadata, manifest, completed


def select_probe_observations(metadata, limit=4):
    """Deterministic balanced debug subset, NOT an independent validation split."""
    normal = [i for i,m in enumerate(metadata) if not m["stuck"]]
    stuck = [i for i,m in enumerate(metadata) if m["stuck"]]
    selected = []
    while len(selected) < min(limit, len(metadata)):
        for pool in (normal, stuck):
            if pool and len(selected) < limit:
                selected.append(pool.pop(0))
    return torch.tensor(selected, dtype=torch.long)


def batch_rows(data, obs_indices, candidate_indices, device):
    return dict(action_deltas=data["actions"][obs_indices, candidate_indices].to(device),
                goal_embed=data["goal"][obs_indices].to(device),
                rgbd_embed=data["rgbd"][obs_indices].to(device),
                embodiment=data["embodiment"][obs_indices].to(device))


@torch.no_grad()
def candidate_metrics(predicted, targets, scores):
    """Unsmoothed cumulative XY diagnostics, not Q evaluation/collision checks.

    Nearest-neighbor distances do not assume correspondence of candidate indices
    or teacher and student noise. All units here refer to path coordinates.
    """
    if not torch.isfinite(predicted).all():
        raise ValueError("nonfinite free sample")
    p, q = (x.cumsum(-2)/4 for x in (predicted, targets))
    distances = (p[:,:,None,:,:2]-q[:,None,:,:,:2]).norm(dim=-1).mean(-1)
    def diversity(path):
        k = path.shape[1]
        matrix = (path[:,:,None,:,:2]-path[:,None,:,:,:2]).norm(dim=-1).mean(-1)
        return matrix[:,torch.triu(torch.ones(k,k,dtype=torch.bool,device=path.device), diagonal=1)].mean()
    top = scores.topk(2, dim=1).indices
    coverage = distances.min(dim=1).values
    return dict(student_to_teacher_nearest_ade_m=distances.min(dim=2).values.mean().item(),
                teacher_to_student_nearest_ade_m=coverage.mean().item(),
                teacher_top2_coverage_ade_m=coverage.gather(1,top).mean().item(),
                student_pairwise_ade_m=diversity(p).item(), teacher_pairwise_ade_m=diversity(q).item(),
                student_raw_abs_max=predicted.abs().max().item(),
                teacher_raw_abs_max=targets.abs().max().item(),
                student_backward_fraction=(p[:,:,5,0]<-.05).float().mean().item(),
                teacher_backward_fraction=(q[:,:,5,0]<-.05).float().mean().item())
