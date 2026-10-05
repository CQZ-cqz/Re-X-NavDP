"""FM v1 collection contract and non-mutating teacher taps (no simulator import)."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "x_navdp_fm_v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_record(path, arrays, metadata):
    """Exclusive creation: never overwrite user samples. No object arrays/pickle."""
    path = Path(path)
    encoded = json.dumps(metadata, default=json_default, allow_nan=False)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays, metadata_json=np.array(encoded))


def load_record(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: data[key].copy() for key in data.files if key != "metadata_json"}
    return arrays, metadata


@contextmanager
def teacher_tap(model):
    """Capture exact raw actions and Q tensors; restore methods even on error.

    Use in single-threaded offline replay only. No RNG or parameter changes.
    Tensor clones are transferred to CPU after the original calls complete.
    """
    captured, saved = {}, []

    def patch(name, factory):
        saved.append((name, name in model.__dict__, model.__dict__.get(name)))
        setattr(model, name, factory(getattr(model, name)))

    def smoothing(original):
        def run(points, *args, **kwargs):
            if "raw" in captured:
                raise RuntimeError("expected exactly one action smoothing call")
            captured["raw"] = points.detach().clone()
            result = original(points, *args, **kwargs)
            captured["smoothed_actions"] = result.detach().clone()
            return result
        return run

    def scoring(original):
        def run(path, rgbd, goal, *args, **kwargs):
            captured["q_path"] = path.detach().clone()
            captured["rgbd"] = rgbd.detach().clone()
            captured["goal"] = goal.detach().clone()
            result = original(path, rgbd, goal, *args, **kwargs)
            captured["q1"], captured["q2"] = [x.detach().clone() for x in result]
            return result
        return run

    try:
        patch("smooth_trajectory", smoothing)
        patch("predict_pointgoal_q", scoring)
        yield captured
    finally:
        for name, existed, original in reversed(saved):
            if existed:
                setattr(model, name, original)
            else:
                delattr(model, name)


def validate_observation(arrays, meta):
    if meta.get("schema") != SCHEMA or meta.get("kind") != "observation":
        raise ValueError("not a fresh FM observation; legacy dumps need explicit migration")
    shapes = {"rgb": (8, 224, 224, 3), "depth": (224, 224, 1), "pointgoal": (3,),
              "prev_action": (24, 3), "robot_pos": (3,), "robot_quat": (4,),
              "rgb_history_times": (8,), "rgb_history_steps": (8,)}
    for key, shape in shapes.items():
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key}; expected finite {shape}")
    if not 0 <= int(meta["valid_segment_len"]) <= 24:
        raise ValueError("invalid RTC prefix length")
    if meta["embodiment"] not in (0, 1, 2):
        raise ValueError("invalid embodiment")
    if not meta.get("episode_id") or not meta.get("scene") or not meta.get("run_id"):
        raise ValueError("missing episode/scene/run identity")
    if meta.get("split") not in ("train", "validation", "test", "debug"):
        raise ValueError("missing explicit split")
    if not np.isfinite(meta["sim_time"]):
        raise ValueError("missing simulation timestamp")
    if arrays["rgb"].min() < 0 or arrays["rgb"].max() > 1:
        raise ValueError("RGB must be preprocessed [0,1]")
    if arrays["depth"].min() < 0 or arrays["depth"].max() > 5:
        raise ValueError("Depth differs from baseline 0.1--5m filter (zero is invalid depth)")
    if np.linalg.norm(arrays["robot_quat"]) < 1e-8:
        raise ValueError("invalid zero quaternion")
    times, steps = arrays["rgb_history_times"], arrays["rgb_history_steps"]
    if times[-1] != meta["sim_time"] or steps[-1] != meta["step"]:
        raise ValueError("current-frame timestamp/step mismatch")
    if ((times > meta["sim_time"]) | (times < -1)).any() or ((steps > meta["step"]) | (steps < -1)).any():
        raise ValueError("history contains future or invalid frame")
    if not np.array_equal(times == -1, steps == -1):
        raise ValueError("history padding mismatch")
    guidance = np.asarray(meta["guidance_factor"])
    if guidance.shape != (8,) or not np.isfinite(guidance).all():
        raise ValueError("invalid guidance factors")


def validate_label(arrays, meta):
    if meta.get("schema") != SCHEMA or meta.get("kind") != "label" or meta.get("rtc_enabled") is not False:
        raise ValueError("expected RTC-off FM label")
    count = meta["candidates"]
    shapes = {"raw_action_deltas": (count, 24, 3), "smoothed_actions": (count, 24, 3),
              "q_path": (count, 24, 3), "trajectories": (count, 24, 3),
              "q1": (count,), "q2": (count,), "scores": (count,),
              "top_indices": (2,), "top_trajectories": (2, 24, 3),
              "rgbd_embed": (128, 384), "goal_embed": (1, 384),
              "initial_noise": (count, 24, 3)}
    for key, shape in shapes.items():
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid label {key}; expected finite {shape}")
    np.testing.assert_allclose(arrays["scores"], (arrays["q1"]+arrays["q2"])/2, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(arrays["trajectories"], np.cumsum(arrays["smoothed_actions"]/4, axis=1), atol=1e-5, rtol=1e-5)
    idx = arrays["top_indices"]
    if idx.dtype.kind not in "iu" or len(set(idx.tolist())) != 2 or (idx < 0).any() or (idx >= count).any():
        raise ValueError("invalid top-2 indices")
    np.testing.assert_allclose(arrays["scores"][idx], np.sort(arrays["scores"])[::-1][:2], atol=1e-6)
    np.testing.assert_allclose(arrays["top_trajectories"], arrays["trajectories"][idx], atol=1e-6)
    if not meta.get("teacher_sha256") or not meta.get("observation_sha256"):
        raise ValueError("missing provenance")
    if meta.get("target_space") != "raw_pre_smoothing_action_deltas":
        raise ValueError("wrong target coordinate space")
    if meta.get("tap_equivalence") is not True or meta.get("q_recompute_verified") is not True:
        raise ValueError("label lacks teacher verification")
