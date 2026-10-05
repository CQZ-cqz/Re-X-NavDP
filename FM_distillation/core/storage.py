"""Immutable completed-scene snapshots and bounded-memory FM label loading.

Kept outside eval/src: adding training tooling must not invalidate live capture.
"""

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from rexnavdp import BASE


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def writer_lock(directory):
    """One writer per output directory; kernel releases lock after interruption."""
    with (Path(directory) / ".writer.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def condition_hashes(hashes):
    return {k: v for k, v in hashes.items()
            if k.startswith(("eval/src/", "third_party/depth_anything/", "../FM_distillation/core/",
                             "../bridge/", "../rl/core/", "../ddim/core/"))}


def verify_code(snapshot):
    for name, expected in snapshot["condition_sha256"].items():
        if digest(BASE / name) != expected:
            raise ValueError(f"teacher/condition source changed: {name}")


def freeze(collection, output, *, completed_only=False):
    """No image decoding or GPU work; hashes come from verified completion indices."""
    root = Path(collection).resolve()
    plan = read_json(root / "collection_manifest.json")
    scenes, records, missing, seen = [], [], [], set()
    conditions = condition_hashes(plan["source_sha256"])
    if not conditions:
        raise ValueError("missing teacher source fingerprints")
    for entry in plan["scenes"]:
        scene, split = entry["scene"], entry["split"]
        if split not in ("train", "validation"):
            continue  # sealed test scenes never enter this pipeline
        if scene in seen:
            raise ValueError(f"duplicate scene / split leakage: {scene}")
        seen.add(scene)
        markers = sorted((root / split / scene).glob("attempt_*/CAPTURE_COMPLETE.json"))
        if not markers:
            missing.append(f"{split}/{scene}")
            continue
        if len(markers) != 1:
            raise ValueError(f"ambiguous completed attempts: {scene}")
        run = markers[0].parent
        report, meta = read_json(markers[0]), read_json(run / "manifest.json")
        for obj in (report, meta):
            if (obj["scene"], obj["split"]) != (scene, split):
                raise ValueError("scene provenance mismatch")
        if meta["teacher_sha256"] != plan["teacher_sha256"] or condition_hashes(meta["code_sha256"]) != conditions:
            raise ValueError("teacher/condition mismatch between completed scenes")
        index = run / "observation_index.jsonl"
        if digest(index) != report["index_sha256"] or digest(run / report["metric_file"]) != report["metric_sha256"]:
            raise ValueError("completed scene index/metrics changed")
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        if len(rows) != report["observations"] or report["episodes"] != entry["episodes"]:
            raise ValueError("incomplete scene")
        unique = set()
        for row in rows:
            name = row["file"]
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError("invalid observation name")
            key = (row["episode_id"], row["step"])
            if key in unique:
                raise ValueError("duplicate episode/step")
            unique.add(key)
            records.append({**row, "scene": scene, "split": split,
                            "run_id": meta["run_id"], "source": str(run / "observations" / name),
                            "id": f"{split}/{scene}/{name}"})
        if len({r["episode_id"] for r in rows}) != report["episodes"]:
            raise ValueError("episode coverage mismatch")
        scenes.append(dict(scene=scene, split=split, count=len(rows), run=str(run),
                           index_sha256=report["index_sha256"]))
    if missing and not completed_only:
        raise ValueError(f"scenes still collecting: {missing}; use --completed-only for a versioned partial snapshot")
    if not records:
        raise ValueError("no completed observations")
    result = dict(schema="fm_training_snapshot_v1", collection=str(root),
                  collection_sha256=digest(root / "collection_manifest.json"),
                  checkpoint=plan["checkpoint"], teacher_sha256=plan["teacher_sha256"],
                  condition_sha256=conditions, scenes=scenes, skipped_incomplete=missing, records=records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publication: snapshots are never refreshed under the same name.
    with output.open("x") as f:
        json.dump(result, f, allow_nan=False)
    return result


def validate_splits(records, require_validation=True):
    groups = defaultdict(list)
    ownership, ids = {}, set()
    for row in records:
        scene, split = row["scene"], row["split"]
        if split not in ("train", "validation") or ownership.setdefault(scene, split) != split:
            raise ValueError("test data or scene leakage")
        if row["id"] in ids:
            raise ValueError("duplicate observation")
        ids.add(row["id"])
        groups[(split, scene)].append(row)
    if not any(k[0] == "train" for k in groups):
        raise ValueError("no training scenes")
    if require_validation and not any(k[0] == "validation" for k in groups):
        raise ValueError("independent validation scenes not ready; do not use training frames as validation")
    return dict(groups)


class LabelCache:
    """Lazy NPZ reader: caches at most capacity observations, not the full corpus."""
    def __init__(self, root, snapshot, capacity=128):
        self.root, self.snapshot = Path(root), snapshot
        self.capacity, self.cache = capacity, OrderedDict()

    def get(self, row):
        from FM_distillation.core.fm_data import load_record, validate_label
        key = row["id"]
        if key not in self.cache:
            path = self.root / key
            if digest(path) != row["label_sha256"]:
                raise ValueError(f"label checksum mismatch: {key}")
            arrays, meta = load_record(path)
            validate_label(arrays, meta)
            for field in ("scene", "split", "episode_id", "step", "run_id"):
                if meta[field] != row[field]:
                    raise ValueError(f"label provenance mismatch: {field}")
            if (meta["teacher_sha256"] != self.snapshot["teacher_sha256"] or
                    meta["observation_sha256"] != row["sha256"]):
                raise ValueError("label teacher/source mismatch")
            self.cache[key] = (arrays, meta)
        self.cache.move_to_end(key)
        result = self.cache[key]
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


def sample_rows(groups, count, generator):
    """Uniform scene, then uniform observation; candidate selection is separate."""
    import torch
    scenes = sorted(k for k in groups if k[0] == "train")
    selected = []
    for _ in range(count):
        key = scenes[torch.randint(len(scenes), (), generator=generator).item()]
        pool = groups[key]
        selected.append(pool[torch.randint(len(pool), (), generator=generator).item()])
    return selected
