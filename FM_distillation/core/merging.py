"""Consolidated FM_distillation library module (entry points live in cli.py)."""

from rexnavdp import BASE, ROOT

"""CPU-only verified index merge of completed train/validation labels.

Links immutable labels; never regenerates labels or copies the image dataset.
"""

import argparse
import json
from pathlib import Path
import sys

from FM_distillation.core.storage import atomic_json, digest, read_json, validate_splits, writer_lock


def inspect_inputs(train_snapshot, train_labels, val_snapshot, val_labels):
    snapshots, completions, roots = [], [], []
    for split, snapshot_path, label_path in (("train", train_snapshot, train_labels),
                                            ("validation", val_snapshot, val_labels)):
        snapshot, complete = read_json(snapshot_path), read_json(Path(label_path)/"COMPLETE.json")
        if (complete["status"] != "complete" or complete["snapshot_sha256"] != digest(snapshot_path)
                or complete["teacher_sha256"] != snapshot["teacher_sha256"]):
            raise ValueError(f"incomplete/mismatched {split} labels")
        records = [{k:v for k,v in r.items() if k != "label_sha256"} for r in complete["records"]]
        if records != snapshot["records"] or not records or any(r["split"] != split for r in records):
            raise ValueError(f"invalid {split} observation set")
        if any(s["split"] != split for s in snapshot["scenes"]):
            raise ValueError("scene split mismatch")
        for row in records:
            identifier = Path(row["id"])
            if identifier.is_absolute() or ".." in identifier.parts:
                raise ValueError("unsafe label identifier")
        snapshots.append(snapshot)
        completions.append(complete)
        roots.append(Path(label_path).resolve())
    for key in ("teacher_sha256", "condition_sha256", "collection_sha256"):
        if snapshots[0][key] != snapshots[1][key]:
            raise ValueError(f"snapshot mismatch: {key}")
    for key in ("seed", "teacher_sha256", "candidates", "rtc_enabled", "pipeline_sha256"):
        if completions[0][key] != completions[1][key]:
            raise ValueError(f"label generation mismatch: {key}")
    rows = completions[0]["records"] + completions[1]["records"]
    groups = validate_splits(rows)
    snapshot = {**snapshots[0], "records": snapshots[0]["records"]+snapshots[1]["records"],
                "scenes": snapshots[0]["scenes"]+snapshots[1]["scenes"],
                "skipped_incomplete": sorted(set(s for snap in snapshots for s in snap["skipped_incomplete"]
                    if s not in {f'{r["split"]}/{r["scene"]}' for r in rows}))}
    sources = [(roots[i]/row["id"], row) for i in (0,1) for row in completions[i]["records"]]
    report = dict(train_observations=len(completions[0]["records"]),
                  validation_observations=len(completions[1]["records"]),
                  scenes={f"{split}/{scene}":len(items) for (split,scene),items in groups.items()})
    return snapshot, completions[0], sources, report


def merge(args):
    snapshot, original, sources, report = inspect_inputs(args.train_snapshot,args.train_labels,
                                                         args.val_snapshot,args.val_labels)
    print(json.dumps(report, indent=2), flush=True)
    if args.inspect:
        print("Metadata inspection only. No output files or GPU jobs created.")
        return
    root = Path(args.output).resolve()
    for source_root in (Path(args.train_labels).resolve(), Path(args.val_labels).resolve()):
        if root == source_root or source_root in root.parents or root in source_root.parents:
            raise ValueError("output must be separate from input label trees")
    root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root):
        labels = root/"labels"
        labels.mkdir(exist_ok=True)
        identity = {name: dict(path=str(Path(getattr(args,name)).resolve()),
                               sha256=digest(Path(getattr(args,name))/"COMPLETE.json") if name.endswith("labels")
                               else digest(getattr(args,name)))
                    for name in ("train_snapshot","train_labels","val_snapshot","val_labels")}
        if (root/"merge_sources.json").exists():
            if read_json(root/"merge_sources.json") != identity:
                raise ValueError("merge inputs changed; use a fresh output")
        else:
            if any(labels.iterdir()) or (root/"snapshot.json").exists():
                raise ValueError("unrecognized merge output")
            atomic_json(root/"merge_sources.json", identity)
        for i, (source,row) in enumerate(sources):
            if digest(source) != row["label_sha256"]:
                raise ValueError(f"label checksum mismatch: {source}")
            target = labels/row["id"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError("existing link points at a different label")
            elif target.exists():
                raise ValueError("refusing to replace existing file")
            else:
                target.symlink_to(source)
            if (i+1) % 5000 == 0 or i+1 == len(sources):
                print(f"verified and linked {i+1}/{len(sources)}", flush=True)
        atomic_json(root/"snapshot.json", snapshot)
        signature = {k:original[k] for k in ("seed","teacher_sha256","candidates","rtc_enabled","pipeline_sha256")}
        signature["snapshot_sha256"] = digest(root/"snapshot.json")
        atomic_json(labels/"label_manifest.json", signature)
        atomic_json(labels/"COMPLETE.json", {**signature,"status":"complete","records":[row for _,row in sources]})
        atomic_json(root/"merge_report.json", report)
        print(f"Ready: --snapshot {root/'snapshot.json'} --labels {labels}", flush=True)


"""Merge legacy+joint FM labels, filtering FAILED TRAIN episodes only.

Immutable manifest-only view. Never modifies, copies, or deletes source NPZs.
Validation retains ALL episodes. Requires the mixed reader, not legacy LabelCache.
"""

import argparse
import csv
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from FM_distillation.core.storage import (atomic_json, condition_hashes, digest, read_json,
                                     validate_splits, writer_lock, verify_code)


def episode_outcomes(rows, metrics):
    """Join by sample_idx, NEVER by episode_id suffix (capture suffix is 1-based)."""
    indexed = {}
    for r in metrics:
        key = int(r['episode_idx'])
        value = float(r['success'])
        if key in indexed or value not in (0.,1.):
            raise ValueError('duplicate/invalid metric outcome')
        indexed[key] = value
    episodes, samples, steps = {}, {}, set()
    for row in rows:
        episode, sample = row['episode_id'], int(row['sample_idx'])
        if sample not in indexed or float(row['outcome']['success']) != indexed[sample]:
            raise ValueError('observation outcome differs from metric')
        if int(row['outcome']['episode_idx']) != sample:
            raise ValueError('outcome episode_idx differs from sample_idx')
        if episodes.setdefault(episode,sample) != sample or samples.setdefault(sample,episode) != episode:
            raise ValueError('episode/sample mapping is not one-to-one')
        if (episode,row['step']) in steps:
            raise ValueError('duplicate episode step')
        steps.add((episode,row['step']))
    if set(samples) != set(indexed):
        raise ValueError('missing observations for completed metric episodes')
    return {episode:indexed[sample] for episode,sample in episodes.items()}


def validate_mixed_record(row, teacher_hash):
    from FM_distillation.core.fm_data import load_record, validate_label
    path = Path(row['label_path'])
    if digest(path) != row['label_sha256']:
        raise ValueError(f'label hash mismatch: {path}')
    arrays,meta = load_record(path)
    if row['label_format'] == 'joint_v1':
        from FM_distillation.core.labeling import validate_joint_label
        validate_joint_label(arrays,meta)
    elif row['label_format'] == 'legacy_v1':
        if 'joint_label_version' in meta:
            raise ValueError('joint label misrepresented as legacy')
        validate_label(arrays,meta)
    else:
        raise ValueError('unknown label format')
    for key in ('scene','split','episode_id','step','run_id'):
        if meta[key] != row[key]:
            raise ValueError(f'label provenance mismatch: {key}')
    if meta['teacher_sha256'] != teacher_hash or meta['observation_sha256'] != row['sha256']:
        raise ValueError('teacher/observation mismatch')
    if int(meta['sample_idx']) != int(row['sample_idx']):
        raise ValueError('label sample_idx mismatch')
    return arrays,meta


class MixedLabelCache:
    """Absolute immutable label references; honest joint structural/full audit flags."""
    def __init__(self,snapshot,capacity=128):
        self.snapshot,self.capacity,self.cache = snapshot,capacity,OrderedDict()
    def get(self,row):
        key = row['id']
        if key not in self.cache:
            self.cache[key] = validate_mixed_record(row,self.snapshot['teacher_sha256'])
        self.cache.move_to_end(key)
        result = self.cache[key]
        while len(self.cache)>self.capacity:
            self.cache.popitem(last=False)
        return result


def load_mixed_dataset(root):
    root = Path(root)
    complete = read_json(root/'COMPLETE.json')
    if complete['status'] != 'complete' or digest(root/'snapshot.json') != complete['snapshot_sha256']:
        raise ValueError('mixed dataset incomplete/changed')
    snap = read_json(root/'snapshot.json')
    if snap['schema'] != 'fm_mixed_success_snapshot_v1':
        raise ValueError('unexpected mixed schema')
    verify_code(snap)
    return snap,validate_splits(snap['records']),MixedLabelCache(snap)


def inventory(legacy,joint):
    legacy,joint = Path(legacy).resolve(),Path(joint).resolve()
    snap = read_json(legacy/'snapshot.json')
    complete = read_json(legacy/'labels/COMPLETE.json')
    if complete['status']!='complete' or complete['snapshot_sha256']!=digest(legacy/'snapshot.json'):
        raise ValueError('legacy labels incomplete/mismatched')
    if [{k:v for k,v in r.items() if k!='label_sha256'} for r in complete['records']] != snap['records']:
        raise ValueError('legacy record set mismatch')
    if complete['teacher_sha256']!=snap['teacher_sha256'] or complete['rtc_enabled'] or complete['candidates']!=8:
        raise ValueError('legacy generation mismatch')
    plan = read_json(joint/'collection_manifest.json')
    if plan['teacher_sha256']!=snap['teacher_sha256'] or condition_hashes(plan['source_sha256'])!=snap['condition_sha256']:
        raise ValueError('joint and legacy teacher/conditions mismatch')
    verify_code(snap)
    pins = {}
    def pin(path):
        pins[str(Path(path).resolve())] = digest(path)
    for path in (legacy/'snapshot.json',legacy/'labels/COMPLETE.json',joint/'collection_manifest.json'):
        pin(path)
    sources = []
    legacy_by_scene = defaultdict(list)
    for row in complete['records']:
        legacy_by_scene[(row['split'],row['scene'])].append(row)
    for scene in snap['scenes']:
        domain = 'clutter_easy' if scene['scene'].startswith('easy_') else 'clutter_hard'
        rows = [{**r,'domain':domain,'label_format':'legacy_v1',
                 'label_path':str((legacy/'labels'/r['id']).resolve())}
                for r in legacy_by_scene[(scene['split'],scene['scene'])]]
        sources.append((Path(scene['run']),rows,domain))
    skipped = []
    for entry in plan['scenes']:
        if entry['split']!='train':
            raise ValueError('joint collection must contain train scenes only')
        markers = list((joint/'train'/entry['scene']).glob('attempt_*/CAPTURE_COMPLETE.json'))
        if not markers:
            skipped.append(entry['scene'])
            continue
        if len(markers)!=1:
            raise ValueError('ambiguous completed joint attempts')
        run = markers[0].parent
        report,meta,labels = (read_json(run/name) for name in ('CAPTURE_COMPLETE.json','manifest.json','labels/COMPLETE.json'))
        if digest(run/'labels/COMPLETE.json')!=report['label_complete_sha256'] or labels['status']!='complete':
            raise ValueError('joint completion mismatch')
        if labels['teacher_sha256']!=snap['teacher_sha256'] or labels['joint_label_version']!=1 or labels['full_audits']<1:
            raise ValueError('joint audit/teacher contract mismatch')
        index = [json.loads(l) for l in (run/'observation_index.jsonl').read_text().splitlines()]
        if len(index)!=labels['count'] or set(labels['sha256'])!={r['file'] for r in index}:
            raise ValueError('joint labels do not cover observation index')
        rows = []
        for r in index:
            if Path(r['file']).name!=r['file'] or not r['file'].endswith('.npz'):
                raise ValueError('unsafe observation filename')
            rows.append({**r,'scene':entry['scene'],'split':'train','run_id':meta['run_id'],
                'source':str(run/'observations'/r['file']), 'id':f"train/{entry['scene']}/{r['file']}",
                'label_path':str(run/'labels'/r['file']),'label_sha256':labels['sha256'][r['file']],
                'label_format':'joint_v1','domain':entry['family']})
        sources.append((run,rows,entry['family']))
        pin(run/'labels/COMPLETE.json')
    kept,excluded,scene_reports = [],[],[]
    for run,rows,domain in sources:
        meta,report = (read_json(run/name) for name in ('manifest.json','CAPTURE_COMPLETE.json'))
        index,metric = run/'observation_index.jsonl',run/report['metric_file']
        if digest(index)!=report['index_sha256'] or digest(metric)!=report['metric_sha256']:
            raise ValueError('source index/metric checksum mismatch')
        if meta['teacher_sha256']!=snap['teacher_sha256'] or condition_hashes(meta['code_sha256'])!=snap['condition_sha256']:
            raise ValueError('source teacher/conditions mismatch')
        if len(rows)!=report['observations'] or any(r['run_id']!=meta['run_id'] or r['scene']!=meta['scene'] or r['split']!=meta['split'] for r in rows):
            raise ValueError('scene/record contract mismatch')
        for f in (run/'manifest.json',run/'CAPTURE_COMPLETE.json',index,metric):
            pin(f)
        with metric.open() as stream:
            outcomes = episode_outcomes(rows,list(csv.DictReader(stream)))
        if len(outcomes)!=report['episodes']:
            raise ValueError('episode count mismatch')
        selected = [r for r in rows if r['split']=='validation' or outcomes[r['episode_id']]==1]
        removed = [r for r in rows if r['split']=='train' and outcomes[r['episode_id']]==0]
        kept.extend(selected)
        excluded.extend(removed)
        scene_reports.append(dict(scene=meta['scene'],split=meta['split'],domain=domain,run=str(run),
            episodes=len(outcomes),success_episodes=int(sum(outcomes.values())),
            observations_before=len(rows),observations_kept=len(selected),observations_excluded=len(removed),
            stuck_kept=sum(bool(r['stuck']) for r in selected)))
    validate_splits(kept)
    report = dict(per_scene=scene_reports,skipped_incomplete_joint_scenes=skipped,
        train_kept=sum(r['split']=='train' for r in kept),validation_kept=sum(r['split']=='validation' for r in kept),
        excluded_train_observations=len(excluded),
        by_domain=dict(Counter(r['domain'] for r in kept if r['split']=='train')),
        policy='filter failed TRAIN episodes by recorded success==1; validation untouched; no raw deletion')
    result = dict(schema='fm_mixed_success_snapshot_v1',checkpoint=snap['checkpoint'],
        teacher_sha256=snap['teacher_sha256'],condition_sha256=snap['condition_sha256'],
        scenes=scene_reports,records=kept,source_pins=pins,policy=report['policy'])
    return result,excluded,report

