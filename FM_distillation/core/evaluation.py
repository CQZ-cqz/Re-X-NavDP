"""Consolidated FM_distillation library module (entry points live in cli.py)."""

from rexnavdp import BASE, ROOT

"""FM + frozen original Q + baseline recovery/MPC, no RTC, no recording.

Independent entry: leaves collection/training code and running GPU1 unchanged.
FM RTC is intentionally rejected until its flow-specific guidance is defined.
"""

import argparse
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
import types

from FM_distillation.core.storage import atomic_json, digest, read_json, writer_lock
from FM_distillation.core.capture import stop_process, completed_metrics
from FM_distillation.core import dataset as fm_dataset


def fm_agent_class(student_path, teacher_sha256):
    import torch
    from eval.src.policy_agent import NavDP_Agent
    from FM_distillation.core.flow_generator import CompactFlowGenerator, generate_and_rank
    class FMAgent(NavDP_Agent):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            teacher = self.navi_former
            if teacher.rtc_enabled:
                raise ValueError("FM RTC is not implemented; use --rtc off")
            state = torch.load(student_path,map_location="cpu",weights_only=True)
            if state["signature"]["teacher_sha256"] != teacher_sha256:
                raise ValueError("student/teacher identity mismatch")
            student = CompactFlowGenerator(teacher,depth=4).to(self.device)
            student.load_state_dict(state["student"],strict=True)
            teacher.eval().requires_grad_(False)
            student.eval().requires_grad_(False)
            self.fm_student = student
            @torch.no_grad()
            def predict(network,goal_point,input_images,input_depths,sample_num=8,**kwargs):
                if network.rtc_enabled or sample_num != 8:
                    raise ValueError("expected RTC-off, eight candidates")
                goal = network.point_encoder(torch.as_tensor(goal_point,dtype=torch.float32,device=network.device)).unsqueeze(1)
                rgbd = network.rgbd_encoder(input_images,input_depths)
                result = generate_and_rank(network,student,goal,rgbd,kwargs.get("embodiment",0),
                                           steps=4,candidates=8)
                return tuple(result[k].cpu().numpy() for k in ("trajectories","scores","top_trajectories"))+(None,)
            teacher.predict_pointgoal_action_with_guidance = types.MethodType(predict,teacher)
    return FMAgent


def serve_closed_loop(args):
    fm_dataset.select_physical_gpu(args.physical_gpu)
    run = Path(args.run)
    meta = fm_dataset.check_frozen(run)
    settings = read_json(run/"fm_settings.json")
    if digest(__file__) != settings["runner_sha256"]:
        raise ValueError("FM runner changed")
    if meta["physical_gpu"] != args.physical_gpu:
        raise ValueError("GPU mismatch")
    from eval.src import policy_server
    if settings.get("deploy"):
        if digest(settings["deploy"]) != settings["deploy_sha256"]:
            raise ValueError("deploy checkpoint changed")
        import torch
        state = torch.load(settings["deploy"], map_location="cpu", weights_only=True, mmap=True)
        if state["signature"]["teacher_sha256"] != meta["teacher_sha256"]:
            raise ValueError("deploy/teacher identity mismatch")
        from FM_distillation.core.fm_backbone import fm_deploy_agent_class
        policy_server.NavDP_Agent = fm_deploy_agent_class(settings["deploy"], rtc_enabled=False)
    else:
        if digest(settings["student"]) != settings["student_sha256"]:
            raise ValueError("student changed")
        policy_server.NavDP_Agent = fm_agent_class(settings["student"],meta["teacher_sha256"])
    policy_server.init_app("humanoid",no_visualization=True,device="cuda:0",checkpoint=meta["checkpoint"],
                           seed=meta["seed"],recovery="baseline",rtc_enabled=False)
    policy_server.run_server(port=args.port)


"""FM trajectory RTC evaluation, 4 steps/8 candidates, no video; separate from old runner."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from FM_distillation.core import dataset as fm_dataset
from FM_distillation.core.storage import atomic_json, digest, read_json, writer_lock
from FM_distillation.core.capture import stop_process, completed_metrics


def metric_summary(rows):
    ne = [float(r["ne"]) for r in rows]
    if not rows or any(not math.isfinite(x) or x < 0 for x in ne):
        raise ValueError("missing or invalid terminal NE")
    def subset(success):
        values = [float(r["ne"]) for r in rows if (float(r["success"]) > .5) == success]
        return statistics.mean(values) if values else None
    return dict(episodes=len(rows),success_rate=statistics.mean(float(r["success"]) for r in rows),
                ne_mean_m=statistics.mean(ne),ne_median_m=statistics.median(ne),
                ne_success_mean_m=subset(True),ne_failure_mean_m=subset(False),
                spl=statistics.mean(float(r["spl"]) for r in rows))


def read_completed(root,scene):
    markers = list((Path(root)/scene).glob("attempt_*/EVAL_COMPLETE.json"))
    if len(markers) != 1:
        raise ValueError(f"need exactly one completed attempt: {root}/{scene}")
    marker = read_json(markers[0])
    path = Path(marker["metric_file"])
    if digest(path) != marker["metric_sha256"]:
        raise ValueError("completed metric checksum changed")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    ids = [int(r["episode_idx"]) for r in rows]
    if len(ids) != marker["episodes"] or len(set(ids)) != len(ids):
        raise ValueError("incomplete/duplicate metric episode IDs")
    return {int(r["episode_idx"]):r for r in rows}


def compare_roots(off,on):
    a,b = (read_json(Path(root)/"evaluation_manifest.json") for root in (off,on))
    source_keys = [k for k in ("student_sha256","deploy_sha256") if k in a and k in b]
    if len(source_keys) != 1:
        raise ValueError("evaluations must agree on exactly one student/deploy source")
    for key in (*source_keys,"teacher_sha256","episodes","seed","code_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"unmatched evaluation setting: {key}")
    if a["rtc"] or not b["rtc"]:
        raise ValueError("expected RTC-off baseline and RTC-on experiment")
    result = dict(ne_definition="XY goal distance at pre-terminal simulation step, metres; lower is better",
                  off=str(off),on=str(on),per_scene={},
                  gpu_off=a["physical_gpu"],gpu_on=b["physical_gpu"])
    all_off,all_on = [],[]
    for scene in ("easy_5","hard_5"):
        left,right = read_completed(off,scene),read_completed(on,scene)
        if set(left) != set(right) or set(left) != set(range(a["episodes"])):
            raise ValueError("episode IDs differ from requested paired evaluation")
        x,y = metric_summary(list(left.values())),metric_summary(list(right.values()))
        result["per_scene"][scene] = dict(off=x,on=y,
            paired_ne_delta_m=statistics.mean(float(right[i]["ne"])-float(left[i]["ne"]) for i in left),
            success_delta=y["success_rate"]-x["success_rate"])
        all_off.extend(left.values())
        all_on.extend(right.values())
    result["overall"] = dict(off=metric_summary(all_off),on=metric_summary(all_on))
    return result


def serve_rtc(args):
    fm_dataset.select_physical_gpu(args.physical_gpu)
    run = Path(args.run)
    meta = fm_dataset.check_frozen(run)
    settings = read_json(run/"fm_settings.json")
    for name,checksum in settings["rtc_source_sha256"].items():
        if digest(BASE/name) != checksum:
            raise ValueError(f"RTC source changed: {name}")
    if meta["physical_gpu"] != args.physical_gpu:
        raise ValueError("GPU mismatch")
    from eval.src import policy_server
    if settings.get("deploy"):
        if digest(settings["deploy"]) != settings["deploy_sha256"]:
            raise ValueError("deploy checkpoint changed")
        import torch
        state = torch.load(settings["deploy"], map_location="cpu", weights_only=True, mmap=True)
        if state["signature"]["teacher_sha256"] != meta["teacher_sha256"]:
            raise ValueError("deploy/teacher identity mismatch")
        from FM_distillation.core.fm_backbone import fm_deploy_agent_class
        policy_server.NavDP_Agent = fm_deploy_agent_class(settings["deploy"],
            rtc_enabled=settings["rtc"], beta=settings["rtc_beta"])
    else:
        if digest(settings["student"]) != settings["student_sha256"]:
            raise ValueError("student changed")
        from FM_distillation.core.rtc import fm_rtc_agent_class
        policy_server.NavDP_Agent = fm_rtc_agent_class(settings["student"],meta["teacher_sha256"],settings["rtc_beta"])
    policy_server.init_app("humanoid",no_visualization=True,device="cuda:0",checkpoint=meta["checkpoint"],
                           seed=meta["seed"],recovery="baseline",rtc_enabled=settings["rtc"])
    policy_server.run_server(port=args.port)

