"""Consolidated FM_distillation library module (entry points live in cli.py)."""

from rexnavdp import BASE, ROOT

"""Wait for a specific FM training completion AND idle GPU, then evaluate RTC.

Run in tmux. Never stops a training process, touches GPU1, or occupies GPU while waiting.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from FM_distillation.src.storage import atomic_json, digest, read_json, writer_lock
from FM_distillation.src import dataset as fm_dataset
from FM_distillation.src.evaluation import read_completed


def gpu_compute_pids(index):
    query = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuids = {int(line.split(',')[0]):line.split(',')[1].strip() for line in query.splitlines()}
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'],text=True)
    return [int(line.split(',')[1]) for line in processes.splitlines()
            if line.split(',')[0].strip() == uuids[index]]

