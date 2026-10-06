"""Launch a named CLI stage from the repository root.

Each work line (FM / RL / DDIM) exposes a single ``cli.py``; this root entry
maps short command names onto those stages and forwards the remaining argv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "baselines/x-navdp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _forward(module, stage, arguments):
    """Delegate to a line's cli.main([stage, *arguments])."""
    from importlib import import_module
    import_module(module).main([stage, *arguments])


# command name -> (cli module, stage within that module)
COMMANDS = {
    # Flow Matching distillation
    "fm": ("FM_distillation.cli", "train"),
    "fm-collect": ("FM_distillation.cli", "collect-dual"),
    "fm-collect-full": ("FM_distillation.cli", "collect-full"),
    "fm-joint": ("FM_distillation.cli", "collect-joint"),
    "fm-joint-train": ("FM_distillation.cli", "collect-joint-train"),
    "fm-all-scenes": ("FM_distillation.cli", "train-all-scenes"),
    "fm-all-candidates": ("FM_distillation.cli", "train-all-candidates"),
    "fm-eval": ("FM_distillation.cli", "eval-closed-loop"),
    "fm-rtc-eval": ("FM_distillation.cli", "eval-rtc"),
    "fm-label-validation": ("FM_distillation.cli", "label-validation"),
    "fm-merge-labels": ("FM_distillation.cli", "merge-labels"),
    "fm-merge-success": ("FM_distillation.cli", "merge-success"),
    "fm-queue": ("FM_distillation.cli", "queue"),
    "fm-bench": ("FM_distillation.cli", "bench"),
    "fm-export": ("FM_distillation.cli", "export-deploy"),
    # RL execution
    "rl-collect": ("rl.cli", "collect"),
    "rl-collect-all": ("rl.cli", "collect-all"),
    "rl-bc": ("rl.cli", "train-bc"),
    "rl-train": ("rl.cli", "train-tracker"),
    "rl-multiscene": ("rl.cli", "train-full"),
    "rl-residual": ("rl.cli", "train-reactive"),
    "rl-bench": ("rl.cli", "bench"),
    # DDIM
    "ddim-benchmark": ("ddim.cli", "benchmark"),
    "ddim-compare": ("ddim.cli", "compare"),
    "ddim-sweep": ("ddim.cli", "sweep"),
}

_HELP = """\
usage: python run.py <command> [args]

FM distillation:
  fm                train (freeze / label / train / rank)
  fm-collect        dual-GPU capture of an existing collection
  fm-collect-full   sequential GPU-0 capture sweep
  fm-joint          GPU1 joint capture + shared-encoder labeling
  fm-joint-train    joint capture bound to safe TRAIN episode pairs
  fm-all-scenes     every optimizer update includes every train scene
  fm-all-candidates all-scene, all-eight-candidate CFM
  fm-eval           FM + frozen Q + recovery/MPC (closed loop)
  fm-rtc-eval       FM trajectory RTC evaluation
  fm-label-validation  label validation scenes
  fm-merge-labels   merge completed train/validation labels
  fm-merge-success  merge legacy+joint labels, filter failed train
  fm-queue          wait for training + idle GPU, then RTC eval
  fm-bench          batch=1 full-network latency benchmark
  fm-export         bundle an existing student + teacher into one deploy.pt

RL execution:
  rl-collect        MPC-teacher BC collection (single scene)
  rl-collect-all    BC collection across all home_train scenes
  rl-bc             behavior-cloning warm-start
  rl-train          MPC-free direct tracker PPO training
  rl-multiscene     sequential multi-scene direct-tracker training
  rl-residual       single-G1 residual PPO training
  rl-bench          visual-encoder latency benchmark

DDIM:
  ddim-benchmark    full-network latency
  ddim-compare      DDPM vs DDIM trajectory-distribution comparison
  ddim-sweep        sampler/step/RTC matrix sweep
"""


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return
    command = argv[0]
    forwarded = argv[1:]
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if command not in COMMANDS:
        print(f"unknown command: {command}\n\n{_HELP}", file=sys.stderr)
        raise SystemExit(2)
    module, stage = COMMANDS[command]
    if stage is not None:
        _forward(module, stage, forwarded)
    else:
        import runpy
        sys.argv = [command, *forwarded]
        runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
