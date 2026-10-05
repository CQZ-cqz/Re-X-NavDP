"""Single source of truth for repository paths and ``sys.path`` bootstrap.

The X-NavDP baseline lives under ``x-navdp/`` and is imported as bare ``eval.*``
and ``src.*`` modules, so ``x-navdp/`` must be on ``sys.path``.  Downstream
packages (``FM_distillation``, ``rl``, ``bridge``, ``ddim``) are regular packages
under the repository root.

Every module that needs a repository path does ``from rexnavdp import BASE, ROOT``
instead of re-deriving it from ``__file__``.  Importing this package also arranges
``sys.path`` once, so there is no per-file bootstrap block anywhere else.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "x-navdp"
CHECKPOINT_DIR = ROOT / "checkpoints"


def default_checkpoint(name="x-navdp_posttrain.ckpt"):
    """Return the repository checkpoint path (may be a symlink)."""
    return CHECKPOINT_DIR / name


def bootstrap():
    """Ensure the repo root and the vendored x-navdp prefix are importable."""
    for path in (ROOT, BASE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


bootstrap()
