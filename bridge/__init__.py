"""Re-X-NavDP downstream package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "x-navdp"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))
