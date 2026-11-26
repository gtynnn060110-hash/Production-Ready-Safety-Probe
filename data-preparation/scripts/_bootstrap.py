"""Resolve shared data-preparation modules for scripts run from their new locations."""
from pathlib import Path
import sys

def setup():
    root = Path(__file__).resolve().parents[1]
    paths = [root, *(root / "scripts" / x for x in ("prepare", "curate", "judge", "rollout", "orchestration", "tools"))]
    for path in reversed(paths):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
