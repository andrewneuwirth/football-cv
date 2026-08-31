"""Data-directory resolution for football-cv.

The data dir defaults to <repo root>/data and can be overridden with the
FOOTBALL_CV_DATA environment variable. Per-play artifacts live in
data/plays/<play_id>/.
"""

from __future__ import annotations

import os
from pathlib import Path

# ml/paths.py -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


def data_dir() -> Path:
    """Resolve the data directory (FOOTBALL_CV_DATA env override, else <repo>/data)."""
    env = os.environ.get("FOOTBALL_CV_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_DIR


def plays_dir(base: Path | None = None) -> Path:
    """The plays/ directory under the data dir."""
    return (base if base is not None else data_dir()) / "plays"


def play_dir(play_id: str, base: Path | None = None) -> Path:
    """Directory for one play's artifacts; created (with parents) if missing."""
    d = plays_dir(base) / play_id
    d.mkdir(parents=True, exist_ok=True)
    return d
