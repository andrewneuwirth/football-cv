"""football-cv: computer-vision pipeline for football film.

High-level API:

    import footballcv
    positions = footballcv.analyze(video="play.mp4", calibration={...}, snap=120)

See footballcv.api for calibrate(), set_snap(), load_positions(), load_labels().
"""
from footballcv.api import (
    analyze,
    calibrate,
    set_snap,
    load_positions,
    load_labels,
)

__all__ = ["analyze", "calibrate", "set_snap", "load_positions", "load_labels"]
