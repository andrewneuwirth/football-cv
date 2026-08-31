"""High-level programmatic API for football-cv.

For people who want the toolkit, not the CLI or the browser UI:

    import footballcv

    positions = footballcv.analyze(
        video="play.mp4",
        calibration={
            "frame": 120,
            "points": [
                {"px": 20,   "py": 1030, "fx": 30, "fy": 0.0},     # 30 yd, near sideline
                {"px": 1900, "py": 300,  "fx": 30, "fy": 53.33},   # 30 yd, far sideline
                {"px": 700,  "py": 1040, "fx": 45, "fy": 0.0},     # 45 yd, near sideline
                {"px": 1905, "py": 470,  "fx": 45, "fy": 53.33},   # 45 yd, far sideline
            ],
        },
        snap=120,          # optional: frame to anchor the pre-snap alignment
    )
    # -> {"3": "DL", "11": "LB", "24": "S", "41": "OL", "52": "QB", ...}

Running `analyze` needs the detection extra (`pip install "football-cv[detect]"`).
Lower-level helpers let you drive the file-based pipeline stage by stage.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from footballcv import ingest, paths


def calibrate(play_id, data_dir, points, frame=0):
    """Write calibration.json for a play from clicked field points.

    points: list of {"px","py","fx","fy"} mapping pixel -> field yards
    (fx = yard line 0-100 downfield, fy = across-field 0..53.33). At least 4,
    spanning two yard lines and both sidelines. Marked source="manual" so
    autocal never overwrites it.
    """
    pdir = paths.play_dir(play_id, Path(data_dir))
    pdir.mkdir(parents=True, exist_ok=True)
    calib = {"frame": int(frame), "source": "manual", "points": list(points)}
    (pdir / "calibration.json").write_text(json.dumps(calib, indent=1))
    return calib


def set_snap(play_id, data_dir, frame):
    """Mark the snap frame (anchors the pre-snap alignment); writes snap.json."""
    pdir = paths.play_dir(play_id, Path(data_dir))
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "snap.json").write_text(json.dumps({"frame": int(frame)}))


def load_positions(data_dir, play_id):
    """Return {track_id: position_token} from a play's positions.json (or {})."""
    f = paths.play_dir(play_id, Path(data_dir)) / "positions.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def load_labels(data_dir, play_id):
    """Return the richer labels.json (teams, LOS, snap, confidences) or {}."""
    f = paths.play_dir(play_id, Path(data_dir)) / "labels.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def _as_calibration(calibration):
    if isinstance(calibration, (str, Path)):
        calibration = json.loads(Path(calibration).read_text())
    if not isinstance(calibration, dict) or "points" not in calibration:
        raise ValueError("calibration must be a dict/path with a 'points' list")
    return calibration


def analyze(video, *, calibration, snap=None, data_dir=None, play_id="play"):
    """Run the full pipeline on one clip and return {track_id: position}.

    video:       path to a video file (one play).
    calibration: dict {"frame", "points"} or a path to a calibration.json.
    snap:        optional frame index to anchor the pre-snap alignment.
    data_dir:    where artifacts are written (a temp dir by default). Every
                 intermediate (detections, tracks, field coords, teams, labels)
                 is left on disk under data_dir/plays/<play_id>/ for inspection.

    Requires the detection extra: pip install "football-cv[detect]".
    """
    from footballcv import cli  # local import: pulls torch-optional stages lazily

    data_dir = Path(data_dir) if data_dir else Path(tempfile.mkdtemp(prefix="footballcv-"))
    calib = _as_calibration(calibration)

    ingest.run(play_id, data_dir, video=video)
    calibrate(play_id, data_dir, calib["points"], frame=calib.get("frame", 0))
    if snap is not None:
        set_snap(play_id, data_dir, snap)

    for stage in cli.ORDER:
        cli.STAGES[stage](play_id, data_dir)

    return load_positions(data_dir, play_id)
