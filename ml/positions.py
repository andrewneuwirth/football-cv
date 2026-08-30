# ml/positions.py
"""Generic player-position classifier. Maps field-relative geometry to
standard positions using only field distances and depths.

Field coordinates follow the pipeline's `field_coords.json`: each player is a
(down, across) pair in yards, where `down` runs along the length of the field
(the yard line, 0-100) and `across` runs sideline-to-sideline (0 .. ~53.33).
The line of scrimmage is a `down` value; a player's depth is how far off it
they are, and how "wide" they are is their distance from the field's middle.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
from ml import geometry, paths

POSITIONS = ("DL", "LB", "CB", "S", "OL", "QB", "RB", "WR")

FIELD_WIDTH_YD = 53.33          # sideline-to-sideline
_CENTER = FIELD_WIDTH_YD / 2.0
_LINE_BAND = 1.6      # within this many yd of the LOS reads as "on the line"
_WIDE_YDS = 12.0      # this far from the field's middle reads as split wide
_SAFETY_DEPTH = 8.0   # defenders deeper than this off the line are safeties
_BACKFIELD_DEPTH = 3.0  # offense bodies deeper than this are backs / QB
_FIELD_MARGIN = 3.0     # allow this far outside the sidelines as still "on field"
_POSITION_RANGE = 25.0  # only classify bodies within this many yd of the LOS


def _on_field(alignment):
    """Drop bodies off the field (sideline crews, refs way wide, bad projections)."""
    return {
        tid: (down, across) for tid, (down, across) in alignment.items()
        if -_FIELD_MARGIN <= across <= FIELD_WIDTH_YD + _FIELD_MARGIN
    }


def classify_player(down, across, los, side) -> str:
    """Classify one player from field coords.

    down/across/los are yards (field_coords convention). side is 'O' or 'D'.
    Returns one of POSITIONS.
    """
    depth = abs(down - los)                 # yards off the line of scrimmage
    lateral = abs(across - _CENTER)         # yards from the middle of the field
    wide = lateral >= _WIDE_YDS
    if side == "D":
        if wide:
            return "CB"
        if depth <= _LINE_BAND:
            return "DL"
        if depth >= _SAFETY_DEPTH:
            return "S"
        return "LB"
    # offense: a body split wide is a receiver even when it aligns on the line
    if wide:
        return "WR"
    if depth <= _LINE_BAND:
        return "OL"
    if depth >= _BACKFIELD_DEPTH:
        # deepest, laterally-centered body is the QB; others are RBs
        return "QB" if lateral <= 3.0 and depth >= _BACKFIELD_DEPTH + 1.5 else "RB"
    return "WR"


def classify_play(alignment, split, los) -> dict[int, str]:
    """alignment: {track_id: (down, across)}. split: {track_id: 'O'|'D'}."""
    return {
        tid: classify_player(down, across, los, split.get(tid, "O"))
        for tid, (down, across) in alignment.items()
    }


def _mean_speeds(positions_by_track) -> list[float]:
    """Team-wide mean per-frame speed series from {track_id: {frame: (x, y)}}."""
    per_frame_speeds: dict[int, list[float]] = {}
    for frames in positions_by_track.values():
        idxs = sorted(frames)
        for prev, cur in zip(idxs, idxs[1:]):
            (x0, y0), (x1, y1) = frames[prev], frames[cur]
            step = max(1, cur - prev)
            speed = math.hypot(x1 - x0, y1 - y0) / step
            per_frame_speeds.setdefault(cur, []).append(speed)
    if not per_frame_speeds:
        return []
    last = max(per_frame_speeds)
    return [
        (sum(v) / len(v)) if (v := per_frame_speeds.get(i)) else 0.0
        for i in range(last + 1)
    ]


def _offense_side(ordered, split_k, los, left_front, right_front):
    """Return the set of track_ids on the OFFENSE side of the split."""
    left_ids = [t for t, _ in ordered[:split_k]]
    right_ids = [t for t, _ in ordered[split_k:]]
    pts = [xy for _t, xy in ordered]
    center_y = sum(a for _d, a in pts) / len(pts)
    left_pts = [xy for _t, xy in ordered[:split_k]]
    right_pts = [xy for _t, xy in ordered[split_k:]]
    # backfield sits away from the LOS: smaller `down` on the left, larger on right
    left_off = geometry.offense_likeness(left_pts, left_front, -1.0, center_y)
    right_off = geometry.offense_likeness(right_pts, right_front, 1.0, center_y)
    return set(left_ids if left_off >= right_off else right_ids)


def run(play_id: str, data_dir: Path) -> None:
    """Read field_coords.json, classify every tracked player, write positions.json."""
    pdir = paths.play_dir(play_id, data_dir)
    field = json.loads((pdir / "field_coords.json").read_text())
    tracks = geometry.track_positions(field)
    smoothed = geometry.smooth_positions(tracks)
    snap_idx, _, _ = geometry.find_snap(_mean_speeds(smoothed))
    alignment = _on_field(geometry.alignment_positions(smoothed, snap_idx))
    if len(alignment) < 4:
        (pdir / "positions.json").write_text(json.dumps({}))
        return

    # first pass: find the LOS over every on-field body, then keep only the
    # bodies near it (the ~22 in the play, not deep-downfield stragglers).
    ordered = sorted(alignment.items(), key=lambda kv: kv[1][0])
    _k, los0, _lf, _rf = geometry.split_and_los([xy for _t, xy in ordered])
    parts = {t: (d, a) for t, (d, a) in alignment.items() if abs(d - los0) <= _POSITION_RANGE}
    if len(parts) < 4:
        parts = alignment

    ordered = sorted(parts.items(), key=lambda kv: kv[1][0])  # by downfield
    split_k, los, left_front, right_front = geometry.split_and_los([xy for _t, xy in ordered])
    offense = _offense_side(ordered, split_k, los, left_front, right_front)
    split = {tid: ("O" if tid in offense else "D") for tid, _ in ordered}

    result = classify_play(parts, split, los)
    (pdir / "positions.json").write_text(
        json.dumps({str(k): v for k, v in result.items()})
    )
