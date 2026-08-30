# ml/positions.py
"""Generic player-position classifier. Maps field-relative geometry to
standard positions using only field distances and depths."""
from __future__ import annotations
from pathlib import Path
import json
import math
from ml import geometry, paths

POSITIONS = ("DL", "LB", "CB", "S", "OL", "QB", "RB", "WR")

# depth bands measured in yards from the line of scrimmage (los)
_LINE_BAND = 1.5      # within this of the front edge reads as "on the line"
_WIDE_YDS = 12.0      # |x| beyond this reads as split wide / boundary
_SAFETY_DEPTH = 8.0   # defenders deeper than this off the line are safeties
_BACKFIELD_DEPTH = 3.0  # offense bodies deeper than this are backs/QB


def classify_player(rel, side, los, front_edge) -> str:
    """rel=(x_across, y_depth_from_los). side='O' or 'D'. Returns a POSITIONS token."""
    x, depth = rel
    on_line = abs(depth - front_edge) <= _LINE_BAND
    wide = abs(x) >= _WIDE_YDS
    if side == "D":
        if wide:
            return "CB"
        if on_line:
            return "DL"
        if depth >= _SAFETY_DEPTH:
            return "S"
        return "LB"
    # offense: a body split wide is a receiver even when it aligns on the line
    if wide:
        return "WR"
    if on_line:
        return "OL"
    if depth >= _BACKFIELD_DEPTH:
        # deepest, laterally-centered body is the QB; others are RBs
        return "QB" if abs(x) <= 2.0 and depth >= _BACKFIELD_DEPTH + 2.0 else "RB"
    return "WR"


def classify_play(alignment, split, los) -> dict[int, str]:
    """alignment: {track_id: (x, y)}. split: {track_id: 'O'|'D'}. Returns {track_id: pos}."""
    o_front = min((y for tid, (x, y) in alignment.items() if split.get(tid) == "O"),
                  default=los)
    d_front = min((y for tid, (x, y) in alignment.items() if split.get(tid) == "D"),
                  default=los)
    out: dict[int, str] = {}
    for tid, (x, y) in alignment.items():
        side = split.get(tid, "O")
        front = o_front if side == "O" else d_front
        out[tid] = classify_player((x, y - los), side, los, front - los)
    return out


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


def run(play_id: str, data_dir: Path) -> None:
    """Read a play's field.json, classify every tracked player, write positions.json."""
    pdir = paths.play_dir(play_id, data_dir)
    field = json.loads((pdir / "field.json").read_text())
    tracks = geometry.track_positions(field)
    smoothed = geometry.smooth_positions(tracks)
    snap_idx, _, _ = geometry.find_snap(_mean_speeds(smoothed))
    alignment = geometry.alignment_positions(smoothed, snap_idx)

    # Split the field at the line of scrimmage using pre-snap x-positions, then
    # label each track by which side of that boundary it aligns on.
    ordered = sorted(alignment.items(), key=lambda kv: kv[1][0])
    pts = [xy for _tid, xy in ordered]
    if len(pts) >= 2:
        split_k, los, _lf, _rf = geometry.split_and_los(pts)
    else:
        split_k, los = len(pts), 0.0
    # Side with the higher on-front crowding fraction is the offense.
    left_ids = [tid for tid, _ in ordered[:split_k]]
    right_ids = [tid for tid, _ in ordered[split_k:]]
    left_front = min((alignment[t][1] for t in left_ids), default=0.0)
    right_front = min((alignment[t][1] for t in right_ids), default=0.0)
    left_is_offense = left_front <= right_front
    split: dict[int, str] = {}
    for tid in left_ids:
        split[tid] = "O" if left_is_offense else "D"
    for tid in right_ids:
        split[tid] = "D" if left_is_offense else "O"

    result = classify_play(alignment, split, los)
    (pdir / "positions.json").write_text(
        json.dumps({str(k): v for k, v in result.items()})
    )
