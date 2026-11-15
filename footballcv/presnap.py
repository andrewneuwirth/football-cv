"""Pre-snap look from film: turn one play's pipeline outputs into play-call features.

Reads labels.json (snap, LOS, offense direction, O/D), positions.json (OL/QB/RB/WR)
and field_coords.json (per-frame field yards) and returns the film-derived columns
of the play-call CSV (see docs/playcall.md):

    yard_ln     yards from the offense's own goal line to the LOS (needs a 0-100 calibration)
    hash        L / M / R: ball spot across the field, from the offense's view
    off_str     L / R / bal: side with more offensive players outside the tackles
    off_form    "<strong>x<weak>": players outside the tackles per side, e.g. 3x1, 2x2
    backfield   number of RBs (0, 1, 2, 3)
    qb_set      gun / under: QB depth behind the LOS
    box         defenders within BOX_WIDTH of the ball and BOX_DEPTH of the LOS
    motion_dir  L / R / none: biggest lateral move by a skill player in the last MOTION_S seconds

Game situation (down, distance, quarter, previous play) is not in the film: the user
supplies it. Only stdlib + positions.py helpers, so the base install can run it.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from footballcv import paths
from footballcv.positions import _alignment_positions, _load_json, _track_positions

FIELD_MID = 53.33 / 2
HASH_BAND = 4.0      # ball within this many yards of the middle = M
TACKLE_LAT = 4.0     # yards from the ball beyond which a player is "outside the tackles"
BOX_WIDTH = 6.0      # box = this far either side of the ball ...
BOX_DEPTH = 7.0      # ... and this deep off the LOS
GUN_DEPTH = 3.5      # QB at least this far behind the LOS = gun
MOTION_S = 2.0       # pre-snap window scanned for motion
MOTION_YD = 2.5      # lateral travel that counts as motion

FILM_COLUMNS = ["yard_ln", "hash", "off_str", "off_form", "backfield", "qb_set", "box", "motion_dir"]
USER_COLUMNS = ["game", "game_date", "play_no", "qtr", "down", "dist", "prev_type", "prev_gain", "run_pass", "play_dir"]


def _side(v: float, band: float) -> str:
    return "L" if v > band else "R" if v < -band else "M"


def features(play_id: str, data_dir) -> dict:
    """Film-derived pre-snap features for one play that has run through `positions`."""
    pdir = paths.play_dir(play_id, Path(data_dir))
    labels = _load_json(pdir / "labels.json")
    tokens = _load_json(pdir / "positions.json")
    tracks = _track_positions(_load_json(pdir / "field_coords.json"))
    meta = pdir / "meta.json"
    fps = float(_load_json(meta).get("fps", 30.0) or 30.0) if meta.exists() else 30.0
    snap, los = int(labels["snap_frame"]), float(labels["los_x"])
    sign = 1.0 if labels.get("offense_direction", "+x") == "+x" else -1.0
    teams = labels.get("teams", {})
    pts = _alignment_positions(tracks, snap)

    off = {t: tokens.get(str(t), "") for t in pts if teams.get(str(t)) == "O"}
    dfn = [t for t in pts if teams.get(str(t)) == "D"]
    ol = [pts[t][1] for t, k in off.items() if k == "OL"] or [pts[t][1] for t in off] or [FIELD_MID]
    ball_y = median(ol)
    # facing +x the offense's left is +y (far sideline); flip for -x
    lat = lambda t: (pts[t][1] - ball_y) * sign
    depth = lambda t: (los - pts[t][0]) * sign   # > 0 = offense's side of the LOS

    outside = [lat(t) for t, k in off.items() if k in ("WR", "RB", "OL") and abs(lat(t)) > TACKLE_LAT]
    n_l, n_r = sum(v > 0 for v in outside), sum(v < 0 for v in outside)
    qbs = [depth(t) for t, k in off.items() if k == "QB"]

    motion, best = "none", MOTION_YD
    lo = snap - int(MOTION_S * fps)
    for t, k in off.items():
        if k not in ("WR", "RB"):
            continue
        frames = [i for i in tracks[t] if lo <= i <= snap]
        if len(frames) >= 2:
            move = (tracks[t][max(frames)][1] - tracks[t][min(frames)][1]) * sign
            if abs(move) >= best:
                motion, best = ("L" if move > 0 else "R"), abs(move)

    return {
        "play_id": play_id,
        "yard_ln": round(los if sign > 0 else 100 - los, 1),
        "hash": _side((ball_y - FIELD_MID) * sign, HASH_BAND),
        "off_str": "L" if n_l > n_r else "R" if n_r > n_l else "bal",
        "off_form": f"{max(n_l, n_r)}x{min(n_l, n_r)}",
        "backfield": str(min(3, sum(k == "RB" for k in off.values()))),
        "qb_set": ("gun" if max(qbs) >= GUN_DEPTH else "under") if qbs else "",
        "box": sum(abs(lat(t)) <= BOX_WIDTH and -BOX_DEPTH <= depth(t) <= 0.5 for t in dfn),
        "motion_dir": motion,
    }


def write_csv(play_ids, data_dir, out) -> list[dict]:
    """One play-call CSV row per play: film columns filled, situation columns left for the user."""
    rows = [features(p, data_dir) for p in play_ids]
    cols = ["play_id", *USER_COLUMNS, *FILM_COLUMNS]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows
