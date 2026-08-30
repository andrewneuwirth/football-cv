"""Self-contained demo: classify a synthetic play and render it.

Needs no film and no torch — it exercises the headline module (``ml.positions``)
on a hand-built formation, then draws the result to a field diagram.

    python -m ml.demo                # writes examples/demo.png, demo.mp4, demo.positions.json
    python -m ml.demo --out /tmp/x   # write somewhere else

Offense markers are cyan, defense gold; each is labeled with the generic
position token the classifier assigned (DL/LB/CB/S, OL/QB/RB/WR).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ml.positions import classify_play

# --- a synthetic 11-on-11 snap ------------------------------------------------
# Each entry: track_id -> (x_across_yd, depth_from_los_yd, side). x is lateral
# (0 = ball), depth is yards from the line of scrimmage. classify_play only sees
# (x, depth) + side; depth is positive on both sides of the ball.
_PLAY = {
    # offense (O)
    1:  (-4.0, 0.3, "O"), 2: (-2.0, 0.3, "O"), 3: (0.0, 0.3, "O"),
    4:  (2.0, 0.3, "O"),  5: (4.0, 0.3, "O"),                 # 5 OL
    6:  (-18.0, 0.3, "O"), 7: (18.0, 0.3, "O"), 8: (-12.0, 0.3, "O"),  # 3 WR (wide)
    9:  (0.0, 5.5, "O"),                                        # QB (centered, deep)
    10: (-3.0, 6.0, "O"), 11: (3.0, 6.0, "O"),                 # 2 RB
    # defense (D)
    21: (-3.0, 1.0, "D"), 22: (-1.0, 1.0, "D"),
    23: (1.0, 1.0, "D"),  24: (3.0, 1.0, "D"),                # 4 DL
    25: (-4.0, 4.5, "D"), 26: (0.0, 4.5, "D"), 27: (4.0, 4.5, "D"),  # 3 LB
    28: (-18.0, 6.0, "D"), 29: (18.0, 6.0, "D"),              # 2 CB (wide)
    30: (-6.0, 12.0, "D"), 31: (6.0, 12.0, "D"),             # 2 S (deep)
}

_CYAN = (220, 200, 40)    # BGR — offense
_GOLD = (30, 179, 230)    # BGR — defense
_WHITE = (245, 245, 245)
_GREEN = (60, 120, 55)    # field
_LINE = (150, 170, 150)

_W, _H = 1280, 720
_MARGIN = 90


def _labeled():
    """Return {tid: (x, depth, side, token)} for the synthetic play."""
    alignment = {tid: (x, d) for tid, (x, d, _s) in _PLAY.items()}
    split = {tid: s for tid, (_x, _d, s) in _PLAY.items()}
    tokens = classify_play(alignment, split, los=0.0)
    return {tid: (x, d, s, tokens[tid]) for tid, (x, d, s) in _PLAY.items()}


def _to_px(x_yd: float, render_y_yd: float):
    """Field yards -> image pixels. x in [-26.65,26.65]; render_y in [-8,14]."""
    fx = (x_yd + 26.65) / 53.3
    px = int(_MARGIN + fx * (_W - 2 * _MARGIN))
    fy = (render_y_yd + 8.0) / 22.0
    py = int(_MARGIN + fy * (_H - 2 * _MARGIN))
    return px, py


def _draw(frame, players, reveal=1.0, show_labels=True):
    """Draw the field + players onto `frame`. reveal in [0,1] fades players in."""
    frame[:] = _GREEN
    # yard lines every ~5yd (vertical bands across the field)
    for xg in range(-25, 26, 5):
        x0, _ = _to_px(xg, -8.0)
        x1, _ = _to_px(xg, 14.0)
        cv2.line(frame, (x0, _MARGIN), (x1, _H - _MARGIN), _LINE, 1)
    # line of scrimmage (render_y = 0), bold white
    lx0, ly = _to_px(-26.65, 0.0)
    lx1, _ = _to_px(26.65, 0.0)
    cv2.line(frame, (lx0, ly), (lx1, ly), _WHITE, 3)
    cv2.putText(frame, "LOS", (lx1 - 70, ly - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)

    n_show = int(round(len(players) * reveal))
    for i, (tid, (x, d, side, tok)) in enumerate(sorted(players.items())):
        if i >= n_show:
            continue
        # offense drawn below LOS, defense above
        render_y = -d if side == "O" else d
        px, py = _to_px(x, render_y)
        color = _CYAN if side == "O" else _GOLD
        cv2.circle(frame, (px, py), 15, color, -1)
        cv2.circle(frame, (px, py), 15, (20, 20, 20), 1, cv2.LINE_AA)
        if show_labels:
            (tw, _th), _ = cv2.getTextSize(tok, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, tok, (px - tw // 2, py + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)

    # title + legend
    cv2.putText(frame, "football-cv  -  generic position classification",
                (_MARGIN, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _WHITE, 2, cv2.LINE_AA)
    cv2.circle(frame, (_MARGIN + 12, _H - 40), 10, _CYAN, -1)
    cv2.putText(frame, "offense", (_MARGIN + 30, _H - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (_MARGIN + 160, _H - 40), 10, _GOLD, -1)
    cv2.putText(frame, "defense", (_MARGIN + 178, _H - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
    return frame


def render(out_dir: Path) -> dict:
    """Write demo.png, demo.mp4, demo.positions.json to out_dir. Returns tokens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    players = _labeled()

    frame = np.zeros((_H, _W, 3), np.uint8)

    # still slide
    _draw(frame, players, reveal=1.0, show_labels=True)
    png = out_dir / "demo.png"
    cv2.imwrite(str(png), frame)

    # short build-up video: players fade in, then labels, then hold
    mp4 = out_dir / "demo.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (_W, _H))
    for f in range(24):  # 1s: markers appear
        _draw(frame, players, reveal=(f + 1) / 24, show_labels=False)
        writer.write(frame)
    for _ in range(72):  # 3s: full labeled slide held
        _draw(frame, players, reveal=1.0, show_labels=True)
        writer.write(frame)
    writer.release()

    tokens = {str(tid): tok for tid, (_x, _d, _s, tok) in players.items()}
    (out_dir / "demo.positions.json").write_text(json.dumps(tokens, indent=2))
    print(f"demo: wrote {png}, {mp4}, and demo.positions.json to {out_dir}")
    return tokens


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ml.demo")
    p.add_argument("--out", default="examples", help="output directory")
    args = p.parse_args(argv)
    render(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
