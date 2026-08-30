"""Self-contained demo: classify a synthetic play and render it.

Needs no film and no torch — it exercises the headline module (``ml.positions``)
on a hand-built formation in the pipeline's real field-coordinate convention
(down = yards along the field, across = yards sideline-to-sideline), then draws
the result to a field diagram.

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

from ml.positions import classify_play, FIELD_WIDTH_YD

_LOS = 50.0                      # line of scrimmage (a `down` value)
_MID = FIELD_WIDTH_YD / 2.0      # field center (across)

# --- a synthetic 11-on-11 snap in (down, across, side) ------------------------
# down runs along the field (offense < LOS < defense); across is sideline-to-
# sideline (0 .. 53.33). classify_play sees (down, across) + side + the LOS.
_PLAY = {
    # offense (down < LOS)
    1: (49.5, _MID - 4, "O"), 2: (49.5, _MID - 2, "O"), 3: (49.5, _MID, "O"),
    4: (49.5, _MID + 2, "O"), 5: (49.5, _MID + 4, "O"),          # 5 OL
    6: (49.5, 5.0, "O"), 7: (49.5, 48.0, "O"), 8: (49.5, 13.0, "O"),  # 3 WR (wide)
    9: (44.5, _MID, "O"),                                         # QB (centered, deep)
    10: (44.0, _MID - 5, "O"), 11: (44.0, _MID + 5, "O"),        # 2 RB
    # defense (down > LOS)
    21: (51.0, _MID - 3, "D"), 22: (51.0, _MID - 1, "D"),
    23: (51.0, _MID + 1, "D"), 24: (51.0, _MID + 3, "D"),        # 4 DL
    25: (54.5, _MID - 4, "D"), 26: (54.5, _MID, "D"), 27: (54.5, _MID + 4, "D"),  # 3 LB
    28: (56.0, 5.0, "D"), 29: (56.0, 48.0, "D"),                 # 2 CB (wide)
    30: (62.0, _MID - 6, "D"), 31: (62.0, _MID + 6, "D"),        # 2 S (deep)
}

_CYAN = (220, 200, 40)    # BGR — offense
_GOLD = (30, 179, 230)    # BGR — defense
_WHITE = (245, 245, 245)
_GREEN = (60, 120, 55)    # field
_LINE = (150, 170, 150)

_W, _H = 1280, 720
_MARGIN = 90
_DOWN_LO, _DOWN_HI = 42.0, 64.0   # visible downfield window


def _labeled():
    """Return {tid: (down, across, side, token)} for the synthetic play."""
    alignment = {tid: (down, across) for tid, (down, across, _s) in _PLAY.items()}
    split = {tid: s for tid, (_d, _a, s) in _PLAY.items()}
    tokens = classify_play(alignment, split, los=_LOS)
    return {tid: (down, across, s, tokens[tid]) for tid, (down, across, s) in _PLAY.items()}


def _to_px(across: float, down: float):
    """Field yards -> image pixels. across in [0,53.33]; down in [42,64]."""
    px = int(_MARGIN + (across / FIELD_WIDTH_YD) * (_W - 2 * _MARGIN))
    py = int(_MARGIN + ((down - _DOWN_LO) / (_DOWN_HI - _DOWN_LO)) * (_H - 2 * _MARGIN))
    return px, py


def _draw(frame, players, reveal=1.0, show_labels=True):
    """Draw the field + players onto `frame`. reveal in [0,1] fades players in."""
    frame[:] = _GREEN
    # yard lines (constant down -> horizontal bands)
    for dg in range(44, 63, 5):
        _, y0 = _to_px(0.0, dg)
        cv2.line(frame, (_MARGIN, y0), (_W - _MARGIN, y0), _LINE, 1)
    # line of scrimmage (down == LOS), bold white
    _, ly = _to_px(0.0, _LOS)
    cv2.line(frame, (_MARGIN, ly), (_W - _MARGIN, ly), _WHITE, 3)
    cv2.putText(frame, "LOS", (_W - _MARGIN - 60, ly - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)

    n_show = int(round(len(players) * reveal))
    for i, (tid, (down, across, side, tok)) in enumerate(sorted(players.items())):
        if i >= n_show:
            continue
        px, py = _to_px(across, down)
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

    tokens = {str(tid): tok for tid, (_d, _a, _s, tok) in players.items()}
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
