"""Render the classified positions onto a play's film.

Reads tracks.json (per-frame pixel boxes) + positions.json (track_id -> token)
and draws each classified player's box + position label on every frame, then
transcodes to browser-playable H.264 with ffmpeg. Only tracks that made it into
positions.json (the on-field participants) are drawn.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2

_OFFENSE = {"OL", "QB", "RB", "WR"}
_CYAN = (220, 200, 40)   # BGR — offense
_GOLD = (30, 179, 230)   # BGR — defense


def _color(tok: str):
    return _CYAN if tok in _OFFENSE else _GOLD


def render(play: str, data_dir: Path, web_dir: Path) -> tuple[str, int]:
    """Write <web_dir>/<play>_overlay.mp4. Returns (url_path, frame_count)."""
    pdir = data_dir / "plays" / play
    tracks = {t["track_id"]: t for t in json.loads((pdir / "tracks.json").read_text())["tracks"]}
    positions = json.loads((pdir / "positions.json").read_text())

    by_frame: dict[int, list[tuple[str, list[float]]]] = {}
    for tid_s, tok in positions.items():
        tr = tracks.get(int(tid_s))
        if not tr:
            continue
        for fr in tr["frames"]:
            by_frame.setdefault(fr["i"], []).append((tok, fr["box"]))

    cap = cv2.VideoCapture(str(pdir / "clip.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    raw = pdir / "_overlay_raw.mp4"
    out = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tok, b in by_frame.get(i, []):
            x1, y1, x2, y2 = map(int, b)
            c = _color(tok)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            cv2.rectangle(frame, (x1, y1 - 26), (x1 + 46, y1), c, -1)
            cv2.putText(frame, tok, (x1 + 4, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2, cv2.LINE_AA)
        out.write(frame)
        i += 1
    cap.release()
    out.release()

    dest = web_dir / f"{play}_overlay.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(raw),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
        check=True,
    )
    raw.unlink(missing_ok=True)
    return f"/{play}_overlay.mp4", i
