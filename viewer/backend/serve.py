"""Local backend for the football-cv viewer.

Serves the static viewer (viewer/web) and exposes a few endpoints so the
browser can:
  - fetch a frame image to calibrate on        GET  /api/frame/<play>?i=<n>
  - save a hand-clicked calibration            POST /api/calibrate/<play>
  - run field -> teamcolor -> positions        POST /api/analyze/<play>
  - read pipeline artifacts                     GET  /api/artifact/<play>/<name>

Run:  python -m viewer.backend.serve            (from the repo root)
      python viewer/backend/serve.py

This is dev tooling, not part of the published pip package.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlay  # noqa: E402  (same-dir module)

REPO = Path(__file__).resolve().parents[2]      # .../football-cv
WEB = REPO / "viewer" / "web"                    # static root
DATA = REPO / "data"
PLAYS = DATA / "plays"
PORT = 8000

# play ids and artifact names are attacker-controlled path segments: allow only
# a strict character set (no separators, no leading '-', no '..').
_PLAY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def _valid_play(play: str) -> bool:
    return bool(play) and _PLAY_RE.match(play) is not None


def _play_dir(play: str) -> Path:
    return PLAYS / play


class Handler(BaseHTTPRequestHandler):
    # ---- helpers -----------------------------------------------------------
    def _send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # no CORS header: the viewer is same-origin, and a wildcard would let any
        # web page drive this unauthenticated local API cross-origin.
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def log_message(self, *a):  # quieter
        pass

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path.startswith("/api/frame/"):
            return self._get_frame(path.split("/")[-1], parse_qs(u.query))
        if path.startswith("/api/meta/"):
            return self._get_meta(path.split("/")[-1])
        if path.startswith("/api/artifact/"):
            parts = path.split("/", 4)
            if len(parts) < 5:
                return self._json({"error": "bad request"}, 400)
            _, _, _, play, name = parts
            if not _valid_play(play) or not _NAME_RE.match(name):
                return self._json({"error": "bad request"}, 400)
            base = _play_dir(play).resolve()
            f = (base / name).resolve()
            if not str(f).startswith(str(base) + "/") or not f.is_file():
                return self._json({"error": "not found"}, 404)
            return self._send(200, f.read_bytes(), "application/json")
        # static
        rel = path.lstrip("/") or "index.html"
        f = (WEB / rel).resolve()
        if not str(f).startswith(str(WEB)) or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = {
            ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
            ".json": "application/json", ".mp4": "video/mp4", ".jpg": "image/jpeg",
            ".png": "image/png",
        }.get(f.suffix, "application/octet-stream")
        return self._send(200, f.read_bytes(), ctype)

    def _get_meta(self, play):
        if not _valid_play(play):
            return self._json({"error": "bad play id"}, 400)
        clip = _play_dir(play) / "clip.mp4"
        if not clip.is_file():
            return self._json({"error": f"no clip for {play}"}, 404)
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        cap.release()
        return self._json({"frames": n, "fps": fps, "width": w, "height": h})

    def _get_frame(self, play, q):
        if not _valid_play(play):
            return self._json({"error": "bad play id"}, 400)
        try:
            i = max(0, int((q.get("i", ["0"]) or ["0"])[0]))
        except ValueError:
            return self._json({"error": "bad frame index"}, 400)
        clip = _play_dir(play) / "clip.mp4"
        if not clip.is_file():
            return self._json({"error": f"no clip for {play}"}, 404)
        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return self._json({"error": f"cannot read frame {i}"}, 400)
        ok, buf = cv2.imencode(".jpg", frame)
        return self._send(200, buf.tobytes(), "image/jpeg")

    # ---- POST --------------------------------------------------------------
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if u.path.startswith("/api/calibrate/"):
            play = u.path.split("/")[-1]
            if not _valid_play(play):
                return self._json({"error": "bad play id"}, 400)
            pdir = _play_dir(play)
            if not pdir.is_dir():
                return self._json({"error": "unknown play"}, 404)
            (pdir / "calibration.json").write_text(json.dumps(body, indent=2))
            return self._json({"ok": True, "points": len(body.get("points", []))})

        if u.path.startswith("/api/analyze/"):
            play = u.path.split("/")[-1]
            return self._analyze(play)

        if u.path.startswith("/api/render/"):
            play = u.path.split("/")[-1]
            if not _valid_play(play) or not _play_dir(play).is_dir():
                return self._json({"error": "bad play id"}, 400)
            if not (_play_dir(play) / "positions.json").is_file():
                return self._json({"error": "run analysis first"}, 400)
            try:
                url, frames = overlay.render(play, DATA, WEB)
            except Exception as e:  # ffmpeg/opencv failures -> report, don't crash
                return self._json({"ok": False, "error": str(e)[:500]}, 500)
            return self._json({"ok": True, "url": url, "frames": frames})

        return self._json({"error": "unknown endpoint"}, 404)

    def _analyze(self, play):
        """Run the classification stages that follow detect+track."""
        if not _valid_play(play) or not _play_dir(play).is_dir():
            return self._json({"error": "bad play id"}, 400)
        stages = ["field", "teamcolor", "positions"]
        py = sys.executable
        logs = {}
        for s in stages:
            p = subprocess.run(
                [py, "-m", "ml.cli", s, play, "--data", str(DATA)],
                cwd=str(REPO), capture_output=True, text=True,
            )
            logs[s] = (p.stdout + p.stderr)[-2000:]
            if p.returncode != 0:
                return self._json({"ok": False, "failed": s, "logs": logs}, 500)
        pos_file = _play_dir(play) / "positions.json"
        positions = json.loads(pos_file.read_text()) if pos_file.is_file() else {}
        return self._json({"ok": True, "positions": positions, "logs": logs})


def main():
    print(f"football-cv backend on http://localhost:{PORT}  (serving {WEB})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
