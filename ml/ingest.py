"""INGEST stage: bring your own film into the pipeline.

Copies a video file to the location every downstream stage reads from —
``<data_dir>/plays/<play_id>/clip.mp4`` — creating the play directory if
needed. This is the "upload film" entry point for the CLI:

    python -m ml.cli ingest my_play --video path/to/film.mp4 --data ./data

After ingest, run the rest of the pipeline on that play id:

    python -m ml.cli all my_play --data ./data
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ml.paths import play_dir


def run(play_id: str, data_dir: Path, video: str | Path | None = None) -> Path:
    """Copy ``video`` to ``<data_dir>/plays/<play_id>/clip.mp4``.

    Returns the destination path. Raises FileNotFoundError if ``video`` is
    missing or None.
    """
    if video is None:
        raise FileNotFoundError(
            "ingest needs a video: pass --video path/to/film.mp4"
        )
    src = Path(video)
    if not src.is_file():
        raise FileNotFoundError(f"no such video file: {src}")

    pdir = play_dir(play_id, data_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / "clip.mp4"
    shutil.copyfile(src, dest)
    print(f"ingest[{play_id}]: {src} -> {dest}", flush=True)
    return dest
