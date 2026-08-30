from pathlib import Path

import pytest

from ml import ingest, paths


def test_ingest_copies_video_to_clip(tmp_path):
    src = tmp_path / "myfilm.mp4"
    src.write_bytes(b"fake-video-bytes")
    data = tmp_path / "data"
    dest = ingest.run("play1", data, video=src)
    assert dest == paths.play_dir("play1", data) / "clip.mp4"
    assert dest.read_bytes() == b"fake-video-bytes"


def test_ingest_without_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.run("play1", tmp_path / "data", video=None)


def test_ingest_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.run("play1", tmp_path / "data", video=tmp_path / "nope.mp4")
