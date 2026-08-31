# tests/test_paths.py
from pathlib import Path
from footballcv import paths

def test_play_dir_under_plays_dir():
    base = Path("/tmp/fcv-test")
    assert paths.play_dir("game1_p01", base) == paths.plays_dir(base) / "game1_p01"
