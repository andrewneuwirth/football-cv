# tests/test_geometry.py
from ml import geometry


def test_find_snap_picks_speed_spike():
    # 16 still frames (arm the detector after LOW_RUN low frames), then a
    # sustained speed spike at index 16 — that is the snap.
    speeds = [0.1] * 16 + [5.0, 5.0, 5.0, 0.2]
    idx, ok, _ = geometry.find_snap(speeds)
    assert idx == 16 and ok is True


def test_find_snap_no_motion_flags_unreliable():
    speeds = [0.1, 0.1, 0.1, 0.1]
    idx, ok, _ = geometry.find_snap(speeds)
    assert ok is False
