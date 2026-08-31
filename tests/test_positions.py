# tests/test_positions.py — exercises pure helpers of the position classifier.
from ml import positions


def test_median():
    assert positions._median([3.0, 1.0, 2.0]) == 2.0
    assert positions._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert positions._median([]) == 0.0


def test_find_snap_detects_spike_after_still():
    # a long still stretch then a sustained speed spike = the snap
    speeds = [0.1] * 16 + [5.0, 5.0, 5.0, 0.2]
    idx, confident, _low = positions._find_snap(speeds)
    assert confident is True
    assert idx == 16


def test_find_snap_no_motion_is_unreliable():
    idx, confident, _low = positions._find_snap([0.1, 0.1, 0.1, 0.1])
    assert confident is False


def test_confident_color_thresholds():
    colors = {1: ("A", 0.9), 2: ("B", 0.5), 3: ("ref", 1.0)}
    assert positions._confident_color(colors, 1) == "A"   # confident A
    assert positions._confident_color(colors, 2) is None  # below the color-conf bar
    assert positions._confident_color(colors, 3) is None  # official, not a team
    assert positions._confident_color(colors, 99) is None  # unknown track


def test_module_exposes_run():
    assert callable(positions.run)
