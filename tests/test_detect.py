# tests/test_detect.py
import importlib.util

import numpy as np
import pytest

from footballcv import detect

def test_letterbox_params_scales_to_square():
    scale, pad_x, pad_y, new_w, new_h = detect.letterbox_params(1920, 1080, 640)
    assert 0 < scale <= 1
    assert new_w <= 640 and new_h <= 640

def test_boxes_to_native_inverts_letterbox():
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
    lb = detect.boxes_to_letterbox(boxes.copy(), 0.5, 5.0, 6.0)
    assert lb.shape == boxes.shape

def test_missing_detect_extra_raises_actionable_error(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name in ("torch", "ultralytics") else real(name),
    )
    with pytest.raises(RuntimeError, match=r"\.\[detect\]"):
        detect._require_detect_deps()
