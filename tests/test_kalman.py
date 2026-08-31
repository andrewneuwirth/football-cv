# tests/test_kalman.py
import numpy as np
from footballcv import kalman

def test_box_to_z_roundtrip():
    box = np.array([10.0, 20.0, 30.0, 50.0])
    z = kalman.box_to_z(box)
    back = kalman.z_to_box(z)
    assert np.allclose(back, box, atol=1e-6)

def test_tracker_predicts_constant_position():
    t = kalman.KalmanBoxTracker(np.array([0.0, 0.0, 10.0, 10.0]))
    pred = t.predict()
    assert pred is not None
