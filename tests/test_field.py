# tests/test_field.py
import numpy as np
from footballcv import field

def test_project_identity_homography():
    H = np.eye(3)
    x, y = field._project(H, 12.0, 34.0)
    assert abs(x - 12.0) < 1e-6 and abs(y - 34.0) < 1e-6
