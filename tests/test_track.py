# tests/test_track.py
import numpy as np
from ml import track

def test_iou_identical_boxes_is_one():
    b = [0.0, 0.0, 10.0, 10.0]
    assert abs(track.iou(b, b) - 1.0) < 1e-9

def test_iou_disjoint_boxes_is_zero():
    assert track.iou([0,0,1,1], [5,5,6,6]) == 0.0

def test_iou_matrix_shape():
    A = np.array([[0,0,1,1],[0,0,2,2]], float)
    B = np.array([[0,0,1,1]], float)
    assert track.iou_matrix(A, B).shape == (2, 1)
