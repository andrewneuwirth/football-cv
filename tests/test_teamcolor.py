import numpy as np
from ml import teamcolor


def test_green_mask_flags_green_pixels():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:, :, 1] = 200  # BGR green channel high
    mask = teamcolor.green_mask(img)
    assert mask.any()


def test_jersey_crop_returns_none_for_degenerate_box():
    frame = np.zeros((10, 10, 3), np.uint8)
    assert teamcolor.jersey_crop(frame, [5, 5, 5, 5]) is None
