"""KalmanBoxTracker: constant-velocity bounding-box Kalman filter, pure numpy.

State (8,): [cx, cy, aspect, height, vcx, vcy, va, vh]
  cx, cy   box center in pixels
  aspect   width / height
  height   box height in pixels
  v*       per-frame velocities of the above

Measurement (4,): [cx, cy, aspect, height] derived from a pixel box
[x1, y1, x2, y2]. No filterpy — the predict/update equations are written out
directly with numpy.

Noise scales follow the DeepSORT/ByteTrack convention: position/size stds are
proportional to the current box height, aspect stds are small constants.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6

# Std-dev weights (relative to box height) for position-like and velocity-like terms.
_STD_WEIGHT_POS = 1.0 / 20.0
_STD_WEIGHT_VEL = 1.0 / 160.0


def box_to_z(box: np.ndarray | list[float]) -> np.ndarray:
    """Pixel box [x1, y1, x2, y2] -> measurement [cx, cy, aspect, height]."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    w = max(x2 - x1, _EPS)
    h = max(y2 - y1, _EPS)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w / h, h], dtype=np.float64)


def z_to_box(z: np.ndarray) -> np.ndarray:
    """Measurement/state head [cx, cy, aspect, height] -> pixel box [x1, y1, x2, y2]."""
    cx, cy, a, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    h = max(h, _EPS)
    w = max(a, _EPS) * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over a single bounding box."""

    def __init__(self, box: np.ndarray | list[float]) -> None:
        z = box_to_z(box)

        # State and covariance.
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z

        h = z[3]
        p_std = np.array(
            [
                2.0 * _STD_WEIGHT_POS * h,
                2.0 * _STD_WEIGHT_POS * h,
                1e-2,
                2.0 * _STD_WEIGHT_POS * h,
                10.0 * _STD_WEIGHT_VEL * h,
                10.0 * _STD_WEIGHT_VEL * h,
                1e-5,
                10.0 * _STD_WEIGHT_VEL * h,
            ]
        )
        self.P = np.diag(p_std**2)

        # Constant-velocity transition (dt = 1 frame) and measurement matrices.
        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8, dtype=np.float64)

        self.age = 0  # predict() calls
        self.time_since_update = 0  # predicts since the last update()

    # ------------------------------------------------------------------ noise

    def _process_noise(self) -> np.ndarray:
        h = max(self.x[3], _EPS)
        q_std = np.array(
            [
                _STD_WEIGHT_POS * h,
                _STD_WEIGHT_POS * h,
                1e-2,
                _STD_WEIGHT_POS * h,
                _STD_WEIGHT_VEL * h,
                _STD_WEIGHT_VEL * h,
                1e-5,
                _STD_WEIGHT_VEL * h,
            ]
        )
        return np.diag(q_std**2)

    def _measurement_noise(self) -> np.ndarray:
        h = max(self.x[3], _EPS)
        r_std = np.array(
            [
                _STD_WEIGHT_POS * h,
                _STD_WEIGHT_POS * h,
                1e-1,
                _STD_WEIGHT_POS * h,
            ]
        )
        return np.diag(r_std**2)

    # ------------------------------------------------------------ predict/update

    def predict(self) -> np.ndarray:
        """Advance one frame; return the predicted box [x1, y1, x2, y2]."""
        Q = self._process_noise()
        self.x = self.F @ self.x
        # Keep the box physically valid even after long coasts.
        self.x[2] = max(self.x[2], _EPS)
        self.x[3] = max(self.x[3], _EPS)
        self.P = self.F @ self.P @ self.F.T + Q
        self.age += 1
        self.time_since_update += 1
        return self.box

    def update(self, box: np.ndarray | list[float]) -> None:
        """Fold a matched detection box into the state."""
        z = box_to_z(box)
        R = self._measurement_noise()

        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = max(self.x[2], _EPS)
        self.x[3] = max(self.x[3], _EPS)

        # Joseph-form covariance update (numerically stable, keeps P symmetric PSD).
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.time_since_update = 0

    # -------------------------------------------------------------- accessors

    @property
    def box(self) -> np.ndarray:
        """Current state as a pixel box [x1, y1, x2, y2]."""
        return z_to_box(self.x[:4])

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vcx, vcy, va, vh]."""
        return self.x[4:].copy()
