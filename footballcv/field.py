"""FIELD stage: pixel->field homography per frame, tracks projected to yards.

Reads calibration.json + clip.mp4 + tracks.json from data/plays/<play_id>/ and
writes field_coords.json (see CONTRACT.md).

Frame-0 homography comes from the clicked calibration points
(cv2.findHomography). It is propagated to every later frame with sparse
optical flow: cv2.goodFeaturesToTrack on the field surface (player boxes
masked out), cv2.calcOpticalFlowPyrLK between consecutive frames, a robust
RANSAC homography/affine estimate of the inter-frame camera motion, composed
onto the running H. Features are re-seeded every ~RESEED_INTERVAL frames or
whenever fewer than MIN_TRACKED points survive. Degenerate frames (too few
points, unstable transform) reuse the previous H and are counted in the
printed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from footballcv.paths import play_dir

# Feature / flow tuning
MAX_FEATURES = 400
FEATURE_QUALITY = 0.01
FEATURE_MIN_DIST = 10
FEATURE_BLOCK = 7
RESEED_INTERVAL = 30  # re-seed features every ~30 frames
MIN_TRACKED = 20  # ... or when fewer than this survive
MIN_PAIRS_FOR_HOMOGRAPHY = 8
MIN_PAIRS_FOR_AFFINE = 3
FB_ERR_MAX = 1.5  # forward-backward error gate, px
RANSAC_REPROJ = 3.0
BOX_MASK_PAD = 8  # px padding around player boxes when masking the field

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


# ---------------------------------------------------------------- helpers


def _calibration_homography(calib: dict) -> np.ndarray:
    """Pixel->field homography from >=4 clicked calibration points."""
    pts = calib.get("points", [])
    if len(pts) < 4:
        raise ValueError(
            f"calibration.json needs >=4 points, got {len(pts)}"
        )
    src = np.array([[p["px"], p["py"]] for p in pts], dtype=np.float64)
    dst = np.array([[p["fx"], p["fy"]] for p in pts], dtype=np.float64)
    method = cv2.RANSAC if len(pts) >= 5 else 0
    H, _ = cv2.findHomography(src, dst, method, RANSAC_REPROJ)
    if H is None:
        raise ValueError("cv2.findHomography failed on calibration points")
    return H / H[2, 2]


def _boxes_by_frame(tracks: dict) -> dict[int, list[list[float]]]:
    out: dict[int, list[list[float]]] = {}
    for tr in tracks.get("tracks", []):
        for fr in tr.get("frames", []):
            out.setdefault(int(fr["i"]), []).append(list(fr["box"]))
    return out


def _field_mask(shape: tuple[int, int], boxes: list[list[float]]) -> np.ndarray:
    """255 where the field surface is (player boxes zeroed out, padded)."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        xa = max(0, int(x1) - BOX_MASK_PAD)
        ya = max(0, int(y1) - BOX_MASK_PAD)
        xb = min(w, int(x2) + BOX_MASK_PAD + 1)
        yb = min(h, int(y2) + BOX_MASK_PAD + 1)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 0
    return mask


def _seed_features(gray: np.ndarray, boxes: list[list[float]]) -> np.ndarray | None:
    mask = _field_mask(gray.shape[:2], boxes)
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_FEATURES,
        qualityLevel=FEATURE_QUALITY,
        minDistance=FEATURE_MIN_DIST,
        blockSize=FEATURE_BLOCK,
        mask=mask,
    )
    return pts  # (N,1,2) float32 or None


def _flow_pairs(
    prev_gray: np.ndarray, gray: np.ndarray, prev_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """LK forward flow with a forward-backward consistency check."""
    cur_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
    if cur_pts is None or st is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, cur_pts, None, **LK_PARAMS)
    good = st.reshape(-1) == 1
    if back_pts is not None and st_b is not None:
        fb_err = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        good &= (st_b.reshape(-1) == 1) & (fb_err < FB_ERR_MAX)
    return prev_pts.reshape(-1, 2)[good], cur_pts.reshape(-1, 2)[good]


def _transform_ok(A: np.ndarray | None) -> bool:
    """Sanity gate on an inter-frame 3x3 transform (prev px -> cur px)."""
    if A is None or not np.all(np.isfinite(A)):
        return False
    if abs(A[2, 2]) < 1e-9:
        return False
    A = A / A[2, 2]
    det2 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    # Consecutive frames: near-identity scale; reject collapses/explosions.
    if not (0.25 < abs(det2) < 4.0):
        return False
    if abs(np.linalg.det(A)) < 1e-9:
        return False
    return True


def _estimate_interframe(p0: np.ndarray, p1: np.ndarray) -> np.ndarray | None:
    """Robust prev->cur pixel transform as a 3x3 matrix (RANSAC), or None."""
    if len(p0) >= MIN_PAIRS_FOR_HOMOGRAPHY:
        H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, RANSAC_REPROJ)
        if _transform_ok(H) and inl is not None and int(inl.sum()) >= 6:
            return H / H[2, 2]
    if len(p0) >= MIN_PAIRS_FOR_AFFINE:
        M, inl = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ
        )
        if M is not None:
            A = np.vstack([M, [0.0, 0.0, 1.0]])
            if _transform_ok(A) and inl is not None and int(inl.sum()) >= 3:
                return A
    return None


def _norm(H: np.ndarray) -> np.ndarray:
    """Normalize a homography so H[2,2] == 1 (numerical hygiene when chaining)."""
    return H / H[2, 2]


def _project(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    v = H @ np.array([px, py, 1.0])
    if abs(v[2]) < 1e-12:
        return float("nan"), float("nan")
    return float(v[0] / v[2]), float(v[1] / v[2])


# ---------------------------------------------------------------- stage


def run(play_id: str, data_dir: Path) -> None:
    pdir = play_dir(play_id, Path(data_dir))
    calib = json.loads((pdir / "calibration.json").read_text())
    tracks = json.loads((pdir / "tracks.json").read_text())
    clip = pdir / "clip.mp4"

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {clip}")

    boxes_at = _boxes_by_frame(tracks)
    H_cal = _calibration_homography(calib)  # pixel -> field AT THE CALIBRATION FRAME
    # Pass 1: inter-frame camera transforms A_i (frame i-1 px -> frame i px);
    # None means degenerate (treated as identity). The homography is anchored
    # at the frame the user actually calibrated on and composed outward BOTH
    # ways from there — anchoring at frame 0 regardless (the old behavior)
    # silently corrupts every mapping when calibration happened mid-clip.
    inter: list[np.ndarray | None] = []

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"{clip} has no frames")
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prev_pts = _seed_features(prev_gray, boxes_at.get(0, []))
    frames_since_seed = 0
    n_degenerate = 0
    n_reseeds = 0

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        A = None
        if prev_pts is not None and len(prev_pts) >= MIN_PAIRS_FOR_AFFINE:
            p0, p1 = _flow_pairs(prev_gray, gray, prev_pts)
            A = _estimate_interframe(p0, p1)
            # Surviving points become next frame's seeds.
            prev_pts = p1.reshape(-1, 1, 2).astype(np.float32) if len(p1) else None
        else:
            prev_pts = None

        if A is None:
            n_degenerate += 1  # identity: reuse neighboring H
        inter.append(A)

        frames_since_seed += 1
        n_tracked = 0 if prev_pts is None else len(prev_pts)
        if frames_since_seed >= RESEED_INTERVAL or n_tracked < MIN_TRACKED:
            prev_pts = _seed_features(gray, boxes_at.get(i, []))
            frames_since_seed = 0
            n_reseeds += 1
        prev_gray = gray

    cap.release()

    # Pass 2: compose homographies outward from the calibration anchor frame.
    n_frames = len(inter) + 1
    anchor = int(calib.get("frame", 0))
    anchor = max(0, min(anchor, n_frames - 1))
    homographies: list[np.ndarray | None] = [None] * n_frames
    homographies[anchor] = H_cal.copy()
    # forward: H_i maps frame-i px -> field; cur = A @ prev => H_cur = H_prev @ A^-1
    for fi in range(anchor + 1, n_frames):
        A = inter[fi - 1]
        Hp = homographies[fi - 1]
        homographies[fi] = Hp if A is None else _norm(Hp @ np.linalg.inv(A))
    # backward: prev = A^-1 @ cur is wrong way round; H_prev = H_cur @ A
    for fi in range(anchor - 1, -1, -1):
        A = inter[fi]
        Hn = homographies[fi + 1]
        homographies[fi] = Hn if A is None else _norm(Hn @ A)

    # Project every track's ground point (bottom-center) through its frame's H.
    out_tracks = []
    for tr in tracks.get("tracks", []):
        frames_out = []
        for fr in tr.get("frames", []):
            fi = int(fr["i"])
            x1, y1, x2, y2 = fr["box"]
            Hi = homographies[min(fi, n_frames - 1)]
            fx, fy = _project(Hi, (x1 + x2) / 2.0, y2)
            frames_out.append({"i": fi, "x": fx, "y": fy})
        out_tracks.append({"track_id": tr["track_id"], "frames": frames_out})

    out = {
        "play_id": play_id,
        "homographies": [
            {"i": idx, "H": [[float(v) for v in row] for row in Hm]}
            for idx, Hm in enumerate(homographies)
        ],
        "tracks": out_tracks,
    }
    (pdir / "field_coords.json").write_text(json.dumps(out))

    print(
        f"[field] {play_id}: {n_frames} frames, "
        f"{n_degenerate} degenerate (reused previous H), {n_reseeds} re-seeds, "
        f"{len(out_tracks)} tracks projected"
    )
