"""AUTOCAL: auto-calibrate a play by TRANSFERRING calibration from another play
in the same game (zero clicks).

The sideline camera is nearly fixed within a game: it only pans/zooms to follow
the play, so the yard lines, hashes, numbers and painted logos sit in almost the
same pixels every snap. If one play in the game was calibrated by hand, we can
register a target play's frame to that reference frame purely on the STATIC FIELD
and compose homographies to obtain the target's pixel->field calibration.

Pipeline (see :func:`transfer`):

  1. Reference manual calibration:   H_ref  (pixel_ref -> field yards), from
     the reference play's calibration.json clicked points (cv2.findHomography).
  2. Grab the reference frame image (calibration.json "frame") and a target
     frame (target snap.json frame if present, else mid-clip).
  3. Feature-match target -> reference on the field ONLY: players are masked out
     (tracks.json / detections.json boxes, dilated) so moving bodies and the
     sideline crowd cannot corrupt the match; the scoreboard/top band is masked
     too. SIFT if the OpenCV build has it, else ORB with many features. A RANSAC
     homography H_match (pixel_target -> pixel_ref) is fit to the good matches;
     too few inliers / poor quality -> return None (fall back to manual).
  4. Compose:   H_target = H_ref @ H_match   (pixel_target -> field).

`transfer` returns a *proposal* dict; it never overwrites the target's own
manual calibration.json.

This module is self-contained and read-only w.r.t. every other ml module.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from footballcv.paths import play_dir

# ----------------------------------------------------------------- tuning

BOX_MASK_PAD = 18          # px dilation around player boxes (bigger than field.py:
                           # players move between plays, so be generous)
TOP_BAND_FRAC = 0.06       # mask the top ~6% (scoreboard / broadcast band)
MAX_ORB_FEATURES = 6000    # plenty of ORB features when SIFT is unavailable
SIFT_FEATURES = 4000
LOWE_RATIO = 0.75          # ratio test for the two-NN matcher
RANSAC_REPROJ = 4.0        # px reprojection threshold for H_match RANSAC
MIN_GOOD_MATCHES = 12      # need at least this many ratio-passing matches
MIN_INLIERS = 25           # ... and this many RANSAC inliers to trust the fit
MIN_INLIER_FRAC = 0.30     # inliers / good_matches must clear this


# ------------------------------------------------------------- small io


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _reference_homography(calib: dict) -> np.ndarray | None:
    """pixel_ref -> field yards from >=4 clicked points (same math as field.py)."""
    pts = calib.get("points", [])
    if len(pts) < 4:
        return None
    src = np.array([[p["px"], p["py"]] for p in pts], dtype=np.float64)
    dst = np.array([[p["fx"], p["fy"]] for p in pts], dtype=np.float64)
    method = cv2.RANSAC if len(pts) >= 5 else 0
    H, _ = cv2.findHomography(src, dst, method, RANSAC_REPROJ)
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _read_frame(clip: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = frame_idx
    if n > 0:
        idx = max(0, min(frame_idx, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# --------------------------------------------------------- player masking


def _boxes_at_frame(pdir: Path, frame_idx: int) -> list[list[float]]:
    """Player boxes at a frame, preferring tracks.json then detections.json.

    If nothing has a box for that exact frame (e.g. a snap frame that a short
    track doesn't reach), fall back to the union of every box in the play so the
    mask still covers where players tend to be.
    """
    tracks = _load_json(pdir / "tracks.json")
    boxes: list[list[float]] = []
    union: list[list[float]] = []
    if tracks:
        for tr in tracks.get("tracks", []):
            for fr in tr.get("frames", []):
                box = [float(v) for v in fr["box"][:4]]
                union.append(box)
                if int(fr["i"]) == frame_idx:
                    boxes.append(box)
        if boxes:
            return boxes
    dets = _load_json(pdir / "detections.json")
    if dets:
        for fr in dets.get("frames", []):
            if int(fr["i"]) == frame_idx:
                for b in fr.get("boxes", []):
                    boxes.append([float(v) for v in b[:4]])
        if boxes:
            return boxes
    return union  # may be empty; caller handles that


def _field_mask(shape: tuple[int, int], boxes: list[list[float]]) -> np.ndarray:
    """255 on the static field, 0 over players and the top broadcast band."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    band = int(h * TOP_BAND_FRAC)
    if band > 0:
        mask[:band, :] = 0
    for x1, y1, x2, y2 in boxes:
        xa = max(0, int(x1) - BOX_MASK_PAD)
        ya = max(0, int(y1) - BOX_MASK_PAD)
        xb = min(w, int(x2) + BOX_MASK_PAD + 1)
        yb = min(h, int(y2) + BOX_MASK_PAD + 1)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 0
    return mask


# ----------------------------------------------------------- matching


def _make_detector() -> tuple[object, int]:
    """SIFT if this OpenCV build has it, else ORB. Returns (detector, norm)."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=SIFT_FEATURES), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=MAX_ORB_FEATURES), cv2.NORM_HAMMING


def _match_features(
    tgt_gray: np.ndarray,
    ref_gray: np.ndarray,
    tgt_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object, object, list]:
    """Ratio-tested good matches between target and reference field pixels.

    Returns (tgt_pts, ref_pts, kp_tgt, kp_ref, good_matches). Points are the
    matched keypoint coordinates as (N,1,2) float32 arrays (empty if too few).
    """
    detector, norm = _make_detector()
    kp_t, des_t = detector.detectAndCompute(tgt_gray, tgt_mask)
    kp_r, des_r = detector.detectAndCompute(ref_gray, ref_mask)
    empty = np.empty((0, 1, 2), np.float32)
    if des_t is None or des_r is None or len(kp_t) < 2 or len(kp_r) < 2:
        return empty, empty, kp_t or [], kp_r or [], []

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(des_t, des_r, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)
    if len(good) < MIN_GOOD_MATCHES:
        return empty, empty, kp_t, kp_r, good

    tgt_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return tgt_pts, ref_pts, kp_t, kp_r, good


# ------------------------------------------------------------- public api


def _resolve_frames(ref_pdir: Path, tgt_pdir: Path, ref_calib: dict) -> tuple[int, int]:
    ref_frame = int(ref_calib.get("frame", 0))
    snap = _load_json(tgt_pdir / "snap.json")
    if snap and "frame" in snap:
        tgt_frame = int(snap["frame"])
    else:
        meta = _load_json(tgt_pdir / "meta.json") or {}
        n = int(meta.get("n_frames", 0))
        tgt_frame = n // 2 if n > 0 else 0
    return ref_frame, tgt_frame


def transfer(
    target_play: str,
    reference_play: str,
    data_dir: Path,
    *,
    debug_prefix: str | None = None,
) -> dict | None:
    """Transfer calibration from ``reference_play`` onto ``target_play``.

    Returns a proposal dict (does NOT touch the target's calibration.json)::

        {"H": [[...],[...],[...]],   # pixel_target -> field yards, nested lists
         "reference": reference_play,
         "frame": target_frame,      # frame the transfer was computed on
         "reference_frame": ref_frame,
         "n_inliers": int,
         "n_matches": int,
         "quality": float}           # inlier fraction

    or ``None`` if the two frames are unmatchable (fall back to manual clicks).
    If ``debug_prefix`` is given, an inlier-match visualization is written to
    ``<debug_prefix>.png``.
    """
    data_dir = Path(data_dir)
    ref_pdir = play_dir(reference_play, data_dir)
    tgt_pdir = play_dir(target_play, data_dir)

    ref_calib = _load_json(ref_pdir / "calibration.json")
    if not ref_calib:
        return None
    H_ref = _reference_homography(ref_calib)
    if H_ref is None:
        return None

    ref_frame, tgt_frame = _resolve_frames(ref_pdir, tgt_pdir, ref_calib)
    ref_img = _read_frame(ref_pdir / "clip.mp4", ref_frame)
    tgt_img = _read_frame(tgt_pdir / "clip.mp4", tgt_frame)
    if ref_img is None or tgt_img is None:
        return None

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
    ref_mask = _field_mask(ref_gray.shape[:2], _boxes_at_frame(ref_pdir, ref_frame))
    tgt_mask = _field_mask(tgt_gray.shape[:2], _boxes_at_frame(tgt_pdir, tgt_frame))

    tgt_pts, ref_pts, kp_t, kp_r, good = _match_features(
        tgt_gray, ref_gray, tgt_mask, ref_mask
    )
    if len(tgt_pts) < MIN_GOOD_MATCHES:
        return None

    H_match, inl = cv2.findHomography(tgt_pts, ref_pts, cv2.RANSAC, RANSAC_REPROJ)
    if H_match is None or inl is None or not np.all(np.isfinite(H_match)):
        return None
    inl = inl.reshape(-1).astype(bool)
    n_inliers = int(inl.sum())
    n_matches = len(good)
    quality = n_inliers / max(1, n_matches)
    if n_inliers < MIN_INLIERS or quality < MIN_INLIER_FRAC:
        return None
    if abs(H_match[2, 2]) < 1e-12:
        return None
    H_match = H_match / H_match[2, 2]

    H_target = H_ref @ H_match            # pixel_target -> field
    if not np.all(np.isfinite(H_target)) or abs(H_target[2, 2]) < 1e-12:
        return None
    H_target = H_target / H_target[2, 2]

    if debug_prefix is not None:
        _save_debug_match(
            tgt_img, ref_img, kp_t, kp_r, good, inl,
            tgt_frame, ref_frame, n_inliers, n_matches, Path(f"{debug_prefix}.png"),
        )

    return {
        "H": [[float(v) for v in row] for row in H_target],
        "reference": reference_play,
        "frame": tgt_frame,
        "reference_frame": ref_frame,
        "n_inliers": n_inliers,
        "n_matches": n_matches,
        "quality": float(quality),
    }


# --------------------------------------------------------- debug viz


def _save_debug_match(
    tgt_img, ref_img, kp_t, kp_r, good, inl,
    tgt_frame, ref_frame, n_inliers, n_matches, out_path: Path,
) -> None:
    try:
        inlier_matches = [m for m, keep in zip(good, inl) if keep]
        vis = cv2.drawMatches(
            tgt_img, kp_t, ref_img, kp_r, inlier_matches, None,
            matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        label = (
            f"target f{tgt_frame}  ->  reference f{ref_frame}   "
            f"inliers {n_inliers}/{n_matches}"
        )
        cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
    except cv2.error:
        pass


# ------------------------------------------------- orchestration (write cal)

def _game_key(play_id: str) -> str:
    """Same-game grouping key: everything up to the trailing play number."""
    import re
    m = re.match(r"(.*?)-?\d+$", play_id)
    return m.group(1) if m else play_id


def _play_num(play_id: str) -> int:
    import re
    m = re.search(r"(\d+)$", play_id)
    return int(m.group(1)) if m else 0


def _H_to_calibration(H: list[list[float]], frame: int, w: int, h: int) -> dict:
    """Turn a pixel->field homography into a calibration.json (4 synthetic point
    correspondences well spread over the frame; findHomography recovers H)."""
    Hn = np.array(H, dtype=np.float64)
    pts = [(w * 0.2, h * 0.35), (w * 0.8, h * 0.35),
           (w * 0.2, h * 0.85), (w * 0.8, h * 0.85)]
    out = []
    for px, py in pts:
        v = Hn @ np.array([px, py, 1.0])
        out.append({"px": float(px), "py": float(py),
                    "fx": float(v[0] / v[2]), "fy": float(v[1] / v[2])})
    return {"frame": int(frame), "points": out, "source": "autocal"}


def autocalibrate(target_play: str, data_dir: Path, max_refs: int = 14) -> dict | None:
    """Auto-calibrate ``target_play`` by transferring from the best-matching
    manually-calibrated play in the same game. Writes calibration.json on
    success and returns info; returns None if nothing transfers well enough.
    Never overwrites a manually-authored calibration (source != 'autocal')."""
    data_dir = Path(data_dir)
    plays_root = data_dir / "plays"
    game = _game_key(target_play)
    tnum = _play_num(target_play)

    # candidate references: same game, has a MANUAL calibration, not the target;
    # try nearest-by-play-number first (nearby plays => nearby camera).
    refs = []
    for pdir in plays_root.iterdir():
        pid = pdir.name
        if pid == target_play or _game_key(pid) != game:
            continue
        calib = _load_json(pdir / "calibration.json")
        if calib and calib.get("source") != "autocal":
            refs.append(pid)
    refs.sort(key=lambda p: abs(_play_num(p) - tnum))
    refs = refs[:max_refs]
    if not refs:
        return None

    best = None
    for ref in refs:
        try:
            prop = transfer(target_play, ref, data_dir)
        except Exception:
            prop = None
        if prop and (best is None or prop["n_inliers"] > best["n_inliers"]):
            best = prop
    if not best:
        return None

    tgt_pdir = play_dir(target_play, data_dir)
    meta = _load_json(tgt_pdir / "meta.json") or {}
    w = int(meta.get("width", 1920)); h = int(meta.get("height", 1080))
    calib = _H_to_calibration(best["H"], best["frame"], w, h)
    with open(tgt_pdir / "calibration.json", "w") as f:
        json.dump(calib, f, indent=1)
    print(f"[autocal] {target_play}: calibrated from {best['reference']} "
          f"({best['n_inliers']} inliers, quality {best['quality']:.2f})")
    return {"reference": best["reference"], "n_inliers": best["n_inliers"],
            "quality": best["quality"], "tried": len(refs)}


def run(play_id: str, data_dir: Path) -> None:
    """Stage entry point (server background runner). Raises if nothing transfers
    so the dashboard can tell the user to calibrate manually."""
    result = autocalibrate(play_id, data_dir)
    if result is None:
        raise RuntimeError(
            "auto-calibrate failed: no same-game play matched well enough — "
            "calibrate this one by hand."
        )
