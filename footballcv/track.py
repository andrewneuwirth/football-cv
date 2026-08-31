"""ByteTrack-style multi-object tracker over detections.json -> tracks.json.

Two-pass association per frame, pure numpy/scipy:

1. Predict every live track one frame forward (Kalman, constant velocity).
2. Pass 1: high-confidence detections (conf >= high_conf) vs all live tracks,
   Hungarian assignment (scipy linear_sum_assignment) on cost 1 - IoU, gated
   at IoU >= iou_gate.
3. Pass 2: remaining low-confidence detections (low_conf <= conf < high_conf)
   vs still-unmatched *confirmed* tracks, same assignment + gate.
4. Unmatched confirmed tracks coast on the Kalman prediction (frames marked
   ``interpolated: true``) for up to ``max_coast`` frames, then retire.
   Unmatched probationary tracks are dropped immediately.
5. Unmatched high-confidence detections start new tracks; a track is emitted
   only after ``probation`` consecutive matched frames (its buffered
   probation frames are then included). Track IDs are positive ints assigned
   in confirmation order and never reused.

Trailing coasted frames of a track that retires without re-matching are
trimmed, so ``interpolated: true`` frames always sit between real detections.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from footballcv.kalman import KalmanBoxTracker
from footballcv.paths import play_dir

# ---------------------------------------------------------------------- IoU


def iou(box_a: np.ndarray | list[float], box_b: np.ndarray | list[float]) -> float:
    """IoU of two pixel boxes [x1, y1, x2, y2]. Degenerate boxes -> 0."""
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h

    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. boxes_a: (N, 4), boxes_b: (M, 4) -> (N, M) float64."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    inter_w = np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])
    inter_h = np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])
    inter = np.clip(inter_w, 0.0, None) * np.clip(inter_h, 0.0, None)

    area_a = np.clip(a[..., 2] - a[..., 0], 0.0, None) * np.clip(a[..., 3] - a[..., 1], 0.0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0.0, None) * np.clip(b[..., 3] - b[..., 1], 0.0, None)
    union = area_a + area_b - inter

    out = np.zeros((n, m), dtype=np.float64)
    np.divide(inter, union, out=out, where=union > 0.0)
    return out


# --------------------------------------------------------------- association

_GATED_COST = 1e6  # cost assigned to sub-gate pairs so the solver avoids them


def associate(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    iou_gate: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Globally-optimal gated IoU assignment of tracks to detections.

    Pairs below ``iou_gate`` are masked to a large constant cost BEFORE
    linear_sum_assignment (SORT/ByteTrack convention), so the solver never
    spends a track on a below-gate detection at the expense of valid
    within-gate pairs; the post-solve filter then only strips forced
    assignments to masked cells. Returns (matches, unmatched_track_indices,
    unmatched_det_indices) where matches are (track_idx, det_idx) pairs.
    """
    track_boxes = np.asarray(track_boxes, dtype=np.float64).reshape(-1, 4)
    det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    n, m = track_boxes.shape[0], det_boxes.shape[0]
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    ious = iou_matrix(track_boxes, det_boxes)
    cost = np.where(ious >= iou_gate, 1.0 - ious, _GATED_COST)
    rows, cols = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_d: set[int] = set()
    for r, c in zip(rows, cols):
        if ious[r, c] >= iou_gate:
            matches.append((int(r), int(c)))
            matched_t.add(int(r))
            matched_d.add(int(c))

    unmatched_tracks = [i for i in range(n) if i not in matched_t]
    unmatched_dets = [j for j in range(m) if j not in matched_d]
    return matches, unmatched_tracks, unmatched_dets


# -------------------------------------------------------------------- tracks


class _Track:
    """Internal per-object bookkeeping around one KalmanBoxTracker."""

    def __init__(self, box: list[float], frame_i: int, probation: int) -> None:
        self.kf = KalmanBoxTracker(box)
        self.track_id: int | None = None  # assigned on confirmation, never reused
        self.hits = 1  # consecutive matched frames
        self.coast = 0  # consecutive coasted frames
        self.confirmed = probation <= 1
        self.frames: list[dict] = [
            {"i": frame_i, "box": [float(v) for v in box], "interpolated": False}
        ]

    def mark_matched(self, box: list[float], frame_i: int, probation: int) -> None:
        self.kf.update(box)
        self.hits += 1
        self.coast = 0
        self.frames.append({"i": frame_i, "box": [float(v) for v in box], "interpolated": False})
        if not self.confirmed and self.hits >= probation:
            self.confirmed = True

    def mark_coasted(self, predicted_box: np.ndarray, frame_i: int) -> None:
        self.hits = 0
        self.coast += 1
        self.frames.append(
            {"i": frame_i, "box": [float(v) for v in predicted_box], "interpolated": True}
        )

    def emit(self) -> dict:
        """Output form, with trailing coasted frames trimmed."""
        frames = list(self.frames)
        while frames and frames[-1]["interpolated"]:
            frames.pop()
        return {"track_id": self.track_id, "frames": frames}


# ------------------------------------------------------------------- seeding

# A detection overlapping an existing track's box this much (in that frame) is
# considered claimed by that track and is never given to a seeded track.
_CLAIM_IOU = 0.6


def _associate_from_seed(
    seed_box: list[float],
    frame_indices: list[int],
    det_frames: dict[int, np.ndarray],
    claimed_index: dict[int, np.ndarray],
    low_conf: float,
    iou_gate: float,
    max_coast: int,
) -> list[dict]:
    """Greedily associate a single fresh Kalman track from ``seed_box``.

    ``frame_indices`` is the walk order (ascending for forward, descending for
    backward), excluding the seed frame itself. Per step: predict, take the
    best unclaimed detection with conf >= low_conf and IoU >= iou_gate, else
    coast (interpolated). Stops after ``max_coast`` consecutive coasts;
    trailing coasted frames (in walk order) are trimmed.
    """
    kf = KalmanBoxTracker(seed_box)
    out: list[dict] = []
    coast = 0
    for i in frame_indices:
        predicted = kf.predict()
        best_box: np.ndarray | None = None
        best_iou = 0.0
        boxes = det_frames.get(i)
        if boxes is not None and boxes.shape[0]:
            candidates = boxes[boxes[:, 4] >= low_conf][:, :4]
            track_boxes = claimed_index.get(i)
            for b in candidates:
                if (
                    track_boxes is not None
                    and track_boxes.shape[0]
                    and iou_matrix(b.reshape(1, 4), track_boxes).max() > _CLAIM_IOU
                ):
                    continue  # claimed by an existing track
                v = iou(predicted, b)
                if v >= iou_gate and v > best_iou:
                    best_iou = v
                    best_box = b
        if best_box is not None:
            kf.update(best_box)
            coast = 0
            out.append(
                {"i": int(i), "box": [float(v) for v in best_box], "interpolated": False}
            )
        else:
            coast += 1
            if coast > max_coast:
                break
            out.append(
                {"i": int(i), "box": [float(v) for v in predicted], "interpolated": True}
            )
    while out and out[-1]["interpolated"]:
        out.pop()
    return out


def seed_track(
    play_id: str,
    data_dir: Path,
    frame: int,
    px: float,
    py: float,
    low_conf: float = 0.1,
    iou_gate: float = 0.2,
    max_coast: int = 30,
) -> int:
    """Human-vouched track birth from a click at ``(px, py)`` in ``frame``.

    Finds the smallest detection box (conf >= low_conf) containing the click,
    seeds a fresh Kalman track there, and associates it forward AND backward
    through detections.json — ignoring the high-conf birth threshold but never
    taking detections already claimed by existing tracks (IoU > 0.6 vs any
    existing track box in that frame). Appends the result to tracks.json as a
    new track_id (max existing + 1) without touching existing tracks, and
    returns the new id. Raises KeyError if no detection contains the click.
    """
    pdir = play_dir(play_id, data_dir)
    detections = json.loads((pdir / "detections.json").read_text())
    tracks_path = pdir / "tracks.json"
    if tracks_path.is_file():
        tracks_doc = json.loads(tracks_path.read_text())
    else:
        tracks_doc = {"play_id": play_id, "tracks": []}
    existing = tracks_doc.get("tracks") or []

    det_frames: dict[int, np.ndarray] = {
        int(f["i"]): np.asarray(f.get("boxes") or [], dtype=np.float64).reshape(-1, 5)
        for f in detections["frames"]
    }

    # 1. The clicked box: smallest detection containing (px, py), any conf >= low_conf.
    frame = int(frame)
    px, py = float(px), float(py)
    clicked: np.ndarray | None = None
    clicked_area = np.inf
    for b in det_frames.get(frame, np.zeros((0, 5))):
        if b[4] < low_conf:
            continue
        if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
            area = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
            if area < clicked_area:
                clicked_area = area
                clicked = b[:4]
    if clicked is None:
        raise KeyError(
            f"no detection with conf >= {low_conf} contains ({px}, {py}) in frame {frame}"
        )
    seed_box = [float(v) for v in clicked]

    # 2. Per-frame index of existing-track boxes (the "claimed" index).
    claimed_lists: dict[int, list[list[float]]] = {}
    for tr in existing:
        for f in tr["frames"]:
            claimed_lists.setdefault(int(f["i"]), []).append(f["box"])
    claimed_index = {
        i: np.asarray(v, dtype=np.float64).reshape(-1, 4) for i, v in claimed_lists.items()
    }

    # 3. Associate forward then backward from the clicked frame.
    all_frames = sorted(det_frames)
    fwd_idx = [i for i in all_frames if i > frame]
    bwd_idx = [i for i in reversed(all_frames) if i < frame]
    fwd = _associate_from_seed(
        seed_box, fwd_idx, det_frames, claimed_index, low_conf, iou_gate, max_coast
    )
    bwd = _associate_from_seed(
        seed_box, bwd_idx, det_frames, claimed_index, low_conf, iou_gate, max_coast
    )

    frames = (
        list(reversed(bwd))
        + [{"i": frame, "box": seed_box, "interpolated": False}]
        + fwd
    )
    frames.sort(key=lambda f: f["i"])

    # 4. Append as a new track and write atomically; existing tracks untouched.
    ids = [int(tr["track_id"]) for tr in existing]
    new_id = (max(ids) + 1) if ids else 1
    tracks_doc.setdefault("tracks", []).append({"track_id": new_id, "frames": frames})

    tmp = tracks_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracks_doc, indent=1))
    tmp.replace(tracks_path)
    return new_id


# ----------------------------------------------------------------------- run


def run(
    play_id: str,
    data_dir: Path,
    # 0.4, not ByteTrack's usual 0.5: on 720p night HS film, engaged linemen
    # score 0.35-0.5 and at 0.5 never birth tracks at all. Measured on real
    # film: 0.4 nearly doubles full-play track coverage; 0.35 fragments
    # (phantom tracks start competing with real ones).
    high_conf: float = 0.4,
    low_conf: float = 0.1,
    iou_gate: float = 0.2,
    max_coast: int = 30,
    probation: int = 3,
) -> None:
    """Read detections.json for ``play_id``, write tracks.json."""
    pdir = play_dir(play_id, data_dir)
    det_path = pdir / "detections.json"
    detections = json.loads(det_path.read_text())

    frames = sorted(detections["frames"], key=lambda f: f["i"])

    live: list[_Track] = []
    finished: list[_Track] = []
    next_id = 1

    for frame in frames:
        i = int(frame["i"])
        boxes = np.asarray(frame.get("boxes") or [], dtype=np.float64).reshape(-1, 5)
        confs = boxes[:, 4]
        high = boxes[confs >= high_conf][:, :4]
        low = boxes[(confs >= low_conf) & (confs < high_conf)][:, :4]

        # 1. Predict every live track one frame forward.
        predicted = np.array([t.kf.predict() for t in live]).reshape(-1, 4)

        # 2. Pass 1: high-conf detections vs all live tracks.
        matches, um_tracks, um_high = associate(predicted, high, iou_gate)
        for ti, di in matches:
            live[ti].mark_matched(list(high[di]), i, probation)

        # 3. Pass 2: leftover low-conf detections vs unmatched *confirmed* tracks.
        second_idx = [ti for ti in um_tracks if live[ti].confirmed]
        matches2, um_second, _um_low = associate(
            predicted[second_idx] if second_idx else np.zeros((0, 4)),
            low,
            iou_gate,
        )
        for si, di in matches2:
            live[second_idx[si]].mark_matched(list(low[di]), i, probation)
        second_matched = {second_idx[si] for si, _ in matches2}

        # 4. Unmatched tracks: confirmed ones coast (up to max_coast), then
        #    retire; probationary ones are dropped immediately.
        survivors: list[_Track] = []
        unmatched = [ti for ti in um_tracks if ti not in second_matched]
        for ti, t in enumerate(live):
            if ti not in unmatched:
                survivors.append(t)
                continue
            if not t.confirmed:
                continue  # probation broken: discard silently
            if t.coast + 1 > max_coast:
                finished.append(t)  # retire without recording another coast
                continue
            t.mark_coasted(predicted[ti], i)
            survivors.append(t)
        live = survivors

        # 5. Unmatched high-conf detections start new probationary tracks.
        for di in um_high:
            live.append(_Track(list(high[di]), i, probation))

        # Assign IDs to newly confirmed tracks (in stable order, never reused).
        for t in live:
            if t.confirmed and t.track_id is None:
                t.track_id = next_id
                next_id += 1

    finished.extend(live)

    tracks = [t.emit() for t in finished if t.confirmed and t.track_id is not None]
    tracks.sort(key=lambda tr: tr["track_id"])

    out = {"play_id": play_id, "tracks": tracks}
    (pdir / "tracks.json").write_text(json.dumps(out, indent=1))
