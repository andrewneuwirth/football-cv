"""Learned offense-position classifier (OL/QB/RB/WR).

The geometric heuristic in ``positions.py`` can't separate QB from RB — they sit
in nearly the same field-relative spot. This module extracts alignment features
and, if a trained model is bundled, predicts the offense positions from them.
It falls back cleanly to the heuristic when the model or scikit-learn is absent,
so the core install stays dependency-free.

Train with ``python -m footballcv.train_offense`` (needs the ``[ml]`` extra).
"""
from __future__ import annotations

from pathlib import Path

_MODEL_PATH = Path(__file__).parent / "models" / "offense.joblib"

# feature names, in order (documented so train + inference never drift)
FEATURES = (
    "depth",         # |down - los|, yards off the line
    "lateral",       # |across - formation center|
    "signed_lat",    # across - center (side)
    "on_line",       # 1 if depth <= line band
    "depth_rank",    # fraction of teammates shallower
    "central_rank",  # fraction of teammates more central
    "n_deeper",      # teammates deeper
    "n_wider",       # teammates wider
    "nearest_mate",  # L1 distance to closest teammate
    "is_deepest",    # 1 if the deepest offense body
    "is_central",    # 1 if the most-central offense body
)

_LINE_BAND = 1.8


def extract(off_ids, pts, los_x):
    """(ids, X) — feature rows for the offense tracks. X is a list of lists."""
    from statistics import median

    ids = [t for t in off_ids if t in pts]
    if len(ids) < 2:
        return ids, [[0.0] * len(FEATURES) for _ in ids]
    depth = {t: abs(pts[t][0] - los_x) for t in ids}
    online = [t for t in ids if depth[t] <= _LINE_BAND]
    cy = median([pts[t][1] for t in (online or ids)])
    lat = {t: pts[t][1] - cy for t in ids}
    max_d = max(depth.values())
    min_l = min(abs(lat[t]) for t in ids)
    X = []
    for t in ids:
        d, l = depth[t], abs(lat[t])
        others = [o for o in ids if o != t]
        n = max(1, len(others))
        X.append([
            d, l, lat[t], 1.0 if d <= _LINE_BAND else 0.0,
            sum(1 for o in others if depth[o] < d) / n,
            sum(1 for o in others if abs(lat[o]) < l) / n,
            float(sum(1 for o in others if depth[o] > d)),
            float(sum(1 for o in others if abs(lat[o]) > l)),
            min((abs(pts[t][0] - pts[o][0]) + abs(pts[t][1] - pts[o][1]) for o in others), default=0.0),
            1.0 if d == max_d else 0.0,
            1.0 if l == min_l else 0.0,
        ])
    return ids, X


def load_model():
    """Return the bundled classifier, or None if unavailable (no [ml] extra / no file)."""
    if not _MODEL_PATH.is_file():
        return None
    try:
        import joblib
        # trusted artifact: this model ships inside the package, not user input
        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


def classify(off_ids, pts, los_x, model=None):
    """{track_id: OL/QB/RB/WR} from the learned model, or None to signal fallback."""
    model = model if model is not None else load_model()
    if model is None or not off_ids:
        return None
    ids, X = extract(off_ids, pts, los_x)
    if not ids:
        return None
    pred = model.predict(X)
    return {t: str(p) for t, p in zip(ids, pred)}
