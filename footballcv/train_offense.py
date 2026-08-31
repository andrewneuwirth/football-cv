"""Train the learned offense classifier (needs the ``[ml]`` extra).

Reads labeled plays under ``--data`` — each play needs field_coords.json,
labels.json (for los_x), snap.json, and a ground_truth.json ({track_id: group}).
Reports leave-one-play-out accuracy, then trains on all plays and saves the
model to footballcv/models/offense.joblib (loaded automatically at inference).

    python -m footballcv.train_offense --data ./labeled_data
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

from footballcv import offense_model as om

OFF = {"OL", "QB", "RB", "WR"}


def _play_rows(pdir: Path):
    """(X, y) feature rows + labels for the offense tracks in one play."""
    gtf, lf = pdir / "ground_truth.json", pdir / "labels.json"
    if not (gtf.is_file() and lf.is_file() and (pdir / "field_coords.json").is_file()):
        return [], []
    gt = json.loads(gtf.read_text())
    los = json.loads(lf.read_text()).get("los_x")
    if los is None:
        return [], []
    snapf = pdir / "snap.json"
    snap = json.loads(snapf.read_text()).get("frame") if snapf.is_file() else None
    fc = json.loads((pdir / "field_coords.json").read_text())
    pos = {}
    for t in fc.get("tracks", []):
        frs = [f for f in t["frames"] if snap is None or abs(f["i"] - snap) <= 8]
        if frs:
            pos[str(t["track_id"])] = (st.median(f["x"] for f in frs), st.median(f["y"] for f in frs))
    off_ids = [t for t in gt if gt[t] in OFF and t in pos]
    if len(off_ids) < 3:
        return [], []
    ids, X = om.extract(off_ids, pos, los)
    return X, [gt[t] for t in ids]


def _dataset(data_dir: Path):
    X, y, groups = [], [], []
    for gtf in sorted(glob.glob(str(data_dir / "plays" / "*" / "ground_truth.json"))):
        pdir = Path(gtf).parent
        Xi, yi = _play_rows(pdir)
        for xi, yv in zip(Xi, yi):
            X.append(xi); y.append(yv); groups.append(pdir.name)
    return X, y, groups


def _make_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=0
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="footballcv.train_offense")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(om._MODEL_PATH))
    args = ap.parse_args(argv)

    X, y, g = _dataset(Path(args.data))
    if len(y) < 10:
        print(f"not enough labeled offense samples ({len(y)})")
        return 1
    print(f"offense samples: {len(y)}   {dict(Counter(y))}")

    # honest estimate: leave-one-play-out cross-validation
    from sklearn.model_selection import LeaveOneGroupOut
    import numpy as np
    Xa, ya, ga = np.array(X), np.array(y), np.array(g)
    per, corr = Counter(), 0
    for tr, te in LeaveOneGroupOut().split(Xa, ya, ga):
        m = _make_model().fit(Xa[tr], ya[tr])
        for i, pi in zip(te, m.predict(Xa[te])):
            per[ya[i]] += 1
            if pi == ya[i]:
                corr += 1
    print(f"leave-one-play-out accuracy: {corr/len(y):.1%}")

    # train final model on everything and save
    model = _make_model().fit(X, y)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, out)
    print(f"saved model -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
