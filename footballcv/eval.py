"""Evaluate position predictions against ground-truth labels.

Ground truth is a per-play `ground_truth.json` mapping track id -> position
group (the same generic vocabulary the classifier emits). This tool reports
overall accuracy and per-class precision / recall / F1 across a set of plays,
plus a confusion matrix.

    python -m footballcv.eval --data ./data --plays play1,play2,play3

Only tracks that appear in ground truth are scored; a track with no prediction
counts as a miss.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from footballcv import paths

GROUPS = ("DL", "LB", "CB", "S", "OL", "QB", "RB", "WR")


def score_play(pred: dict, truth: dict) -> collections.Counter:
    """Confusion counts {(truth, pred): n} for one play, over ground-truth tracks."""
    conf: collections.Counter = collections.Counter()
    for tid, t in truth.items():
        p = pred.get(str(tid), "<none>")
        conf[(t, p)] += 1
    return conf


def report(conf: collections.Counter) -> dict:
    """Overall accuracy + per-class precision/recall/F1 from confusion counts."""
    total = sum(conf.values())
    correct = sum(n for (t, p), n in conf.items() if t == p)
    labels = sorted({t for (t, _p) in conf} | {p for (_t, p) in conf if p in GROUPS})
    per_class = {}
    for c in labels:
        tp = conf.get((c, c), 0)
        fp = sum(n for (t, p), n in conf.items() if p == c and t != c)
        fn = sum(n for (t, p), n in conf.items() if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "per_class": per_class,
    }


def evaluate(data_dir, plays, truth_name: str = "ground_truth.json") -> dict:
    """Aggregate metrics across `plays` (each needs positions.json + truth_name)."""
    conf: collections.Counter = collections.Counter()
    used = 0
    for play in plays:
        pdir = paths.play_dir(play, Path(data_dir))
        pred_f, truth_f = pdir / "positions.json", pdir / truth_name
        if not (pred_f.is_file() and truth_f.is_file()):
            continue
        conf += score_play(json.loads(pred_f.read_text()), json.loads(truth_f.read_text()))
        used += 1
    out = report(conf)
    out["plays"] = used
    return out


def format_report(r: dict) -> str:
    lines = [
        f"plays: {r.get('plays', '?')}   players scored: {r['n']}",
        f"overall accuracy: {r['accuracy']:.1%}",
        "",
        f"{'group':<6}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>6}",
    ]
    for c in GROUPS:
        m = r["per_class"].get(c)
        if not m:
            continue
        lines.append(f"{c:<6}{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}{m['support']:>6}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="footballcv.eval")
    p.add_argument("--data", required=True)
    p.add_argument("--plays", required=True, help="comma-separated play ids")
    p.add_argument("--truth", default="ground_truth.json")
    args = p.parse_args(argv)
    r = evaluate(args.data, [s for s in args.plays.split(",") if s], args.truth)
    print(format_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
