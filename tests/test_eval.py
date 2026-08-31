from footballcv import eval as ev


def test_score_and_report_perfect():
    pred = {"1": "DL", "2": "LB", "3": "CB"}
    truth = {"1": "DL", "2": "LB", "3": "CB"}
    r = ev.report(ev.score_play(pred, truth))
    assert r["accuracy"] == 1.0
    assert r["n"] == 3
    assert r["per_class"]["DL"]["f1"] == 1.0


def test_missing_prediction_counts_as_miss():
    conf = ev.score_play({"1": "DL"}, {"1": "DL", "2": "S"})
    r = ev.report(conf)
    assert r["n"] == 2
    assert r["accuracy"] == 0.5              # track 2 has no prediction
    assert r["per_class"]["S"]["recall"] == 0.0


def test_precision_recall_on_confusion():
    # two DL truths: one right, one called LB; one LB truth called LB
    pred = {"1": "DL", "2": "LB", "3": "LB"}
    truth = {"1": "DL", "2": "DL", "3": "LB"}
    r = ev.report(ev.score_play(pred, truth))
    assert r["per_class"]["DL"]["recall"] == 0.5   # 1 of 2 DL found
    assert r["per_class"]["LB"]["precision"] == 0.5  # 1 of 2 "LB" calls correct
