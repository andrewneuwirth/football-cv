from footballcv import offense_model as om


def test_extract_shape():
    pts = {1: (48.0, 26.0), 2: (48.0, 5.0), 3: (44.0, 26.0), 4: (44.0, 30.0)}
    ids, X = om.extract([1, 2, 3, 4], pts, los_x=50.0)
    assert ids == [1, 2, 3, 4]
    assert all(len(row) == len(om.FEATURES) for row in X)


def test_classify_returns_none_when_no_offense():
    assert om.classify([], {}, los_x=50.0) is None


def test_classify_with_bundled_model():
    # the model ships in the package; predictions must be valid tokens
    model = om.load_model()
    if model is None:
        return  # no [ml] extra / model absent — heuristic path is used instead
    pts = {1: (48.0, 26.0), 2: (48.0, 5.0), 3: (44.0, 26.0), 4: (44.0, 30.0)}
    out = om.classify([1, 2, 3, 4], pts, los_x=50.0)
    assert set(out) == {1, 2, 3, 4}
    assert all(v in ("OL", "QB", "RB", "WR") for v in out.values())
