# tests/test_positions.py
from ml import positions


def test_defensive_line_body_is_DL():
    # on the line, defensive side, at the front edge
    assert positions.classify_player(
        rel=(0.0, 1.0), side="D", los=0.0, front_edge=1.0) == "DL"


def test_deep_middle_defender_is_S():
    # far behind the LB level, near the middle
    assert positions.classify_player(
        rel=(0.0, 12.0), side="D", los=0.0, front_edge=1.0) == "S"


def test_wide_defender_is_CB():
    # near the line depth but split wide toward the sideline
    assert positions.classify_player(
        rel=(22.0, 3.0), side="D", los=0.0, front_edge=1.0) == "CB"


def test_off_line_defender_is_LB():
    # a few yards off the line, between the front and the safeties
    assert positions.classify_player(
        rel=(2.0, 4.0), side="D", los=0.0, front_edge=1.0) == "LB"


def test_offensive_line_body_is_OL():
    assert positions.classify_player(
        rel=(0.0, 0.5), side="O", los=0.0, front_edge=0.5) == "OL"


def test_split_offense_body_is_WR():
    assert positions.classify_player(
        rel=(20.0, 0.5), side="O", los=0.0, front_edge=0.5) == "WR"


def test_deep_offense_back_is_RB_or_QB():
    lbl = positions.classify_player(
        rel=(0.0, 6.0), side="O", los=0.0, front_edge=0.5)
    assert lbl in ("RB", "QB")


def test_classify_play_labels_both_sides():
    alignment = {1: (0.0, 0.0), 2: (25.0, 0.0), 3: (0.0, 1.0), 4: (0.0, 12.0)}
    split = {1: "O", 2: "O", 3: "D", 4: "D"}
    out = positions.classify_play(alignment, split, los=0.0)
    assert out[1] == "OL" and out[2] == "WR"
    assert out[3] == "DL" and out[4] == "S"
