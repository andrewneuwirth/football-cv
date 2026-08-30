# tests/test_positions.py — field coords are (down, across) in yards; los is a `down`.
from ml import positions

LOS = 50.0
MID = positions.FIELD_WIDTH_YD / 2.0  # field center (across)


def test_defensive_line_on_the_line_is_DL():
    assert positions.classify_player(down=50.9, across=MID, los=LOS, side="D") == "DL"


def test_deep_middle_defender_is_S():
    assert positions.classify_player(down=61.0, across=MID, los=LOS, side="D") == "S"


def test_wide_defender_is_CB():
    assert positions.classify_player(down=56.0, across=4.0, los=LOS, side="D") == "CB"


def test_off_line_defender_is_LB():
    assert positions.classify_player(down=54.5, across=MID, los=LOS, side="D") == "LB"


def test_offensive_line_is_OL():
    assert positions.classify_player(down=49.4, across=MID, los=LOS, side="O") == "OL"


def test_split_offense_is_WR():
    assert positions.classify_player(down=49.4, across=6.0, los=LOS, side="O") == "WR"


def test_quarterback_is_centered_and_deep():
    assert positions.classify_player(down=44.5, across=MID, los=LOS, side="O") == "QB"


def test_offset_back_is_RB():
    assert positions.classify_player(down=44.0, across=MID + 5.0, los=LOS, side="O") == "RB"


def test_classify_play_labels_both_sides():
    alignment = {1: (49.4, MID), 2: (49.4, 4.0), 11: (50.9, MID), 12: (61.0, MID)}
    split = {1: "O", 2: "O", 11: "D", 12: "D"}
    out = positions.classify_play(alignment, split, los=LOS)
    assert out[1] == "OL" and out[2] == "WR"
    assert out[11] == "DL" and out[12] == "S"
