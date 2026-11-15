# tests/test_playcall.py — play-call predictor on the synthetic snaps fixture (no real data).
import csv
import importlib.util
import io
from pathlib import Path

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("pandas")

from footballcv import cli, playcall  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAPS = ROOT / "examples" / "synthetic_snaps.csv"


def test_fixture_is_the_generator_output():
    spec = importlib.util.spec_from_file_location("synth", ROOT / "examples" / "synthetic_snaps.py")
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=synth.COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(synth.snaps())
    assert SNAPS.read_text() == buf.getvalue()


def test_split_by_game_never_by_snap():
    df = playcall.load(SNAPS)
    sp = playcall.split(df)
    sets = [set(v) for v in sp.values()]
    assert sum(map(len, sets)) == len(set().union(*sets)) == df["gid"].nunique()
    assert sp["test"] == sorted(df["gid"].unique())[-2:]       # latest games by date


def test_too_few_games_is_an_error():
    df = playcall.load(SNAPS)
    with pytest.raises(ValueError, match="at least"):
        playcall.split(df[df["gid"].isin(df["gid"].unique()[:3])])


def test_missing_required_column(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("down,dist\n1,10\n")
    with pytest.raises(ValueError, match="game, run_pass"):
        playcall.load(f)


def test_train_and_predict(tmp_path):
    m = playcall.train(SNAPS, tmp_path / "m")
    e = m["eval"]
    assert {"model", "majority", "tendency", "logo", "direction"} <= set(e)
    assert e["model"]["n"] == e["majority"]["n"] == e["tendency"]["n"] == m["data"]["test"]   # same test snaps
    assert e["model"]["acc"] > e["majority"]["acc"]            # the synthetic tendencies are learnable
    assert "model" in e["direction"]

    snaps = tmp_path / "new.csv"   # unlabeled snap with an unseen formation and missing columns
    snaps.write_text("down,dist,off_form,qb_set,backfield\n3,9,6x0,gun,0\n1,10,2x1,under,2\n")
    out = playcall.predict(tmp_path / "m", snaps)
    assert out["p_pass"].between(0, 1).all() and "p_right" in out
    assert out["p_pass"][0] > out["p_pass"][1]                 # empty gun on 3rd & 9 vs two-back under center


def test_cli_train_predict(tmp_path, capsys):
    assert cli.main(["playcall", "train", "--data", str(SNAPS), "--out", str(tmp_path / "m")]) == 0
    assert "leave-one-game-out" in capsys.readouterr().out
    out = tmp_path / "pred.csv"
    assert cli.main(["playcall", "predict", "--model", str(tmp_path / "m"), "--data", str(SNAPS), "--out", str(out)]) == 0
    assert "p_pass" in out.read_text().splitlines()[0]
