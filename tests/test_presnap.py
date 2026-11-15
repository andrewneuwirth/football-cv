# tests/test_presnap.py — film -> pre-snap features on a hand-built synthetic play.
import json

from footballcv import presnap

SNAP = 60


def _play(tmp_path):
    """Offense going +x, LOS at x=30, ball in the middle; facing +x the offense's left is +y."""
    bodies = {  # tid: (team, token, x, y)
        **{i: ("O", "OL", 29.0, 24.67 + i - 1) for i in range(1, 6)},
        6: ("O", "QB", 25.0, 26.67), 7: ("O", "RB", 24.0, 26.67),
        8: ("O", "WR", 30.0, 45.0), 9: ("O", "WR", 30.0, 40.0), 10: ("O", "WR", 30.0, 8.0),
        **{10 + i: ("D", "DL", 31.0, 24.0 + 1.5 * i) for i in range(1, 5)},
        15: ("D", "LB", 35.0, 22.0), 16: ("D", "LB", 35.0, 27.0), 17: ("D", "LB", 35.0, 32.0),
        18: ("D", "CB", 37.0, 45.0), 19: ("D", "CB", 37.0, 8.0),
        20: ("D", "S", 42.0, 20.0), 21: ("D", "S", 42.0, 33.0),
    }
    tracks = []
    for tid, (_, _, x, y) in bodies.items():
        # WR 10 motions from y=14 to y=8 (toward the offense's right) a second before the snap
        ys = [14.0 if (tid == 10 and i < 45) else y for i in range(SNAP + 1)]
        tracks.append({"track_id": tid, "frames": [{"i": i, "x": x, "y": ys[i]} for i in range(SNAP + 1)]})
    d = tmp_path / "plays" / "p1"
    d.mkdir(parents=True)
    (d / "field_coords.json").write_text(json.dumps({"tracks": tracks}))
    (d / "labels.json").write_text(json.dumps({"snap_frame": SNAP, "los_x": 30.0, "offense_direction": "+x",
                                               "teams": {str(t): b[0] for t, b in bodies.items()}}))
    (d / "positions.json").write_text(json.dumps({str(t): b[1] for t, b in bodies.items()}))
    (d / "meta.json").write_text(json.dumps({"fps": 30}))
    return tmp_path


def test_features_from_synthetic_play(tmp_path):
    f = presnap.features("p1", _play(tmp_path))
    assert f == {"play_id": "p1", "yard_ln": 30.0, "hash": "M", "off_str": "L", "off_form": "2x1",
                 "backfield": "1", "qb_set": "gun", "box": 7, "motion_dir": "R"}


def test_write_csv_leaves_situation_blank(tmp_path):
    data = _play(tmp_path)
    out = tmp_path / "snaps.csv"
    presnap.write_csv(["p1"], data, out)
    header, row = out.read_text().splitlines()
    cols = dict(zip(header.split(","), row.split(",")))
    assert cols["down"] == "" and cols["run_pass"] == "" and cols["off_form"] == "2x1"
