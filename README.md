# football-cv

Computer-vision pipeline for football film. Point it at a play and it detects
every player, tracks them across frames, calibrates the camera to field
coordinates, assigns teams by jersey color, and classifies both teams into
standard position groups.

MIT licensed.

## What it does

```
film → person detection → multi-object tracking → field calibration
     → team assignment → position classification → positions.json
```

Each stage is an independent module under `footballcv/` that reads and writes
per-play artifacts on disk. The headline module is `footballcv/positions.py`: it
takes the field-projected tracks, strips officials and sideline crews, splits
offense from defense by jersey color, finds the line of scrimmage and the snap,
and labels each player by alignment. Output is only the generic position groups
`DL / LB / CB / S` and `OL / QB / RB / WR` — no playbook-specific naming.

## Install

```bash
pip install -e .              # the classifier (numpy, scipy) + pipeline
pip install -e ".[detect]"    # + YOLO person detection (pulls in torch)
pip install -e ".[dev]"       # + test dependencies (pytest)
```

Requires Python 3.11+. Person detection (`ultralytics`/YOLO, and therefore
torch) is an optional extra — the classifier and tests run without it.

## Usage

### Bring your own film

```bash
# copy your film into place as a named play
python -m footballcv.cli ingest my_play --video path/to/film.mp4 --data ./data

# run the full pipeline end to end (needs the [detect] extra)
python -m footballcv.cli all my_play --data ./data
```

`ingest` copies your video to `data/plays/<play_id>/clip.mp4`, which every
downstream stage reads. You can also drop a `clip.mp4` there yourself.

Field calibration needs a one-time reference: mark ≥4 field points (yard-line ×
sideline intersections) on one frame to write `calibration.json`. The browser
tool under `viewer/` does this; `autocal` then transfers that calibration to
other plays in the same game by feature matching.

### Run individual stages

```bash
python -m footballcv.cli detect my_play --data ./data
```

Installing the package also puts a `football-cv` command on your PATH, so
`football-cv detect my_play --data ./data` works too.

Stages, in pipeline order: `detect`, `track`, `teamcolor`, `stitch`, `autocal`,
`field`, `positions`. `all` runs them in sequence. Each stage reads the
artifacts written by the previous one from the play's directory under `--data`.
The final stage writes `positions.json` (a map from track id to position group)
and a richer `labels.json` (teams, line of scrimmage, snap).

See [`examples/`](examples/) for a sample `positions.json`.

## Use as a library

Prefer code over the CLI? The whole pipeline is one call:

```python
import footballcv

positions = footballcv.analyze(
    video="play.mp4",
    calibration={
        "frame": 120,
        "points": [                                   # pixel -> field yards
            {"px": 20,   "py": 1030, "fx": 30, "fy": 0.0},    # 30 yd @ near sideline
            {"px": 1900, "py": 300,  "fx": 30, "fy": 53.33},  # 30 yd @ far sideline
            {"px": 700,  "py": 1040, "fx": 45, "fy": 0.0},    # 45 yd @ near sideline
            {"px": 1905, "py": 470,  "fx": 45, "fy": 53.33},  # 45 yd @ far sideline
        ],
    },
    snap=120,   # optional: frame to anchor the pre-snap alignment
)
# -> {"3": "DL", "11": "LB", "24": "S", "41": "OL", "52": "QB", ...}
```

`analyze()` needs the detection extra (`pip install "football-cv[detect]"`).
Artifacts land under a temp dir by default; pass `data_dir=` to keep them.

Lower-level helpers for driving it yourself:

```python
from footballcv import calibrate, set_snap, load_positions, load_labels

calibrate("play1", "./data", points=[...], frame=120)   # write calibration.json
set_snap("play1", "./data", frame=120)                  # write snap.json
positions = load_positions("./data", "play1")           # {track_id: group}
labels = load_labels("./data", "play1")                 # teams, LOS, snap, ...
```

Every stage is also importable and callable on its own
(`from footballcv import detect, track, positions`), each reading and writing
the per-play artifacts on disk.

## Position groups

Players are labeled with standard groups, inferred from field-relative
alignment. Defense (full alignment logic):

| Token | Position       | Read                                        |
| ----- | -------------- | ------------------------------------------- |
| `DL`  | Defensive line | On the line of scrimmage                    |
| `LB`  | Linebacker     | A few yards off the line, inside            |
| `CB`  | Cornerback     | Pressed wide, near the boundary             |
| `S`   | Safety         | Deep off the line, toward the middle        |

Offense (generic alignment): `OL` on the line, `WR` split wide, `QB` the deep
centered back, `RB` the other backs.

## Viewer

`viewer/` holds a browser-based field viewer and calibration tool:

- `viewer/web/` — a zero-build calibration UI (`calibrate.html`, with a frame
  scrubber) and an overlay viewer (`result.html`) that draws the classified
  positions on the film.
- `viewer/backend/serve.py` — a small local server that serves the UI, saves
  calibrations, runs the pipeline, and renders the overlay.
- `viewer/field/` — React Native / Expo + Skia components (an early-stage port;
  full Expo build wiring is out of scope for v0.1).

Run the local viewer with `python viewer/backend/serve.py` and open
`http://localhost:8000/calibrate.html?play=<play_id>`.

## License

MIT. See [`LICENSE`](LICENSE).
