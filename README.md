# football-cv

Computer-vision pipeline for football film. Point it at a play and it detects
every player, tracks them across frames, calibrates the camera to field
coordinates, assigns teams by jersey color, and classifies the **defense** into
standard position groups.

MIT licensed.

## What it does

```
film → person detection → multi-object tracking → field calibration
     → team assignment → position classification → positions.json
```

Each stage is an independent module under `ml/` that reads and writes per-play
artifacts on disk. The headline module is `ml/positions.py`: it takes the
field-projected tracks, strips officials and sideline crews, splits offense from
defense by jersey color, finds the line of scrimmage and the snap, and labels
each defender by alignment. Output is only the generic position groups
`DL / LB / CB / S` — no playbook-specific naming.

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
python -m ml.cli ingest my_play --video path/to/film.mp4 --data ./data

# run the full pipeline end to end (needs the [detect] extra)
python -m ml.cli all my_play --data ./data
```

`ingest` copies your video to `data/plays/<play_id>/clip.mp4`, which every
downstream stage reads. You can also drop a `clip.mp4` there yourself.

Field calibration needs a one-time reference: mark ≥4 field points (yard-line ×
sideline intersections) on one frame to write `calibration.json`. The browser
tool under `viewer/` does this; `autocal` then transfers that calibration to
other plays in the same game by feature matching.

### Run individual stages

```bash
python -m ml.cli detect my_play --data ./data
```

Stages, in pipeline order: `ingest`, `detect`, `track`, `autocal`, `field`,
`teamcolor`, `positions`. `all` runs `detect`→`positions` in sequence. Each
stage reads the artifacts written by the previous one from the play's directory
under `--data`. The final stage writes `positions.json` (a map from track id to
position group) and a richer `labels.json` (teams, line of scrimmage, snap).

See [`examples/`](examples/) for a sample `positions.json`.

## Position groups

The classifier labels defenders with four standard groups, inferred from
field-relative alignment:

| Token | Position       | Read                                        |
| ----- | -------------- | ------------------------------------------- |
| `DL`  | Defensive line | On the line of scrimmage                    |
| `LB`  | Linebacker     | A few yards off the line, inside            |
| `CB`  | Cornerback     | Pressed wide, near the boundary             |
| `S`   | Safety         | Deep off the line, toward the middle        |

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
