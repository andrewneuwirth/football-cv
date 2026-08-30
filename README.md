# football-cv

Generic computer-vision pipeline for tracking football players from film and
classifying them into standard positions. Point it at a play, and it detects
every player, tracks them across frames, calibrates the camera to field
coordinates, assigns teams by jersey color, and labels each tracked player with
a generic position token.

MIT licensed.

## What it does

```
film → person detection → multi-object tracking → field calibration
     → team assignment → generic position classification → positions.json
```

Each stage is an independent module under `ml/` that reads and writes per-play
artifacts on disk. The headline module is `ml/positions.py`: a pure,
geometry-driven classifier that maps a player's field-relative alignment to a
standard position. It uses only generic football geometry (line of scrimmage,
offense/defense split, alignment depth and width) — no proprietary playbook
logic.

## Install

```bash
pip install -e .
# with test dependencies
pip install -e ".[dev]"
```

Requires Python 3.11+. Core dependencies: `numpy`, `opencv-python`, `scipy`,
`ultralytics` (YOLO).

## Usage

Run a single stage or the whole pipeline for a play:

```bash
# run one stage
python -m ml.cli detect <play_id> --data ./data

# run the full pipeline end to end
python -m ml.cli all <play_id> --data ./data
```

Stages, in pipeline order: `detect`, `track`, `autocal`, `field`, `teamcolor`,
`positions`. Use `all` to run them in sequence. Each stage reads the artifacts
written by the previous one from the play's directory under `--data`. The final
stage writes `positions.json`, a map from track id to position token.

See [`examples/`](examples/) for a sample `positions.json`.

## Position taxonomy

The classifier emits eight generic tokens — four per side of the ball:

| Side    | Tokens                                   |
| ------- | ---------------------------------------- |
| Defense | `DL` `LB` `CB` `S`                       |
| Offense | `OL` `QB` `RB` `WR`                      |

| Token | Position          | Read                                         |
| ----- | ----------------- | -------------------------------------------- |
| `DL`  | Defensive line    | On the line, defensive side                  |
| `LB`  | Linebacker        | A few yards off the line, inside             |
| `CB`  | Cornerback        | Near the line depth but split wide           |
| `S`   | Safety            | Deep off the line, toward the middle         |
| `OL`  | Offensive line    | On the line, offensive side                  |
| `QB`  | Quarterback       | Deepest, laterally centered back             |
| `RB`  | Running back      | In the backfield, off center                 |
| `WR`  | Wide receiver     | Split wide from the offensive line           |

These are inferred purely from field-relative geometry.

## Viewer

`viewer/` holds a React Native / Expo + Skia field renderer that draws a field
and overlays player markers by position token, plus a film-replay overlay. It
ships with a neutral color palette and no team branding. The viewer is an
early-stage port; full Expo build wiring is out of scope for v0.1.

## License

MIT. See [`LICENSE`](LICENSE).
