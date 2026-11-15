# Play-call prediction

From the pre-snap look, predict **run vs pass** and, for runs, **direction (L/R)**.
football-cv reads part of that look off the film; you add the game situation; an
XGBoost model trained on your own tagged snaps does the rest.

```
film → football-cv all → positions → playcall features ─┐
                     your situation columns (down, dist…) ┴→ snaps.csv → playcall train / predict
```

```bash
pip install -e ".[playcall]"     # xgboost + pandas; the base install stays light
```

## Commands

```bash
# 1. film -> one CSV row per play (after `football-cv all <play>`)
football-cv playcall features play1 play2 play3 --data ./data --out snaps.csv

# 2. fill in down, dist, game, run_pass, ... in snaps.csv (or join your own tagging export)

# 3. train + evaluate; saves model.json, direction_model.json, categories.json, metrics.json
football-cv playcall train --data snaps.csv --out model/

# 4. predict new snaps (labels not needed): adds p_pass, pick, p_right
football-cv playcall predict --model model/ --data upcoming.csv --out predictions.csv
```

Python:

```python
from footballcv import presnap, playcall

row = presnap.features("play1", "./data")          # film-derived columns for one play
metrics = playcall.train("snaps.csv", "model/")     # dict, also written to model/metrics.json
print(playcall.summary(metrics))
preds = playcall.predict("model/", "upcoming.csv")  # pandas DataFrame
```

## CSV schema

One row per offensive snap. Only `game` and `run_pass` are required to train;
every other column is optional. A missing column or a blank cell is treated as
"unknown" (blank categoricals are their own category), so you can start with
whatever you tag.

| column       | type        | source | meaning |
| ------------ | ----------- | ------ | ------- |
| `game`       | text        | you    | game id (any label) |
| `game_date`  | YYYY-MM-DD  | you    | orders games for the split; a rematch on another date is its own game. Without it, file order is used |
| `play_no`    | int         | you    | snap order (bookkeeping only) |
| `qtr`        | int         | you    | quarter |
| `down`       | int         | you    | 1-4 |
| `dist`       | int         | you    | yards to go |
| `prev_type`  | Run / Pass  | you    | the previous snap's call |
| `prev_gain`  | int         | you    | the previous snap's gain |
| `yard_ln`    | number      | film or you | yards from the offense's own goal line to the LOS (1-99) |
| `hash`       | L / M / R   | film or you | ball spot, from the offense's view |
| `off_str`    | L / R / bal | film or you | side with more offensive players outside the tackles |
| `off_form`   | text        | film or you | formation label. From film: `<strong>x<weak>` outside the tackles, e.g. `3x1`. Any consistent labels work |
| `backfield`  | text        | film or you | backfield set. From film: number of RBs (`0`-`3`) |
| `qb_set`     | gun / under | film or you | QB depth: ≥ 3.5 yd off the LOS = `gun` |
| `box`        | int         | film or you | defenders within 6 yd of the ball and 7 yd of the LOS |
| `motion_dir` | L / R / none | film or you | biggest pre-snap lateral move (≥ 2.5 yd in the last 2 s) by a WR/RB |
| `run_pass`   | Run / Pass  | you    | **label** to learn (other values are skipped when training) |
| `play_dir`   | L / R       | you    | **label** for run direction (Run snaps only) |

## What film can and can't give you

**From film** (`presnap.features`, read from the `positions` stage outputs at the snap):
offensive strength, a receiver-split formation label, backfield count, QB set,
players in the box, pre-snap motion, hash, and field position.

Caveats — these inherit every error upstream:

- They lean on the offense position labels (~61% accurate, QB/RB the weakest), so
  `backfield` and `qb_set` are the noisiest film features. Check them before trusting them.
- `yard_ln` is only right if the calibration uses absolute yard lines, 0-100 from one
  goal line to the other. With midfield-relative marks (two "30"s) override it by hand.
- Motion needs the motion man tracked through the last 2 s before the snap.
- A tight end lands in `OL` or `WR`, so formation labels are generic, not playbook names.

**Only from you:** down, distance, quarter, previous play and its gain, the
game/date, and the labels (`run_pass`, `play_dir`). Down & distance are usually the
strongest features, so the film columns alone will not get you far.

## How it's evaluated

The rules matter more than the model. A model scored on random snaps from games it
has already seen looks far better than it is.

- **Split by game, never by snap.** Games in date order: the latest 2 are test
  (`--test-games`), the one before them validation (early stopping only), the rest
  train. Needs at least 4 games. A snap never sees its own game during training.
- **Baselines on the same test snaps**, both built from train + validation:
  "always the most common play", and a down × distance-bucket (1-3, 4-6, 7-10, 11+)
  pass-rate table — roughly what a coach does by hand. A model that doesn't beat the
  table isn't adding anything.
- **Confident-snap accuracy.** Share of test snaps where the model is ≥ 70% / ≥ 80%
  one way, and how often those calls are right. This is the number a play-caller
  uses: "when it's sure, can I trust it?"
- **Leave-one-game-out** is also reported, but later games help predict earlier
  ones, so it is optimistic. The test games are the headline.
- **Run direction** is a second model on Run snaps only, same split, against
  "always the more common side".

## Results on synthetic data

**These numbers are from `examples/synthetic_snaps.csv` — a made-up season with
invented tendencies (`examples/synthetic_snaps.py`), not real football.** They show
the tool working, nothing about how well it predicts any real team. Measure on your
own snaps.

```
618 snaps, 10 games; test = the last 2 games (126 snaps)
test accuracy      model 67.5%   always Pass 57.1%   down x distance table 50.8%
confident >= 70%   42.1% of snaps, right 77.4%
confident >= 80%   18.3% of snaps, right 78.3%
leave-one-game-out 68.6% (optimistic; majority 52.1%, table 53.4%)
run direction      model 61.1%   always L 59.3%  (54 test runs)
```

Reproduce: `football-cv playcall train --data examples/synthetic_snaps.csv --out /tmp/m`.

Small data warning: with a few hundred snaps, 2 test games is ~100-150 snaps, so a
few points either way is noise. Treat a gap to the baselines of under ~5 points as a tie.
