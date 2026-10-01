# AGENTS.md — football-cv

## Purpose
Open-source computer-vision pipeline: single-play football film → detect, track,
team-split by jersey color, calibrate to field yards, find LOS + snap, filter
officials, and classify all 22 players (DL/LB/CB/S, OL/QB/RB/WR) → positions.json.
Generic toolkit (Python API, `football-cv` CLI, browser viewer); public on GitHub.

## Stack
Python >= 3.11 package `footballcv/` (numpy, scipy, opencv; torch/YOLO only via the
`[detect]` extra). Viewer: zero-build HTML in `viewer/web/` + `viewer/backend/serve.py`;
`viewer/field|anim|theme` are React Native Skia TS components. MIT license.

## Install / run / test
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[detect]" for YOLO detection (torch)
football-cv ingest myplay --video play.mp4 --data ./data
football-cv all myplay --data ./data
python viewer/backend/serve.py     # http://localhost:8000/calibrate.html?play=<id>
pytest -q                          # what CI runs (.github/workflows/ci.yml)
```

## Where outputs / data go
- Per-play artifacts: `<data dir>/plays/<play_id>/` — data dir from `--data`,
  else `$FOOTBALL_CV_DATA`, else `./data` in the repo. `data/` and `*.mp4` are gitignored.
- `footballcv/models/offense.joblib` — trained offense classifier, written by
  `footballcv/train_offense.py`, shipped as package data.

## Do not move
- `footballcv/models/offense.joblib` — loaded by path in `offense_model.py` and
  declared as package data in `pyproject.toml`.
- `footballcv/`, `viewer/`, `examples/`, `tests/`, `docs/`, `README.md`, `LICENSE`,
  `pyproject.toml` — the CI scrub step greps exactly these paths.
- `docs/hero.png` — README hero image.

## Rules
- Docs go in `docs/`, never the repo root.
- Public repo: keep it generic — no team names, playbook terms, or real film.
  CI's optional scrub step fails on terms in the `SCRUB_PATTERN` secret.
- Never commit video (`*.mp4`) or `data/`.
