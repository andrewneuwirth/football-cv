# ml/cli.py
"""Command-line runner: python -m ml.cli <stage> <play_id> [--data DIR]."""
import argparse
from pathlib import Path
from ml import detect, track, field, autocal, teamcolor, positions, paths

STAGES = {
    "detect": detect.run, "track": track.run, "field": field.run,
    "autocal": autocal.run, "teamcolor": teamcolor.run, "positions": positions.run,
}
ORDER = ["detect", "track", "autocal", "field", "teamcolor", "positions"]

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="football-cv")
    p.add_argument("stage", choices=list(STAGES) + ["all"], help="pipeline stage to run")
    p.add_argument("play_id")
    p.add_argument("--data", default=None)
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        # argparse exits on --help (code 0) or an invalid stage (code 2);
        # surface that code as the return value instead of propagating.
        return int(exc.code or 0)
    data = Path(args.data) if args.data else paths.data_dir()
    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        STAGES[s](args.play_id, data)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
