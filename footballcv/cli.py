# ml/cli.py
"""Command-line runner: python -m footballcv.cli <stage> <play_id> [--data DIR] [--video FILM].

Upload your own film first, then run the pipeline:

    python -m footballcv.cli ingest my_play --video film.mp4 --data ./data
    python -m footballcv.cli all my_play --data ./data

Play-call predictor (needs the [playcall] extra; see docs/playcall.md):

    football-cv playcall features my_play --data ./data --out snaps.csv
    football-cv playcall train --data snaps.csv --out model/
    football-cv playcall predict --model model/ --data snaps.csv
"""
import argparse
import sys
from pathlib import Path
from footballcv import detect, track, field, autocal, teamcolor, stitch, positions, ingest, paths

# stages that run over an already-ingested play (in pipeline order)
STAGES = {
    "detect": detect.run, "track": track.run, "field": field.run,
    "autocal": autocal.run, "teamcolor": teamcolor.run,
    "stitch": stitch.stitch, "positions": positions.run,
}
# teamcolor + stitch run before field so the fragment-merged tracks get
# projected once; positions then sees one track per player.
ORDER = ["detect", "track", "teamcolor", "stitch", "autocal", "field", "positions"]

def playcall_main(argv) -> int:
    p = argparse.ArgumentParser(prog="football-cv playcall")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="film -> pre-snap feature rows (after the positions stage)")
    f.add_argument("play_ids", nargs="+")
    f.add_argument("--data", default=None, help="football-cv data dir")
    f.add_argument("--out", default="snaps.csv")
    t = sub.add_parser("train", help="train run/pass + run direction on a snaps CSV")
    t.add_argument("--data", required=True, help="snaps CSV")
    t.add_argument("--out", default="playcall_model")
    t.add_argument("--test-games", type=int, default=2)
    r = sub.add_parser("predict", help="score a snaps CSV with a trained model")
    r.add_argument("--model", required=True)
    r.add_argument("--data", required=True, help="snaps CSV")
    r.add_argument("--out", default=None, help="CSV to write (default: stdout)")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.cmd == "features":
        from footballcv import presnap
        presnap.write_csv(args.play_ids, Path(args.data) if args.data else paths.data_dir(), args.out)
        print(f"wrote {args.out}: fill in the situation columns (down, dist, ...) before train/predict")
        return 0
    from footballcv import playcall  # needs the [playcall] extra
    if args.cmd == "train":
        print(playcall.summary(playcall.train(args.data, args.out, args.test_games)))
        print(f"saved to {args.out}/")
    else:
        playcall.predict(args.model, args.data).to_csv(args.out or sys.stdout, index=False)
    return 0

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["playcall"]:
        return playcall_main(argv[1:])
    p = argparse.ArgumentParser(prog="football-cv", epilog="also: football-cv playcall {features,train,predict} (see docs/playcall.md)")
    p.add_argument("stage", choices=["ingest", *STAGES, "all"], help="pipeline stage to run")
    p.add_argument("play_id")
    p.add_argument("--data", default=None)
    p.add_argument("--video", default=None, help="film to ingest (for the `ingest` stage)")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        # argparse exits on --help (code 0) or an invalid stage (code 2);
        # surface that code as the return value instead of propagating.
        return int(exc.code or 0)
    data = Path(args.data) if args.data else paths.data_dir()
    if args.stage == "ingest":
        ingest.run(args.play_id, data, video=args.video)
        return 0
    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        STAGES[s](args.play_id, data)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
