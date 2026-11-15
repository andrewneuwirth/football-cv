"""Write examples/synthetic_snaps.csv: a made-up season for trying `football-cv playcall`.

Every row is random. Tendencies are invented but plausible (gun / empty / long
distance lean pass, two-back under center leans run, runs lean toward the
formation's strength and the motion), so the model has something to find.
It says nothing about any real team.

    python examples/synthetic_snaps.py [--games 10] [--seed 7] [--out examples/synthetic_snaps.csv]
"""
import argparse
import csv
import math
import random
from datetime import date, timedelta

COLUMNS = ["game", "game_date", "play_no", "qtr", "down", "dist", "yard_ln", "hash", "off_form", "off_str",
           "backfield", "qb_set", "box", "motion_dir", "prev_type", "prev_gain", "run_pass", "play_dir"]


def _form(rng, backfield):
    n = max(1, min(5, 4 - backfield + rng.choice([0, 0, 1])))  # players outside the tackles
    strong = rng.randint(math.ceil(n / 2), min(3, n))
    side = "bal" if strong == n - strong else rng.choice("LR")
    return f"{strong}x{n - strong}", side


def snaps(games=10, seed=7):
    rng = random.Random(seed)
    start = date(2025, 9, 5)
    for g in range(games):
        lean = rng.gauss(0, 0.35)                 # this game's game plan: a bit more run or pass
        prev_type, prev_gain = "", ""
        for n in range(1, rng.randint(52, 68) + 1):
            down = rng.choices([1, 2, 3, 4], [40, 30, 22, 8])[0]
            dist = 10 if down == 1 and rng.random() < 0.8 else rng.randint(1, 15)
            yard_ln = max(1, min(99, round(rng.gauss(50, 20))))
            gun = rng.random() < 0.6
            backfield = rng.choices([0, 1, 2], [15, 70, 15])[0] if gun else rng.choices([1, 2, 3], [40, 50, 10])[0]
            form, side = _form(rng, backfield)
            box = max(5, min(9, 6 + (backfield >= 2) + (not gun) + rng.choice([-1, 0, 0, 1])))
            motion = rng.choices(["none", "L", "R"], [60, 20, 20])[0]
            z = (-0.7 + lean + 1.2 * gun - 0.9 * (backfield - 1) + 0.13 * (dist - 7) + 0.6 * (down == 3)
                 + 0.4 * (down == 4 and dist > 3) - 0.8 * (yard_ln > 95) + 0.25 * (prev_type == "Run") + rng.gauss(0, 0.6))
            run_pass = "Pass" if rng.random() < 1 / (1 + math.exp(-z)) else "Run"
            play_dir = ""
            if run_pass == "Run":
                r = 0.8 * ((side == "R") - (side == "L")) + 0.6 * ((motion == "R") - (motion == "L"))
                play_dir = "R" if rng.random() < 1 / (1 + math.exp(-r)) else "L"
            yield {"game": f"game-{g + 1:02d}", "game_date": (start + timedelta(days=7 * g)).isoformat(), "play_no": n,
                   "qtr": min(4, 1 + 4 * (n - 1) // 64), "down": down, "dist": dist, "yard_ln": yard_ln,
                   "hash": rng.choice("LMR"), "off_form": form, "off_str": side, "backfield": backfield,
                   "qb_set": "gun" if gun else "under", "box": box, "motion_dir": motion,
                   "prev_type": prev_type, "prev_gain": prev_gain, "run_pass": run_pass, "play_dir": play_dir}
            prev_type, prev_gain = run_pass, max(-5, round(rng.gauss(6 if run_pass == "Pass" else 4, 6)))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--games", type=int, default=10)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--out", default="examples/synthetic_snaps.csv")
    args = a.parse_args()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(snaps(args.games, args.seed))
    print(f"wrote {args.out}")
