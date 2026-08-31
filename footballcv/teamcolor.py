"""Jersey-color team assignment (the `teams` stage).

Writes team_colors.json per CONTRACT.md: sample every ~5th frame, crop each
track's jersey region, mask field pixels, describe the crop by its mean Lab
color, catch striped refs before clustering, KMeans k=2 (numpy, seeded) on the
rest, majority-vote each track into "A" (darker) / "B" (lighter) / "ref" /
"unknown", and surface sustained per-frame cluster switches as flip_events
(a strong ID-swap signal).

No sklearn — KMeans is implemented here with numpy and a deterministic init.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- tunables
# Jersey crop: upper ~40% of box height (skipping the very top, which is mostly
# helmet), center ~60% of box width.
JERSEY_TOP = 0.12          # fraction of box height where the crop starts
JERSEY_BOT = 0.44          # fraction of box height where the crop ends
JERSEY_HALF_W = 0.30       # half-width as fraction of box width (center 60%)

# Field mask: green-dominant HSV pixels (OpenCV hue 0-179).
GREEN_H_LO = 30
GREEN_H_HI = 95
GREEN_S_MIN = 35
GREEN_V_MIN = 25

MIN_USABLE_PX = 30         # skip crops with fewer non-field pixels
MIN_VOTES = 3              # tracks with fewer sampled votes -> "unknown"

# Ref detection (before clustering). Refs wear a VERTICAL black-and-white
# striped shirt: on the torso, column intensity OSCILLATES horizontally at a
# characteristic spatial frequency, and every torso row shows the *same*
# alternation (vertical coherence). A solid jersey -- even one with a big
# contrasting number -- does not: a number is one wide, localized blob (a
# single low-frequency cycle confined to a few rows), and busy sideline crowd
# contrast is horizontally incoherent. We measure, per crop:
#   - per-row autocorrelation PERIOD (px between stripes) and its STRENGTH,
#     and how CONSISTENT that period is across rows (stripes span the torso;
#     a number does not);
#   - the GLOBAL column-mean-profile autocorrelation strength (`gstr`): true
#     vertical stripes make the whole-torso column profile itself periodic,
#     while a number or crowd contrast leaves it near-flat (gstr ~ 0);
#   - NCYC, stripe cycles across the torso width (a real ref torso is ~3-5
#     stripe pairs wide; a wide busy crop shows too many);
#   - BOTH, the bimodal support = min(near-black fraction, near-white fraction)
#     (a striped shirt has substantial dark AND light; a dark/white player is
#     unimodal).
# A track is called ref by majority of well-resolved crops meeting one of three
# evidence branches (global coherence / bimodal-backed / tight-period), chosen
# by sweeping against a set of manually-labeled Ref and non-Ref crops: the
# tuned gate reaches high recall at full precision, with the few misses being a
# blurred official and a "ref" whose visible torso is a solid dark jersey with
# a white number -- catching either would flag real numbered players.
REF_PER_LO, REF_PER_HI = 3.0, 17.0    # plausible stripe period (px), scale-free
REF_NCYC_LO, REF_NCYC_HI = 2.4, 5.2   # stripe cycles across torso width
REF_CONS_MIN = 0.55        # share of rows agreeing on the modal period
REF_BOTH_MIN = 0.03        # min(dark frac, light frac): bimodal support
REF_GLOBAL_MIN = 0.12      # column-profile autocorr strength => vertical stripe
REF_ROW_STR_MIN = 0.18     # per-row autocorr strength for the backup branches
REF_MIN_ROWS = 6           # torso rows needed to trust a crop's stripe read
REF_MIN_W = 14             # torso width (px) needed to resolve stripes
REF_MIN_RESOLVED = 5       # well-resolved crops needed to judge a track
REF_TRACK_FRAC = 0.40      # track -> "ref" when >= this share of votes are ref

FLIP_MIN_STAY = 15         # a switch must hold for >= this many sampled votes
FLIP_MIN_PREFIX = 3        # ... and have at least this many votes before it

KMEANS_SEED = 0
KMEANS_ITERS = 100


# ---------------------------------------------------------------- primitives


def green_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of field (green-dominant) pixels in a BGR crop."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    return (
        (h >= GREEN_H_LO) & (h <= GREEN_H_HI)
        & (s >= GREEN_S_MIN) & (v >= GREEN_V_MIN)
    )


def jersey_crop(frame: np.ndarray, box: list[float]) -> np.ndarray | None:
    """Crop the jersey region of a track box; None if it falls off-frame."""
    x1, y1, x2, y2 = box
    bh, bw = y2 - y1, x2 - x1
    if bh <= 0 or bw <= 0:
        return None
    cx = (x1 + x2) / 2.0
    top = int(round(y1 + JERSEY_TOP * bh))
    bot = int(round(y1 + JERSEY_BOT * bh))
    left = int(round(cx - JERSEY_HALF_W * bw))
    right = int(round(cx + JERSEY_HALF_W * bw))
    H, W = frame.shape[:2]
    top, bot = max(0, top), min(H, bot)
    left, right = max(0, left), min(W, right)
    if bot - top < 2 or right - left < 2:
        return None
    return frame[top:bot, left:right]


def _autocorr_period(sig: np.ndarray) -> tuple[int, float] | None:
    """Dominant (period_px, strength) of a mean-removed 1D signal, or None.

    strength is the normalized autocorrelation at the first strong peak away
    from lag 0. Vertical stripes give a sharp peak at a small lag (~2*stripe_w
    and its harmonics); a single number blob or flat texture gives no clear
    peak (or one only at a large lag).
    """
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 8:
        return None
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] <= 1e-9:
        return None
    ac = ac / ac[0]
    best_lag = None
    best_val = -2.0
    for lag in range(2, n // 2):
        if (
            ac[lag] > ac[lag - 1]
            and ac[lag] >= ac[min(lag + 1, n - 1)]
            and ac[lag] > best_val
        ):
            best_val = ac[lag]
            best_lag = lag
    if best_lag is None:
        return None
    return best_lag, best_val


def stripe_metrics(crop: np.ndarray) -> dict | None:
    """Vertical-stripe descriptors for one jersey crop, or None if unresolved.

    Restricts to the central torso band (avoiding cap/pants and crop edges),
    keeps only rows that are mostly on-body (not masked field), and returns the
    median per-row autocorrelation period/strength, the per-row period
    consistency, the global column-profile autocorrelation strength (`gstr`),
    the stripe-cycle count across the torso width (`ncyc`), and the bimodal
    support `both` = min(near-black fraction, near-white fraction).
    """
    h, w = crop.shape[:2]
    if h < 10 or w < REF_MIN_W:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
    usable = ~green_mask(crop)
    r0, r1 = int(0.12 * h), int(0.92 * h)
    c0, c1 = int(0.06 * w), int(0.94 * w)
    g = gray[r0:r1, c0:c1]
    u = usable[r0:r1, c0:c1]
    if g.shape[0] < 6 or g.shape[1] < 10:
        return None
    row_ok = u.mean(axis=1) >= 0.6      # rows that sit on the body, not field
    if int(row_ok.sum()) < REF_MIN_ROWS:
        return None
    G = g[row_ok]
    periods: list[int] = []
    strengths: list[float] = []
    for row in G:
        r = _autocorr_period(row)
        if r is not None:
            periods.append(r[0])
            strengths.append(r[1])
    if len(periods) < REF_MIN_ROWS:
        return None
    per = np.asarray(periods, dtype=np.float64)
    med_p = float(np.median(per))
    consistent = float(np.mean(np.abs(per - med_p) <= 1.5))
    gr = _autocorr_period(G.mean(axis=0))   # global column-profile periodicity
    gstr = float(gr[1]) if gr is not None else 0.0
    body = g[u]
    blo = float((body < 75).mean())
    bhi = float((body > 155).mean())
    ncyc = G.shape[1] / med_p if med_p > 0 else 0.0
    return {
        "period": med_p,
        "cons": consistent,
        "rowstr": float(np.median(strengths)),
        "gstr": gstr,
        "ncyc": ncyc,
        "both": min(blo, bhi),
    }


def describe_crop(crop: np.ndarray) -> dict | None:
    """Descriptor for one jersey crop, or None if too few usable pixels.

    Returns {"lab": mean Lab (float, cv2 8-bit scale), "rgb": mean RGB,
    "sat": mean HSV saturation, "std": grayscale std, "stripe": stripe_metrics
    dict or None, "n": usable px}.
    """
    usable = ~green_mask(crop)
    n = int(usable.sum())
    if n < MIN_USABLE_PX:
        return None
    px = crop[usable].reshape(-1, 1, 3)  # (n,1,3) BGR uint8
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)[usable]
    bgr = px.reshape(-1, 3).astype(np.float64)
    return {
        "lab": lab.mean(axis=0),
        "rgb": bgr.mean(axis=0)[::-1],
        "sat": float(hsv[:, 1].mean()),
        "std": float(gray.std()),
        "stripe": stripe_metrics(crop),
        "n": n,
    }


def is_stripe_ref(m: dict) -> bool:
    """Vertical-stripe test on one crop's stripe_metrics (see tunables).

    Gate first on the shape of the periodicity (plausible period, cycle count,
    bimodal support, cross-row consistency), then accept via one of three
    evidence branches: strong global vertical coherence; weaker global backed
    by bimodal + per-row strength; or a tight, highly consistent small period
    with strong per-row response (recovers small, dark-topped refs whose global
    profile is washed out).
    """
    if not (REF_PER_LO <= m["period"] <= REF_PER_HI):
        return False
    if m["both"] < REF_BOTH_MIN:
        return False
    if not (REF_NCYC_LO <= m["ncyc"] <= REF_NCYC_HI):
        return False
    if m["cons"] < REF_CONS_MIN:
        return False
    global_branch = m["gstr"] >= REF_GLOBAL_MIN
    bimodal_branch = (
        m["gstr"] >= 0.07 and m["both"] >= 0.15 and m["rowstr"] >= REF_ROW_STR_MIN
    )
    tight_branch = (
        m["period"] <= 6.0 and m["cons"] >= 0.70 and m["rowstr"] >= 0.22
    )
    return bool(global_branch or bimodal_branch or tight_branch)


def track_is_ref(descs: list[dict]) -> bool:
    """Whole-track ref decision from its per-crop descriptors.

    Judges on the MEDIAN stripe metrics of the well-resolved crops (robust to
    per-frame occlusion/blur noise), requiring a minimum count of resolved
    crops so small/blurred/occluded tracks defer to non-ref.
    """
    ms = [d["stripe"] for d in descs if d.get("stripe") is not None]
    if len(ms) < REF_MIN_RESOLVED:
        return False
    med = {
        k: float(np.median([m[k] for m in ms]))
        for k in ("period", "cons", "rowstr", "gstr", "ncyc", "both")
    }
    return is_stripe_ref(med)


def kmeans2(
    X: np.ndarray, seed: int = KMEANS_SEED, n_iter: int = KMEANS_ITERS
) -> tuple[np.ndarray, np.ndarray]:
    """KMeans k=2 (Lloyd's) with a deterministic init. Returns (centers, labels).

    Init: centers seeded at the means of the bottom/top deciles along the first
    feature (Lab L => dark vs light), which is deterministic and well-matched to
    the dark-vs-light jersey split. `seed` only breaks exact ties.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("kmeans2 needs a (n>=2, d) array")
    order = np.argsort(X[:, 0], kind="stable")
    k = max(1, len(X) // 10)
    centers = np.stack([X[order[:k]].mean(axis=0), X[order[-k:]].mean(axis=0)])
    if np.allclose(centers[0], centers[1]):  # degenerate: nudge deterministically
        rng = np.random.default_rng(seed)
        centers = centers + rng.normal(0.0, 1e-3, size=centers.shape)
    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for j in (0, 1):
            members = X[new_labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels, centers = new_labels, new_centers
    return centers, labels


def detect_flip(votes: list[tuple[int, str]]) -> int | None:
    """Frame of a sustained cluster switch in one track's (frame, "A"|"B") votes.

    A flip is a switch that STAYS switched for the rest of the track for
    >= FLIP_MIN_STAY sampled votes, with >= FLIP_MIN_PREFIX earlier votes whose
    majority is the other team. Returns the frame at the switch, else None.
    """
    if len(votes) < FLIP_MIN_STAY + FLIP_MIN_PREFIX:
        return None
    teams = [t for _, t in votes]
    # length of the final constant run
    run = 1
    while run < len(teams) and teams[-run - 1] == teams[-1]:
        run += 1
    if run < FLIP_MIN_STAY or run >= len(teams):
        return None
    prefix = teams[:-run]
    if len(prefix) < FLIP_MIN_PREFIX:
        return None
    other = "A" if teams[-1] == "B" else "B"
    if prefix.count(other) <= len(prefix) / 2:
        return None
    return votes[len(teams) - run][0]


# ---------------------------------------------------------------- stage


def run(
    play_id: str,
    data_dir: Path,
    sample_every: int = 5,
    debug_dir: Path | None = None,
) -> None:
    """Assign teams by jersey color; writes team_colors.json (CONTRACT.md)."""
    pdir = Path(data_dir) / "plays" / play_id
    clip = pdir / "clip.mp4"
    tracks_path = pdir / "tracks.json"
    if not clip.is_file():
        raise FileNotFoundError(clip)
    with tracks_path.open("r", encoding="utf-8") as f:
        tracks_doc = json.load(f)

    # frame -> [(track_id, box)] for sampled frames only
    by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    track_ids: list[int] = []
    for t in tracks_doc.get("tracks", []):
        tid = t["track_id"]
        track_ids.append(tid)
        for fr in t["frames"]:
            i = fr["i"]
            if i % sample_every == 0:
                by_frame.setdefault(i, []).append((tid, fr["box"]))

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {clip}")

    # per-track ordered samples: (frame, descriptor)
    samples: dict[int, list[tuple[int, dict]]] = {tid: [] for tid in track_ids}
    n_debug_saved = 0
    i = 0
    while True:
        boxes = by_frame.get(i)
        if boxes is None:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, box in boxes:
                crop = jersey_crop(frame, box)
                if crop is None:
                    continue
                desc = describe_crop(crop)
                if desc is None:
                    continue
                samples[tid].append((i, desc))
                if debug_dir is not None and n_debug_saved < 400 and i % 25 == 0:
                    Path(debug_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(Path(debug_dir) / f"{play_id}_t{tid:03d}_f{i:04d}.png"),
                        crop,
                    )
                    n_debug_saved += 1
        i += 1
    cap.release()

    # ref detection BEFORE clustering: a whole-track vertical-stripe decision on
    # the median stripe metrics of each track's well-resolved crops. A confident
    # ref never enters the A/B clustering (that is how muddy refs used to be
    # mislabeled as a high-confidence team). Non-ref tracks' samples cluster with
    # KMeans k=2 in Lab.
    ref_tracks: set[int] = {
        tid for tid, seq in samples.items()
        if track_is_ref([d for _, d in seq])
    }
    flat: list[tuple[int, int, dict]] = []  # (tid, frame, desc) for non-ref samples
    ref_votes: dict[int, list[int]] = {tid: [] for tid in track_ids}  # frames
    for tid, seq in samples.items():
        for frame_i, desc in seq:
            if tid in ref_tracks:
                ref_votes[tid].append(frame_i)
            else:
                flat.append((tid, frame_i, desc))

    cluster_team: dict[tuple[int, int], str] = {}  # (tid, frame) -> "A"|"B"
    clusters_rgb = {"A": [0, 0, 0], "B": [255, 255, 255]}
    if len(flat) >= 2:
        X = np.stack([d["lab"] for _, _, d in flat])
        centers, labels = kmeans2(X, seed=KMEANS_SEED)
        a_idx = int(np.argmin(centers[:, 0]))  # "A" = darker cluster (lower L)
        for (tid, frame_i, _), lab in zip(flat, labels):
            cluster_team[(tid, frame_i)] = "A" if lab == a_idx else "B"
        for team, j in (("A", a_idx), ("B", 1 - a_idx)):
            member_rgb = [
                d["rgb"] for (tid, frame_i, d), lab in zip(flat, labels) if lab == j
            ]
            if member_rgb:
                clusters_rgb[team] = [
                    int(round(v)) for v in np.mean(member_rgb, axis=0)
                ]

    # per-track majority vote + flip events
    assignments: dict[str, dict] = {}
    flip_events: list[dict] = []
    counts = {"A": 0, "B": 0, "ref": 0, "unknown": 0}
    for tid in track_ids:
        seq = samples[tid]
        votes: list[tuple[int, str]] = []
        for frame_i, _ in seq:
            if (tid, frame_i) in cluster_team:
                votes.append((frame_i, cluster_team[(tid, frame_i)]))
            elif frame_i in set(ref_votes[tid]):
                votes.append((frame_i, "ref"))
        color = (
            [int(round(v)) for v in np.mean([d["rgb"] for _, d in seq], axis=0)]
            if seq
            else [128, 128, 128]
        )
        if len(votes) < MIN_VOTES:
            team, conf = "unknown", 0.0
        else:
            tally = {"A": 0, "B": 0, "ref": 0}
            for _, v in votes:
                tally[v] += 1
            # refs rarely win an outright majority of noisy per-frame votes, so
            # a sustained ref-vote share is enough to call the track a ref
            if tally["ref"] / len(votes) >= REF_TRACK_FRAC:
                team = "ref"
            elif tally["A"] or tally["B"]:
                team = "A" if tally["A"] >= tally["B"] else "B"  # tie -> A
            else:
                team = "unknown"
            conf = tally.get(team, 0) / len(votes) if team != "unknown" else 0.0
        counts[team] += 1
        assignments[str(tid)] = {
            "team": team,
            "confidence": round(conf, 4),
            "color": color,
        }
        if team in ("A", "B"):
            ab_votes = [(f, v) for f, v in votes if v in ("A", "B")]
            flip_frame = detect_flip(ab_votes)
            if flip_frame is not None:
                flip_events.append({"track_id": tid, "frame": flip_frame})

    flip_events.sort(key=lambda e: (e["frame"], e["track_id"]))
    out = {
        "play_id": play_id,
        "clusters": clusters_rgb,
        "assignments": assignments,
        "flip_events": flip_events,
    }
    out_path = pdir / "team_colors.json"
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    tmp.replace(out_path)

    n_samples = sum(len(s) for s in samples.values())
    print(
        f"[teams] {play_id}: A={counts['A']} B={counts['B']} ref={counts['ref']} "
        f"unknown={counts['unknown']} flips={len(flip_events)} "
        f"(tracks={len(track_ids)}, samples={n_samples})"
    )
