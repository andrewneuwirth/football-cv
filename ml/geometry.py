# ml/geometry.py
"""Generic football-field geometry: snap detection, offense/defense split,
line of scrimmage, and formation reads. Pure position math only."""
from __future__ import annotations
from pathlib import Path
import numpy as np

# -- snap detection thresholds ------------------------------------------------
LOW_RUN = 15          # frames of low motion required before the snap
SPIKE_RUN = 3         # consecutive spike frames required to call the snap
LOW_SPEED = 0.75      # yd/s: below this counts as "still" (floor for adaptive thr)
SPIKE_SPEED = 1.5     # yd/s: above this counts as a snap-level spike (floor)
SMOOTH_WIN = 9        # centered moving-average window on positions (noise supp.)
PRE_SNAP_WIN = 10     # frames before the snap averaged for alignment positions

# -- split / line-of-scrimmage geometry --------------------------------------
FRONT_CROWD = 1.0     # yd from a front edge that counts as crowding it
FRONT_BAND = 1.5      # yd from a front that counts as "on the line"
GAP_LO, GAP_HI = 0.6, 2.3   # plausible neutral-zone width (yd)
GAP_BONUS = 3.0
BACKFIELD_LO, BACKFIELD_HI = 3.0, 8.0   # QB/RB depth behind the O front
BACKFIELD_DY = 8.0                       # ...near the formation middle
BACKFIELD_BONUS = 3.0

# -- formation-read weights and bands ----------------------------------------
LINE_BAND = 1.2       # yd from a side's front edge that reads as "on the line"
LINE_MAX_DY = 13.0    # an on-the-line body sits within this of the formation's
                      # lateral center; wider bodies are officials / sideline subs
W_LINE = 1.4          # offense-likeness per on-the-line body (signal 1)
W_BACKFIELD = 3.0     # offense-likeness for having a compact central backfield
W_CONTIG = 1.2        # offense-likeness per shoulder-to-shoulder line body
W_CORNER = 1.5        # DEFENSE-likeness per wide-deep corner (signal 2 inverse)
W_FRONT_FRAC = 2.0    # offense-likeness per unit of front-crowding fraction
BACKFIELD_MIN = 1     # a backfield cluster needs at least this many bodies
BACKFIELD_MAX = 3     # ...and at most this (QB + 1-2 backs); more = not a cluster
BACKFIELD_CENTER_DY = 6.0  # backfield bodies sit within this of the front's center
CONTIG_DEPTH = 1.2    # line bodies share a common depth within this band
CONTIG_GAP = 2.6      # adjacent line bodies sit within this lateral gap
CONTIG_MIN = 3        # a contiguous run this long counts as a line block
CONTIG_MIN_SPAN = 3.5 # ...and it must stretch at least this wide laterally
CONTIG_STRADDLE = 4.0 # ...and the run must straddle the formation center within
NARROW_SPAN = 8.0     # a formation side spans far more than this laterally
W_NARROW = 8.0        # DEFENSE-likeness penalty for a narrow (knotted) side

CORNER_DY = 8.0            # |dy| off the side's center that reads wide
CORNER_DEPTH_LO = 1.5      # a corner presses off its front by 1.5-9 yd;
CORNER_DEPTH_HI = 9.0      # split receivers sit ON the line, backs sit central


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def track_positions(field: dict) -> dict[int, dict[int, tuple[float, float]]]:
    """{track_id: {frame_i: (x, y)}} from field coords (defensive parsing)."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for t in field.get("tracks", []) or []:
        try:
            tid = int(t["track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        frames: dict[int, tuple[float, float]] = {}
        for fr in t.get("frames", []) or []:
            try:
                frames[int(fr["i"])] = (float(fr["x"]), float(fr["y"]))
            except (KeyError, TypeError, ValueError):
                continue
        if frames:
            out[tid] = frames
    return out


def smooth_positions(
    positions: dict[int, dict[int, tuple[float, float]]], window: int = SMOOTH_WIN
) -> dict[int, dict[int, tuple[float, float]]]:
    """Centered moving average of each track's positions (noise suppression)."""
    half = max(0, window // 2)
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for tid, frames in positions.items():
        idxs = sorted(frames)
        pts = [frames[i] for i in idxs]
        sm: dict[int, tuple[float, float]] = {}
        for k, i in enumerate(idxs):
            lo, hi = max(0, k - half), min(len(pts), k + half + 1)
            seg = pts[lo:hi]
            sm[i] = (
                sum(p[0] for p in seg) / len(seg),
                sum(p[1] for p in seg) / len(seg),
            )
        out[tid] = sm
    return out


def find_snap(speeds: list[float]) -> tuple[int, bool, float]:
    """(snap_frame, confident, low_thr) from a team-wide mean-speed series.

    The first sustained mean-speed spike after a low-motion stretch, else the
    largest single jump. Returns the still-period ``low_thr`` alongside the
    detected snap frame; the boolean flags whether the detection is confident.
    """
    n = len(speeds)
    low_thr, spike_thr = LOW_SPEED, SPIKE_SPEED
    if n >= 2:
        body = sorted(speeds[1:])  # frame 0 has no previous frame -> 0.0
        floor = body[int(0.2 * (len(body) - 1))]
        peak = body[int(0.9 * (len(body) - 1))]
        low_thr = max(LOW_SPEED, floor + 0.2 * (peak - floor))
        spike_thr = max(SPIKE_SPEED, floor + 0.5 * (peak - floor))
    armed = False
    low_run = 0
    for i, v in enumerate(speeds):
        if not armed:
            low_run = low_run + 1 if v <= low_thr else 0
            if low_run >= LOW_RUN:
                armed = True
            continue
        if v >= spike_thr and all(
            speeds[j] >= spike_thr for j in range(i, min(i + SPIKE_RUN, n))
        ) and i + SPIKE_RUN <= n:
            return i, True, low_thr
    if n >= 2:
        jumps = [speeds[i] - speeds[i - 1] for i in range(1, n)]
        return 1 + max(range(len(jumps)), key=jumps.__getitem__), False, low_thr
    return 0, False, low_thr


def alignment_positions(
    positions: dict[int, dict[int, tuple[float, float]]], snap: int
) -> dict[int, tuple[float, float]]:
    """Each track's pre-snap alignment: mean over the PRE_SNAP_WIN frames
    before the snap; nearest recorded frame when a track has no pre-snap data
    (fragments far from the snap still get a best-effort position)."""
    out: dict[int, tuple[float, float]] = {}
    for tid, frames in positions.items():
        window = [
            frames[i] for i in range(max(0, snap - PRE_SNAP_WIN), snap) if i in frames
        ]
        if window:
            out[tid] = (
                sum(p[0] for p in window) / len(window),
                sum(p[1] for p in window) / len(window),
            )
    for tid, frames in positions.items():
        if tid not in out:
            nearest = min(frames, key=lambda i: abs(i - snap))
            out[tid] = frames[nearest]
    return out


def split_and_los(
    pts: list[tuple[float, float]],
    lo_bound: float | None = None,
    hi_bound: float | None = None,
) -> tuple[int, float, float, float]:
    """(split_k, los_x, left_front, right_front) for pts sorted by x.

    Optional bounds restrict candidate boundaries to midpoints inside
    [lo_bound, hi_bound] — used to let jersey colors pick the region and
    geometry pick the exact neutral zone within it.

    The line of scrimmage is the gap where many bodies crowd both facing edges
    (the two fronts stack on the ball), the gap looks like a neutral zone
    (~0.5-2.5 yd), and — decisively on real film, where inner defensive row
    gaps can crowd just as hard — the offense side of the split has a
    QB/backfield: a body 3-8 yd behind its front near the formation's middle.
    No balance term: clutter makes cluster sizes meaningless.
    """
    xs = [p[0] for p in pts]
    n = len(xs)
    min_side = max(1, min(3, n // 2))
    best = (None, -float("inf"))
    for k in range(min_side, n - min_side + 1):
        lf, rf = xs[k - 1], xs[k]
        gap = rf - lf
        mid = (lf + rf) / 2.0
        if lo_bound is not None and not (lo_bound <= mid <= hi_bound):
            continue
        crowd_l = sum(1 for v in xs[:k] if lf - v <= FRONT_CROWD)
        crowd_r = sum(1 for v in xs[k:] if v - rf <= FRONT_CROWD)
        score = float(crowd_l + crowd_r)
        if GAP_LO <= gap <= GAP_HI:
            score += GAP_BONUS
        elif gap > 3.5:
            score -= 2.0
        # Offense side of this candidate = higher on-front fraction; a real
        # line of scrimmage puts a backfield behind that front.
        frac_l = crowd_l / max(1, k)
        frac_r = crowd_r / max(1, n - k)
        if frac_l >= frac_r:
            o_side, front, sign, crowd_o = pts[:k], lf, -1.0, crowd_l
        else:
            o_side, front, sign, crowd_o = pts[k:], rf, 1.0, crowd_r
        mid_y = _median([p[1] for p in o_side])
        if crowd_o >= 3 and any(
            BACKFIELD_LO <= (p[0] - front) * sign <= BACKFIELD_HI
            and abs(p[1] - mid_y) <= BACKFIELD_DY
            for p in o_side
        ):
            score += BACKFIELD_BONUS
        if score > best[1]:
            best = (k, score)
    k = best[0] if best[0] is not None else n // 2
    lf, rf = xs[k - 1], xs[k]
    return k, (lf + rf) / 2.0, lf, rf


def _front_fraction(xs: list[float], front: float, toward: float) -> float:
    """Fraction of a cluster's bodies within FRONT_BAND of its front edge."""
    if not xs:
        return 0.0
    on = sum(1 for v in xs if abs(v - front) <= FRONT_BAND)
    return on / len(xs)


def wide_deep_count(pts: list[tuple[float, float]], front: float,
                    behind_sign: float) -> int:
    """Bodies split wide AND off their own front line = corners = defense.

    The offense's wide bodies (receivers) align on/near the line of scrimmage
    and its deep bodies (QB/backs) align centrally — only a defense posts wide
    bodies yards behind its front. The most reliable O/D tell on cluttered film.
    """
    if len(pts) < 3:
        return 0
    center = _median([p[1] for p in pts])
    return sum(
        1 for x, y in pts
        if abs(y - center) >= CORNER_DY
        and CORNER_DEPTH_LO <= (front - x) * behind_sign <= CORNER_DEPTH_HI
    )


def line_count(
    pts: list[tuple[float, float]], front: float, center_y: float | None = None
) -> int:
    """Bodies within LINE_BAND of a side's front edge = on the line of scrimmage.

    Signal 1 (7-on-the-line): the offense stacks ~7 bodies on the ball (line +
    on-line receivers/tight ends); a defense fronts only 3-4 down linemen. The
    side with clearly more bodies at its own front is the offense — the strongest,
    purely rule-based O/D tell, and it survives a mid-play frame better than any
    depth heuristic because the LINE band hugs the ball the whole rep.

    When a formation lateral center is given, a body counts only if it sits
    within LINE_MAX_DY of it — the real front lives in the tackle box; a knot
    of bodies parked at a sideline (a bunch / a track cluster) is not a front.
    """
    return sum(
        1 for x, y in pts
        if abs(x - front) <= LINE_BAND
        and (center_y is None or abs(y - center_y) <= LINE_MAX_DY)
    )


def backfield_cluster(
    pts: list[tuple[float, float]], front: float, behind_sign: float
) -> bool:
    """True when a side has a COMPACT central backfield behind its front.

    Signal 2: the offense hides a tight 1-3 body cluster (QB + 1-2 backs)
    3-8 yd directly behind its front's lateral center. A defense has no such
    thing — its deep bodies are the secondary, spread WIDE across the field.
    So we require a small count of bodies in the backfield depth band that also
    sit near the front's lateral center; a wide spread back there is a defense
    and fails this test (it is caught instead by wide_deep_count).
    """
    if len(pts) < 3:
        return False
    center = _median([p[1] for p in pts])
    behind = [
        y for x, y in pts
        if BACKFIELD_LO <= (front - x) * behind_sign <= BACKFIELD_HI
        and abs(y - center) <= BACKFIELD_CENTER_DY
    ]
    return BACKFIELD_MIN <= len(behind) <= BACKFIELD_MAX


def ol_contiguity(
    pts: list[tuple[float, float]], front: float, center_y: float | None = None
) -> int:
    """Longest run of shoulder-to-shoulder bodies at the front (the line block).

    Signal 3: the offense packs ~5 linemen in one contiguous lateral row at a
    common depth; a defensive front is gapped (interior gaps, edges set off the
    ball). Take the bodies near the front depth, sort by lateral, and return the
    longest run where each adjacent pair sits within CONTIG_GAP — the line row is
    the only place on the field this many bodies line up shoulder to shoulder.

    A run is only counted when it spans a real lateral width (>= CONTIG_MIN_SPAN
    yd): a knot of bodies crammed into a couple of yards at a sideline is a track
    cluster / a bunch, not a line that stretches across the formation.
    When a formation center is given the run must also STRADDLE it (bodies on
    both sides of center) — a line centers on the ball, a sideline knot does not.
    """
    on = sorted(
        (y for x, y in pts if abs(x - front) <= CONTIG_DEPTH)
    )
    if len(on) < 2:
        return len(on)
    best_run = 1
    run_start = 0
    for i in range(1, len(on)):
        if on[i] - on[run_start] and on[i] - on[i - 1] > CONTIG_GAP:
            run_start = i
        seg = on[run_start:i + 1]
        run_len = len(seg)
        wide_enough = seg[-1] - seg[0] >= CONTIG_MIN_SPAN
        straddles = center_y is None or (seg[0] <= center_y + CONTIG_STRADDLE
                                         and seg[-1] >= center_y - CONTIG_STRADDLE)
        if wide_enough and straddles and run_len > best_run:
            best_run = run_len
    return best_run


def offense_likeness(
    pts: list[tuple[float, float]], front: float, behind_sign: float,
    center_y: float | None = None,
) -> float:
    """Weighted offense-likeness score for one side of the line of scrimmage.

    Combines the rule-based football signals, each weighted by reliability:
    7-on-the-line (strongest), a compact central backfield, and a contiguous
    line block push a side toward OFFENSE; wide-deep corners (the secondary)
    pull it toward DEFENSE. The front-crowding fraction is folded in at low
    weight as the legacy corroborator. Higher score = more offense-like; the
    caller compares the two sides. center_y (the formation's lateral center,
    over BOTH teams) anchors the front near the tackle box so a sideline knot
    cannot pose as an offensive front.
    """
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    # A real formation-side spreads across the field (line + split receivers span
    # tens of yards); a side whose bodies are all crammed into a few yards of
    # lateral width is a track cluster / bunch pinned at a sideline, never an
    # 11-man offense. A side that narrow is physically incapable of fielding an
    # offensive FRONT, so it earns NONE of the offense-front bonuses (7-on-line,
    # line block, backfield) — those would be false positives on a knot (5 bodies
    # within CONTIG_GAP read as a "line") — and takes a flat DEFENSE-likeness
    # penalty. Zeroing the bonuses (rather than adding then subtracting) is what
    # makes the knot lose decisively: line(7) + contig(6) can otherwise swamp any
    # fixed penalty.
    span = (max(ys) - min(ys)) if len(ys) >= 2 else 0.0
    if span < NARROW_SPAN:
        score = -W_NARROW
        score -= W_CORNER * wide_deep_count(pts, front, behind_sign)
        score += W_FRONT_FRAC * _front_fraction(xs, front, front)
        return score
    score = W_LINE * line_count(pts, front, center_y)
    if backfield_cluster(pts, front, behind_sign):
        score += W_BACKFIELD
    contig = ol_contiguity(pts, front, center_y)
    if contig >= CONTIG_MIN:
        score += W_CONTIG * contig
    score -= W_CORNER * wide_deep_count(pts, front, behind_sign)
    score += W_FRONT_FRAC * _front_fraction(xs, front, front)
    return score
