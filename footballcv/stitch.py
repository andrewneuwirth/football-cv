"""STITCH stage: merge track fragments belonging to one player into one track.

The tracker (track.py) splits a single player into multiple track IDs across
occlusions and pileups. That fragmentation is the biggest remaining error
source: it produces duplicate position labels (duplicate defenders), split
safeties, and the "swapped?" color-flip cases. This post-tracking stage stitches
fragments of the SAME body back together WITHOUT renumbering in a way that loses
manually-authored labels.

Pipeline position: runs AFTER track/teams/field, reading tracks.json (+
team_colors.json, + field_coords.json if present) and writing a new stitched
tracks.json. The original is backed up to ``tracks.prestitch.json`` first so the
pipeline (which reads tracks.json) sees the stitched result while the raw
tracker output remains recoverable.

ALGORITHM
---------
A *merge candidate* is an ordered pair of fragments (A, B) where:

  * TEMPORAL GAP: A's last real frame precedes B's first real frame, and the gap
    ``B.first - A.last`` is in ``[1, MAX_GAP]`` frames. A negative or zero gap
    (they share/overlap frames) is never a candidate — two simultaneously-alive
    tracks are two different bodies.
  * NO OVERLAP: A and B share no frame at all (checked on full frame sets, not
    just endpoints).
  * SAME TEAM: both A and B are jersey-color "A", or both "B" (from
    team_colors.json). Never merge across teams; never merge a "ref"; never
    merge when either side is "unknown" (too little color evidence to be safe).
  * MOTION CONSISTENCY: predict A's box forward to B's first frame using A's
    tail velocity (and predict B's box back using its head velocity); the
    predicted-vs-actual center gap must be small relative to the players' box
    size (scale-aware) — i.e. the two endpoints lie on one continuous
    trajectory. If field_coords.json is present, the field-space (yards) jump is
    also gated as a hard sanity bound.

Candidates are scored (smaller temporal + spatial cost is better) and applied
greedily, chaining A->B->C into merged *groups*. The group keeps the track_id of
its LONGEST fragment (most real frames) as the canonical id; the other fragments
are absorbed — their frames concatenated, sorted by frame index, and deduped
(keeping the real box over an interpolated one, higher confidence otherwise).

MANUAL LABELS (lossless, hard requirement)
------------------------------------------
For plays with manual_labels.json: if an absorbed fragment carried a manual
label, it is migrated to the group's canonical id. If merging two fragments
would union two DIFFERENT manual labels (a cross-player fusion — a bug), the
merge is BLOCKED. If the canonical already holds a label and an absorbed one
holds the SAME label, that is fine. After stitching, every manual label still
points at a live track and no label value is lost; manual_labels.json is
rewritten with migrated keys.
"""

from __future__ import annotations

import json
from pathlib import Path

# ------------------------------------------------------------------- tunables

# Temporal gap (frames) between A's last real frame and B's first real frame.
# 30 fps film: ~20 frames ~= 0.66 s, long enough to bridge a pileup/occlusion,
# short enough that two truly-different players rarely line up on one trajectory.
MAX_GAP = 20

# Motion consistency, scale-aware. The predicted-vs-actual endpoint gap is
# compared to the box scale (mean box height of the two endpoints). A player
# moving at a normal speed lands within a fraction of a body-height of the
# straight-line prediction; a wrong merge across two bodies overshoots.
# MOTION_TOL is the base tolerance (multiples of box height) plus a small
# per-gap-frame growth to allow for prediction drift over longer gaps.
MOTION_TOL_BASE = 1.6          # body-heights of slack at zero gap
MOTION_TOL_PER_FRAME = 0.08    # extra body-heights of slack per gap frame

# Absolute pixel floor so tiny far-field boxes still get usable slack.
MOTION_TOL_MIN_PX = 45.0

# Field-space hard sanity bound (yards) when field_coords.json is present: a
# merge whose endpoints are farther apart than this in yards is never plausible
# on one continuous trajectory within MAX_GAP frames.
FIELD_MAX_JUMP_YD = 12.0

# Minimum jersey-color confidence (from team_colors.json) required on BOTH
# fragments of a merge. A low-confidence color read is exactly the "uncertain"
# case the stage must refuse: the team could be wrong, so a same-"team" merge on
# a muddy read risks fusing two different bodies (and, empirically, perturbs the
# downstream position fit of a labeled track it lands on). Conservative by
# design — a confident fragment simply waits for a confident partner.
MIN_TEAM_CONF = 0.60

# How many trailing/leading real frames to use when estimating an endpoint's
# velocity (median of per-frame deltas over this window).
VEL_WINDOW = 5


# ------------------------------------------------------------------- helpers


def _load_json(path: Path):
    return json.loads(path.read_text())


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_height(box: list[float]) -> float:
    return max(box[3] - box[1], 1.0)


def _real_frames(track: dict) -> list[dict]:
    """Frames of a track that are real detections (not Kalman-coasted)."""
    return [f for f in track["frames"] if not f.get("interpolated", False)]


def _endpoint_velocity(frames: list[dict], at_end: bool) -> tuple[float, float]:
    """Median per-frame center velocity over the tail (at_end) or head window.

    Returns (vx, vy) in pixels/frame. Zero if fewer than two real frames.
    """
    window = frames[-VEL_WINDOW:] if at_end else frames[:VEL_WINDOW]
    if len(window) < 2:
        return 0.0, 0.0
    vxs: list[float] = []
    vys: list[float] = []
    for a, b in zip(window, window[1:]):
        di = b["i"] - a["i"]
        if di <= 0:
            continue
        ca, cb = _box_center(a["box"]), _box_center(b["box"])
        vxs.append((cb[0] - ca[0]) / di)
        vys.append((cb[1] - ca[1]) / di)
    if not vxs:
        return 0.0, 0.0
    vxs.sort()
    vys.sort()
    mid = len(vxs) // 2
    if len(vxs) % 2:
        return vxs[mid], vys[mid]
    return (vxs[mid - 1] + vxs[mid]) / 2.0, (vys[mid - 1] + vys[mid]) / 2.0


class _Fragment:
    """One track's stitch-relevant state."""

    def __init__(self, track: dict, team: str, team_conf: float = 0.0) -> None:
        self.track_id = int(track["track_id"])
        self.frames = track["frames"]
        self.team = team
        self.team_conf = team_conf
        real = _real_frames(track)
        self.real = real
        self.n_real = len(real)
        self.frame_set = {int(f["i"]) for f in track["frames"]}
        if real:
            self.first_i = int(real[0]["i"])
            self.last_i = int(real[-1]["i"])
            self.first_box = real[0]["box"]
            self.last_box = real[-1]["box"]
        else:
            # coast-only remnant (rare); treat with full frame set
            allf = track["frames"]
            self.first_i = int(allf[0]["i"]) if allf else 0
            self.last_i = int(allf[-1]["i"]) if allf else 0
            self.first_box = allf[0]["box"] if allf else [0, 0, 1, 1]
            self.last_box = allf[-1]["box"] if allf else [0, 0, 1, 1]


def _overlap(a: _Fragment, b: _Fragment) -> bool:
    """True if the two fragments occupy any common frame."""
    lo = a if len(a.frame_set) <= len(b.frame_set) else b
    hi = b if lo is a else a
    return any(i in hi.frame_set for i in lo.frame_set)


def _motion_gap(a: _Fragment, b: _Fragment) -> float | None:
    """Predicted-vs-actual pixel gap for candidate A->B, or None if implausible.

    Predict A forward from its last real box by its tail velocity to B's first
    frame, and predict B backward from its first real box by its head velocity
    to A's last frame; average the two endpoint errors. Returns the average
    pixel error, or None if it exceeds the scale-aware tolerance.
    """
    gap = b.first_i - a.last_i
    if gap <= 0:
        return None

    a_vx, a_vy = _endpoint_velocity(a.real, at_end=True)
    b_vx, b_vy = _endpoint_velocity(b.real, at_end=False)

    a_cx, a_cy = _box_center(a.last_box)
    b_cx, b_cy = _box_center(b.first_box)

    # A projected forward to B's first frame.
    pa_x = a_cx + a_vx * gap
    pa_y = a_cy + a_vy * gap
    err_a = ((pa_x - b_cx) ** 2 + (pa_y - b_cy) ** 2) ** 0.5

    # B projected backward to A's last frame.
    pb_x = b_cx - b_vx * gap
    pb_y = b_cy - b_vy * gap
    err_b = ((pb_x - a_cx) ** 2 + (pb_y - a_cy) ** 2) ** 0.5

    err = (err_a + err_b) / 2.0

    scale = (_box_height(a.last_box) + _box_height(b.first_box)) / 2.0
    tol_bodies = MOTION_TOL_BASE + MOTION_TOL_PER_FRAME * gap
    tol = max(tol_bodies * scale, MOTION_TOL_MIN_PX)
    if err > tol:
        return None
    return err


def _field_jump_ok(
    a: _Fragment,
    b: _Fragment,
    field_by_track: dict[int, dict[int, tuple[float, float]]],
) -> bool:
    """Hard field-space (yards) sanity bound on the A->B endpoint jump.

    True when field coords are missing for either endpoint (can't judge — defer
    to pixel motion) or when the yard-space jump is within FIELD_MAX_JUMP_YD.
    """
    fa = field_by_track.get(a.track_id)
    fb = field_by_track.get(b.track_id)
    if not fa or not fb:
        return True
    pa = fa.get(a.last_i)
    pb = fb.get(b.first_i)
    if pa is None or pb is None:
        return True
    ax, ay = pa
    bx, by = pb
    if any(v != v for v in (ax, ay, bx, by)):  # NaN projection: can't judge
        return True
    jump = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return jump <= FIELD_MAX_JUMP_YD


# ---------------------------------------------------------------- frame merge


def _merge_frames(tracks_by_id: dict[int, dict], ids: list[int]) -> list[dict]:
    """Concatenate frames of ``ids``, sort by frame index, dedupe per frame.

    On a frame collision (should be rare — merges never overlap — but coast
    frames can bracket a boundary) keep the real box over an interpolated one.
    """
    best: dict[int, dict] = {}
    for tid in ids:
        for f in tracks_by_id[tid]["frames"]:
            i = int(f["i"])
            cur = best.get(i)
            if cur is None:
                best[i] = f
                continue
            # prefer a real (non-interpolated) frame
            if cur.get("interpolated", False) and not f.get("interpolated", False):
                best[i] = f
    return [best[i] for i in sorted(best)]


# ---------------------------------------------------------------------- core


def plan_merges(
    tracks_doc: dict,
    team_colors: dict,
    field_coords: dict | None,
    manual_labels: dict | None,
) -> dict:
    """Compute the stitch plan without touching disk.

    Returns a dict with:
      groups:        list of {"canonical": id, "absorbed": [ids...]} (only
                     multi-member groups)
      id_to_group:   {old_track_id: canonical_id} for every track
      label_migrations: list of {"track_id": old, "to": canonical, "label": v}
      conflicts:     list of {"a": id, "b": id, "labels": [la, lb]} merges
                     blocked because they'd union two different manual labels
      merged_pairs:  number of applied A->B links
    """
    assignments = team_colors.get("assignments", {})

    def team_of(tid: int) -> tuple[str, float]:
        entry = assignments.get(str(tid))
        if not entry:
            return "unknown", 0.0
        return entry.get("team", "unknown"), float(entry.get("confidence", 0.0))

    frags: dict[int, _Fragment] = {}
    for tr in tracks_doc.get("tracks", []):
        tid = int(tr["track_id"])
        if not tr.get("frames"):
            continue
        team, conf = team_of(tid)
        frags[tid] = _Fragment(tr, team, conf)

    # field coords lookup: track_id -> {frame -> (x, y)}
    field_by_track: dict[int, dict[int, tuple[float, float]]] = {}
    if field_coords:
        for tr in field_coords.get("tracks", []):
            tid = int(tr["track_id"])
            field_by_track[tid] = {
                int(f["i"]): (f.get("x"), f.get("y")) for f in tr.get("frames", [])
            }

    manual = {int(k): v.get("label") for k, v in (manual_labels or {}).items()}

    # ---- enumerate candidate A->B links ----
    candidates: list[tuple[float, int, int]] = []  # (cost, a_id, b_id)
    ids = sorted(frags)
    for a_id in ids:
        a = frags[a_id]
        if a.team not in ("A", "B") or a.team_conf < MIN_TEAM_CONF:
            continue
        for b_id in ids:
            if a_id == b_id:
                continue
            b = frags[b_id]
            if b.team != a.team:
                continue  # cross-team / ref / unknown never merge
            if b.team_conf < MIN_TEAM_CONF:
                continue  # muddy color read on B: uncertain, don't merge
            gap = b.first_i - a.last_i
            if gap < 1 or gap > MAX_GAP:
                continue
            if _overlap(a, b):
                continue
            if not _field_jump_ok(a, b, field_by_track):
                continue
            err = _motion_gap(a, b)
            if err is None:
                continue
            # cost: motion error dominates, gap breaks ties
            cost = err + gap * 0.5
            candidates.append((cost, a_id, b_id))

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    # ---- greedy chaining with union-find, guarding manual-label conflicts ----
    parent: dict[int, int] = {tid: tid for tid in frags}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # each component's set of distinct manual labels currently held
    comp_labels: dict[int, set[str]] = {
        tid: ({manual[tid]} if manual.get(tid) else set()) for tid in frags
    }
    used_as_head: set[int] = set()   # a fragment already has an incoming link (a B)
    used_as_tail: set[int] = set()   # a fragment already has an outgoing link (an A)

    conflicts: list[dict] = []
    applied = 0
    for cost, a_id, b_id in candidates:
        # keep the chain linear: one successor per fragment, one predecessor
        if a_id in used_as_tail or b_id in used_as_head:
            continue
        ra, rb = find(a_id), find(b_id)
        if ra == rb:
            continue
        la, lb = comp_labels[ra], comp_labels[rb]
        merged_labels = la | lb
        if len(merged_labels) > 1:
            # would union two different human labels: cross-player fusion. Block.
            conflicts.append(
                {"a": a_id, "b": b_id, "labels": sorted(merged_labels)}
            )
            continue
        # apply
        parent[rb] = ra
        comp_labels[ra] = merged_labels
        used_as_tail.add(a_id)
        used_as_head.add(b_id)
        applied += 1

    # ---- assemble groups: component -> members, canonical = longest ----
    members: dict[int, list[int]] = {}
    for tid in frags:
        members.setdefault(find(tid), []).append(tid)

    groups: list[dict] = []
    id_to_group: dict[int, int] = {}
    label_migrations: list[dict] = []
    for root, mem in members.items():
        mem_sorted = sorted(mem)
        if len(mem_sorted) == 1:
            id_to_group[mem_sorted[0]] = mem_sorted[0]
            continue
        # canonical = fragment with most real frames; ties -> smallest id
        canonical = max(mem_sorted, key=lambda t: (frags[t].n_real, -t))
        absorbed = [t for t in mem_sorted if t != canonical]
        for t in mem_sorted:
            id_to_group[t] = canonical
        groups.append({"canonical": canonical, "absorbed": absorbed})
        # manual-label migration: an absorbed fragment's label moves to canonical
        for t in absorbed:
            lbl = manual.get(t)
            if lbl and lbl != manual.get(canonical):
                label_migrations.append(
                    {"track_id": t, "to": canonical, "label": lbl}
                )

    return {
        "groups": groups,
        "id_to_group": id_to_group,
        "label_migrations": label_migrations,
        "conflicts": conflicts,
        "merged_pairs": applied,
    }


def apply_plan(tracks_doc: dict, plan: dict) -> dict:
    """Build a new stitched tracks document from a plan (pure, no disk)."""
    tracks_by_id = {int(tr["track_id"]): tr for tr in tracks_doc.get("tracks", [])}
    id_to_group = plan["id_to_group"]

    # canonical -> its member ids
    group_members: dict[int, list[int]] = {}
    for tid, canon in id_to_group.items():
        group_members.setdefault(canon, []).append(tid)

    out_tracks: list[dict] = []
    for canon in sorted(group_members):
        mem = group_members[canon]
        if len(mem) == 1:
            out_tracks.append(
                {"track_id": canon, "frames": tracks_by_id[canon]["frames"]}
            )
        else:
            frames = _merge_frames(tracks_by_id, mem)
            out_tracks.append({"track_id": canon, "frames": frames})

    return {"play_id": tracks_doc.get("play_id"), "tracks": out_tracks}


def migrate_manual_labels(manual_labels: dict, plan: dict) -> dict:
    """Return manual_labels rekeyed onto canonical ids (lossless).

    Every label whose track was absorbed is moved to the group's canonical id.
    If the canonical already holds a label, the canonical's is kept (conflict
    merges were already blocked upstream, so a same-label collision is benign).
    """
    id_to_group = plan["id_to_group"]
    out: dict[str, dict] = {}
    # first place canonical-held labels so absorbed ones never clobber them
    order = sorted(
        manual_labels.items(),
        key=lambda kv: 0 if id_to_group.get(int(kv[0]), int(kv[0])) == int(kv[0]) else 1,
    )
    for tid_s, entry in order:
        tid = int(tid_s)
        canon = id_to_group.get(tid, tid)
        key = str(canon)
        if key in out:
            continue  # canonical already labeled: keep it (do not overwrite)
        out[key] = entry
    return out


def validate(tracks_doc_stitched: dict, manual_labels: dict | None, plan: dict) -> None:
    """Hard invariants; raise AssertionError on any violation.

    * every manual label still points at a LIVE stitched track,
    * no manual label VALUE is lost,
    * no stitched group unions two different manual labels (zero cross-player
      fusion among label carriers).
    """
    live_ids = {int(tr["track_id"]) for tr in tracks_doc_stitched.get("tracks", [])}
    if not manual_labels:
        return
    migrated = migrate_manual_labels(manual_labels, plan)

    # 1. every migrated key is a live track
    for tid_s in migrated:
        assert int(tid_s) in live_ids, (
            f"manual label points at dead track {tid_s} after stitch"
        )

    # 2. no label VALUE lost. A merge of two fragments carrying the SAME label
    #    (same player, split track) legitimately collapses two identical entries
    #    into one — that is a dedupe, not a loss — so compare the SET of distinct
    #    label values, which must be preserved exactly.
    def label_set(m: dict) -> set[str]:
        return {e.get("label") for e in m.values() if e.get("label") is not None}

    before = label_set(manual_labels)
    after = label_set(migrated)
    assert before == after, (
        f"manual label value lost: before={sorted(before)} after={sorted(after)}"
    )

    # 3. no group unions two different original labels
    id_to_group = plan["id_to_group"]
    group_labels: dict[int, set[str]] = {}
    for tid_s, entry in manual_labels.items():
        lbl = entry.get("label")
        if lbl is None:
            continue
        canon = id_to_group.get(int(tid_s), int(tid_s))
        group_labels.setdefault(canon, set()).add(lbl)
    for canon, labels in group_labels.items():
        assert len(labels) <= 1, (
            f"cross-player fusion: group {canon} unions labels {sorted(labels)}"
        )


# ---------------------------------------------------------------------- stage


def stitch(play_id: str, data_dir: Path) -> dict:
    """Merge track fragments for ``play_id``; write stitched tracks.json.

    Backs the raw tracker output up to ``tracks.prestitch.json`` (only the first
    time — an existing backup is never overwritten, so re-running is idempotent
    against the original), writes the stitched result to ``tracks.json``, and,
    for labeled plays, rewrites manual_labels.json with migrated keys after
    validating that no label was lost. Returns a report dict.
    """
    data_dir = Path(data_dir)
    pdir = data_dir / "plays" / play_id

    tracks_doc = _load_json(pdir / "tracks.json")
    tc_path = pdir / "team_colors.json"
    team_colors = _load_json(tc_path) if tc_path.is_file() else {"assignments": {}}
    fc_path = pdir / "field_coords.json"
    field_coords = _load_json(fc_path) if fc_path.is_file() else None
    ml_path = pdir / "manual_labels.json"
    manual_labels = _load_json(ml_path) if ml_path.is_file() else None

    n_before = len(tracks_doc.get("tracks", []))

    plan = plan_merges(tracks_doc, team_colors, field_coords, manual_labels)
    stitched = apply_plan(tracks_doc, plan)
    validate(stitched, manual_labels, plan)

    n_after = len(stitched.get("tracks", []))

    # back up the raw tracker output ONCE, then overwrite tracks.json.
    backup = pdir / "tracks.prestitch.json"
    if not backup.is_file():
        backup.write_text(json.dumps(tracks_doc, indent=1))

    tmp = (pdir / "tracks.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stitched, indent=1))
    tmp.replace(pdir / "tracks.json")

    migrated_labels = None
    if manual_labels is not None:
        migrated_labels = migrate_manual_labels(manual_labels, plan)
        # back up original labels once, then write migrated
        ml_backup = pdir / "manual_labels.prestitch.json"
        if not ml_backup.is_file():
            ml_backup.write_text(json.dumps(manual_labels, indent=1))
        mtmp = ml_path.with_suffix(".json.tmp")
        mtmp.write_text(json.dumps(migrated_labels, indent=1))
        mtmp.replace(ml_path)

    return {
        "play_id": play_id,
        "tracks_before": n_before,
        "tracks_after": n_after,
        "fragments_merged": n_before - n_after,
        "merged_pairs": plan["merged_pairs"],
        "groups": plan["groups"],
        "label_migrations": plan["label_migrations"],
        "conflicts": plan["conflicts"],
        "manual_labels_migrated": migrated_labels,
    }
