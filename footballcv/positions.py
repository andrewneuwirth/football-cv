"""Label stage: snap frame, O/D split, LOS, and defensive positions.

Reads field_coords.json (+ meta.json for fps) for a play and writes labels.json:

    {"play_id": ..., "snap_frame": int, "los_x": float,
     "offense_direction": "+x"|"-x",
     "teams": {"<track_id>": "O"|"D", ...},
     "positions": {"<track_id>": "lb_b", ...},        # D tracks only
     "position_confidence": {"<track_id>": 0.9, ...}}

All heuristics are best-effort: on weird alignments we still emit labels, just
with lower confidence. This module never intentionally raises on odd data.

Snap: a human-marked snap.json wins; otherwise the first sustained team-wide
mean-speed spike after a low-motion stretch.

Real-film realities this module is built around (measured on hand-labeled
plays, not idealized formations):

- Tracking also picks up sideline benches, chain crews and officials; only a
  "core" of tracks alive within a small window of the snap, on the field, and
  inside the densest x-window is used to fit the formation. Everything else
  still gets a team (side of the LOS) and, for defenders, a best-effort
  position inherited from the nearest fitted defender / role slot.
- LOS: the neutral zone is the ~0.5-2.5 yd gap between two crowded fronts
  (OL and DL). Candidate splits of the sorted x's are scored by how many
  bodies crowd each facing edge plus a neutral-zone-width prior. A balance
  term is useless on real film (far more clutter on one side than the other).
- Offense = the side with the higher *fraction* of its bodies on its front
  line (OL + on-LOS receivers vs. a 4-man DL in front of a spread-out back
  seven). x-tightness of the front is NOT reliable.
- Homography error compresses depth on real film: the free safety can project
  at 5-8 yd and press corners at 2-4, so absolute depth bands fail. Structure
  is used instead: the back seven orders across the field as the coach-
  verified chain cb_w - saf_w - lb_c - lb_b - lb_a - F - cb_s
  (weak -> strong), with F the deepest middle body when depth is informative.
- Positions are named by a GLOBAL one-to-one assignment of the 11 system
  roles to candidate defenders over a soft-cost matrix of football-geometry
  features. The DL group comes from a Hungarian solve
  (scipy.optimize.linear_sum_assignment); the back seven from an
  order-preserving DP (monotone alignment of the coach chain to the bodies
  sorted weak -> strong), which enforces every pairwise RELATIVE-ORDER
  constraint by construction — bodies can walk far from their usual spots
  (a lb_a apexed out under the F) but the order across the field holds,
  and there is never a saf_w without a cb_w outside him. Deep safeties (the
  F, a two-high saf_w) are the one legitimate exemption and are matched
  out of band when they read truly deep. Dummy skips at the same abstain
  cost mean a missing body leaves its role unfilled without shifting
  neighbors, and an extra body goes unassigned instead of stealing a role —
  errors stay local instead of cascading down the chain.
- Officials inside the formation (umpire deep middle, head linesman on the
  LOS at the sidelines) are pruned as depth outliers / sideline-LOS bodies.

Defensive vocabulary is THIS TEAM'S system (CONTRACT): dl_es T N dl_ew (DL),
lb_a lb_b lb_c (LB), cb_s cb_w F saf_w (DB); _strong = strong side = the side
with more offensive players outside the tackles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from footballcv.paths import play_dir

# ---------------------------------------------------------------- tunables

LOW_RUN = 15          # frames of low motion required before the snap
SPIKE_RUN = 3         # consecutive spike frames required to call the snap
LOW_SPEED = 0.75      # yd/s: below this counts as "still" (floor for adaptive thr)
SPIKE_SPEED = 1.5     # yd/s: above this counts as a snap-level spike (floor)
SMOOTH_WIN = 9        # centered moving-average window on positions (noise supp.)
PRE_SNAP_WIN = 10     # frames before the snap averaged for alignment positions

# --- interior-line-burst snap detector (the primary automatic snap) ---
SNAP_INTERIOR_WIN = 16.0  # yd: densest x-window that holds the interior line stack
SNAP_MIN_TRACK = 8        # a track needs this many frames to vote on the cluster
SNAP_SET_RUN = 10         # frames of low interior motion required before the snap
SNAP_SET_SLACK = 0.55     # yd/s above the interior floor still counts as "set"
SNAP_BURST_RUN = 6        # consecutive burst frames required to confirm the snap
SNAP_HI_FRAC = 0.40       # burst threshold as a fraction of the floor->peak range
SNAP_HI_MIN = 1.4         # yd/s: absolute floor for the strong-burst threshold
SNAP_FB_FRAC = 0.50       # gentler relative burst threshold (soft/noisy clips)
SNAP_LAG = 6              # frames the detected onset trails the true snap
                          #   (centered-smoothing + line acceleration ramp lag)

CORE_BEFORE = 25      # core tracks must be recorded within [snap-25, snap+1]
CORE_AFTER = 1
CORE_Y_MIN, CORE_Y_MAX = -2.5, 53.8   # plausibly-on-field band (edges stretch)
DENSE_WIN = 26.0      # densest x-window that holds the formation
FRONT_CROWD = 1.0     # yd from a front edge that counts as crowding it
FRONT_BAND = 1.5      # yd from a front that counts as "on the line"
GAP_LO, GAP_HI = 0.6, 2.3   # plausible neutral-zone width (yd)
GAP_BONUS = 3.0
BACKFIELD_LO, BACKFIELD_HI = 3.0, 8.0   # QB/RB depth behind the O front
BACKFIELD_DY = 8.0                       # ...near the formation middle
BACKFIELD_BONUS = 3.0
MIN_CORE_FRAMES = 3   # 1-2 frame ghost tracks distort the formation fit

# ---- offense-vs-defense ROLE detection (which SIDE has the ball) -----------
# The two teams are separated cleanly by jersey color; the fragile part is
# deciding which side is OFFENSE. Football rules give several corroborating
# signals — combine them into a per-side "offense-likeness" score and pick the
# higher. Every signal is a rule-based structural count, weighted by how
# reliably it survives real film (tracking dropout, a mid-play snap frame).
#
#   1. 7-ON-THE-LINE (strongest, rule-based): the offense must have ~7 bodies
#      within ~1 yд of the LOS; a defense fields 3-4 down linemen. The side
#      with clearly MORE bodies at its own front is the offense.
#   2. BACKFIELD TELL: the offense hides a COMPACT 1-3 body cluster (QB + backs)
#      3-8 yд directly behind its front's lateral center; the defense's
#      "backfield" is the secondary — spread WIDE and deep. Score each side on
#      whether it has a tight near-center cluster (offense) rather than wide
#      deep bodies (the _wide_deep_count corners tell, folded in).
#   3. OL CONTIGUITY: the offense packs ~5 bodies shoulder-to-shoulder in a
#      tight contiguous lateral row at one depth; a defensive front is sparser.
#
# The score differences are compared with a small dead-band so a near-tie
# defers to the next-strongest signal rather than flipping on noise.
LINE_BAND = 1.2       # yд from a side's front edge that reads as "on the line"
LINE_MARGIN = 1       # a line-count lead this size (bodies) is a decisive
                      # 7-on-line tell (offense ~7, defense ~3-4)
LINE_PRESENT = 5      # a side needs this many bodies on the line for the
                      # combined score to be trusted over the legacy ordering;
                      # fewer than this on BOTH sides = no recognizable front
LINE_MAX_DY = 13.0    # an on-the-line body sits within this of the formation's
                      # lateral center (the tackle box + tight ends); a body
                      # farther out is a split receiver or sideline clutter, not
                      # part of the interior front the 7-on-line count measures
W_LINE = 1.4          # offense-likeness per on-the-line body (signal 1)
W_BACKFIELD = 3.0     # offense-likeness for having a compact central backfield
W_CONTIG = 1.2        # offense-likeness per shoulder-to-shoulder OL body
W_CORNER = 1.5        # DEFENSE-likeness per wide-deep corner (signal 2 inverse)
W_FRONT_FRAC = 2.0    # offense-likeness per unit of front-crowding fraction
                      # (the legacy tie-breaker, now a low-weight corroborator)
OD_DEADBAND = 0.75    # score gap below this is a tie -> defer to the legacy
                      # wide-deep / front-fraction ordering (never regress a
                      # play the old heuristic already got right)
OD_CONF_SCALE = 6.0   # softening constant in the od_confidence margin ratio
                      # |ol-orr| / (|ol|+|orr|+SCALE): keeps a small absolute
                      # gap between two small scores from reading as "certain",
                      # and a genuine runaway (big gap) near 1.0
OD_CONF_BENCH_CAP = 0.35  # a bench-corrupted front resolved by the wide-deep
                      # fallback is inherently uncertain -> cap its auto
                      # confidence low so the dashboard surfaces it for a
                      # one-click human confirm (sample-077's shape)
BACKFIELD_MIN = 1     # a backfield cluster needs at least this many bodies
BACKFIELD_MAX = 3     # ...and at most this (QB + 1-2 backs); more = not a
                      # backfield (a spread secondary)
BACKFIELD_CENTER_DY = 6.0  # backfield bodies sit within this of the front's
                           # lateral center (a defense's deep bodies spread wider)
CONTIG_DEPTH = 1.2    # OL bodies share a common depth within this band
CONTIG_GAP = 2.6      # adjacent OL sit within this lateral gap (shoulder pads
                      # ~0.7 yд + spacing); a wider gap breaks the contiguous row
CONTIG_MIN = 3        # a contiguous run this long counts as an OL block
CONTIG_MIN_SPAN = 3.5 # ...and it must stretch at least this wide laterally (an
                      # OL spans the tackle box); a knot inside a couple of yards
                      # is a bunch / track cluster, not a line
CONTIG_STRADDLE = 4.0 # ...and the run must straddle the formation center within
                      # this margin (the OL centers on the ball; a run entirely
                      # off to one sideline is a bunch, not the offensive line)
NARROW_SPAN = 8.0     # a formation side spans far more than this laterally (OL
                      # + split receivers); below it the side is a sideline knot
W_NARROW = 8.0        # DEFENSE-likeness penalty for a narrow (knotted) side, big
                      # enough that a bunched cluster never out-scores a field-
                      # spanning opponent (sample-053's dark sideline knot)

# ---- snap-robustness for the O/D decision ----------------------------------
# A single auto-snap frame can fire a few frames late and catch the play mid-
# rep — bodies stream downfield, fronts blur, and the O/D signals read from a
# scrambled formation. So the O/D decision is measured over a short PRE-snap
# window and, within it, on the CLEANEST (most formation-like) frame: the one
# where the two fronts are tightest and fewest bodies have leaked far downfield.
# Human snap.json is honoured exactly (the window still ends at the human snap);
# direction.json still wins absolutely downstream.
OD_WINDOW = 15        # frames before the snap scanned for a clean formation
OD_WINDOW_STEP = 3    # stride through the window (every Nth frame sampled)
OD_STRAY_DEPTH = 12.0 # a body this far onto the DEFENSE side of the LOS at the
                      # sampled frame is streaming downfield mid-play — its
                      # presence marks the frame as less formation-like

DL_DEPTH = 2.0        # yd from LOS that reads as on the line
DL_BEHIND = 3.0       # allow slight across-LOS projection noise for DL
DL_MAX_DY = 9.0       # wider on-LOS bodies are refs / press corners, not DL
SIDELINE_OFFICIAL_DEPTH = 3.5  # on-LOS + at the sideline = head linesman
SIDELINE_Y = 2.5
BACK_MAX_DEPTH = 9.5  # deeper *central* bodies are officials/clutter
BACK_KEEP_MIN = 7     # ...but only pruned when a full back seven remains
WIDE_EXEMPT_DY = 10.0 # wide bodies may be deep corners; not depth-pruned...
WIDE_EXEMPT_MAX = 12.0  # ...unless implausibly deep even for a corner
UMPIRE_MARGIN = 2.5   # deep-middle outlier beyond the LB row
BOX_PRUNE_DEV = 2.0   # box body deviating this much from the row depth is
                      # an official when the box is overfull
F_DEPTH_MARGIN = 1.5  # clearly-deeper of the two extra backs is the F
SAFETY_DEPTH = 4.0    # this far beyond the LB row is true safety depth
                      # (the umpire stands ~2-3 beyond; a two-deep shell 4+)
WIDE_DY = 5.0         # |dy| that reads as a split-out receiver
WIDE_ON_LOS = 2.5     # receiver depth for the primary strength count
WIDE_ANY = 6.5        # receiver depth for the fallback strength count
WIDE_FALLBACK_DY = 8.0
NOSE_WIDTH = 2.2      # |dy| that reads as a head-up nose/shade
EDGE_DY = 3.0         # |dy| that reads as an edge outside the tackle
POSITION_RANGE = 22.0 # only defenders within this of the LOS get positions

# ---- jersey-color-first team split (team_colors.json is the primary signal)
COLOR_CONF = 0.8      # cluster vote share below this = color unknown (used for
                      # the LOS-region split and the soft formation-fit cost)
TEAM_COLOR_CONF = 0.6 # ...but the O/D TEAM tag is decided GLOBALLY per color:
                      # once the two clusters are known to separate cleanly
                      # (a trustworthy color split), a body only has to CLEARLY
                      # belong to cluster A or B to inherit that color's team.
                      # This bar sits above the ref band (~0.47-0.56) so striped
                      # officials never earn a team, but below COLOR_CONF so the
                      # 0.6-0.8 players that geometry would strand on the wrong
                      # side of a mis-read LOS follow their color instead.
REF_CONF = 0.55       # ref votes below this still exclude from the
                      # formation FIT, but keep team/fallback-position claims:
                      # real players in muddy jerseys poll ref at 0.4-0.5
COLOR_MIN_SIDE = 3    # confident bodies per cluster needed to trust colors
# ---- neutral-zone color bind (the on-the-line team signal) ------------------
# Right at the ball, side-of-LOS is the LEAST reliable team tell: an O-lineman
# and a D-lineman sit inches apart across the neutral zone, so a small snap-
# timing or LOS error flips them (sample-080: an auto-snap fired a few frames
# late let dark DEFENSIVE linemen penetrate ~1-1.5 yд onto the OFFENSE side of
# the estimated LOS, and side-of-LOS then stamped them OFFENSE — corrupting the
# defensive front). JERSEY COLOR is the reliable signal there. So a body inside
# the neutral-zone band ON THE OFFENSE SIDE of the LOS (a penetrated defender)
# binds its team to a MODERATE color vote, regardless of the los_clean gate —
# geometry cannot be trusted exactly where it is worst. Only the offense side
# is rescued: an offensive lineman essentially never lines up across the ball,
# so a body on its OWN (defense) side polling the offense color at moderate
# confidence is a muddy-jersey defender and geometry stays right (sample-005:
# a real dl_es polled the offense color 0.72 but sat on its own side -> stays D).
# Far-from-LOS bodies keep the existing los_clean behavior (geometry can
# corroborate in the secondary / backfield).
NEAR_LOS_BAND = 2.75  # depth onto the OFFENSE side of the LOS (yд) that still
                      # reads as neutral-zone / on the line — where color, not
                      # side-of-LOS, decides a penetrated body's team
NEAR_LOS_COLOR_CONF = 0.6  # relaxed per-body color bar for the near-LOS bind:
                      # a moderate cluster vote (sample-080's penetrated
                      # D-linemen poll 0.71-0.78 for the dark cluster) overrides
                      # geometry on the line. Sits well below the strict
                      # COLOR_CONF (0.8) the far-field split uses; a body below
                      # it is genuinely color-unknown (ref-band ~0.5) and still
                      # falls back to side-of-LOS. Matches _cluster_color's bar.
CONFLICT_DEPTH = 5.0  # pre-snap track this deep on its color's wrong side of
                      # the LOS = sideline staff in team colors -> no claims
TEAM_Y_MIN, TEAM_Y_MAX = -2.5, 54.3  # on-field band for team/position claims
TEAM_X_SLACK = 10.0   # ...but near-LOS bodies keep claims despite y noise
DL_ROW_TOL = 1.2      # DL row = shallow cluster within this of the shallowest
BACK_MIN_DEPTH = -1.0 # back-seven bodies are never across the LOS
WIDE_IDLE_DY = 12.0   # on-LOS body this wide with no offense to cover
WIDE_IDLE_DEPTH = 1.5 # ...is a sideline sub, not a press corner
WIDE_COVER_DY = 8.0   # a press corner has offense within this in y
F_CLEAR_MARGIN = 2.5  # interior back this much deeper than its row = the F
F_SIDE_DY_MIN = 3.0   # F declares strength only when clearly off-center...
F_SIDE_DY_MAX = 10.0  # ...but still a safety, not a deep corner

# ---- global one-to-one role assignment (Hungarian) --------------------------
# The 11 roles are matched to candidate defenders by scipy's
# linear_sum_assignment on a soft-cost matrix. Every cost is a generic
# football-geometry feature in (depth-from-LOS, signed-lateral) space —
# absolute geometry, never ranks, so one misread body cannot cascade down
# the chain: its role goes unfilled (dummy row/col) and neighbors keep their
# own best fits.
ROLES = ["dl_ew", "N", "T", "dl_es", "lb_c", "lb_b", "lb_a", "saf_w", "F",
         "cb_w", "cb_s"]
# The back seven in coach-verified across-field order, weak -> strong. The
# order-preserving DP matches this chain to the bodies sorted by signed
# lateral, so e.g. an assigned saf_w is ALWAYS weak of an assigned lb_c.
BACK_CHAIN = ["cb_w", "saf_w", "lb_c", "lb_b", "lb_a", "F", "cb_s"]
ORDER_PEN = 6.0        # filling an in-chain overhang (saf_w/F) while its
                       # outside corner role is empty: near-impossible per
                       # the coach — the outermost body on each side is a
                       # corner. When cb_w is empty every body weak of the
                       # saf_w is necessarily unassigned (or the saf_w body
                       # is itself the outermost); both read as "the
                       # overhang has no corner outside him" and pay this.
GAP_PEN = 5.0          # leaving a back-seven chain role UNFILLED with filled
                       # roles on BOTH sides (a bracketed interior hole) —
                       # football nonsense: the outermost weak second-level
                       # body is the saf_w, so when short a body you push the
                       # fill outward and empty an END role, never a middle
                       # one (coach read, sample-089: saf_w was left in a hole
                       # between a filled cb_w and a filled lb_c). Charged in
                       # _order_align per bracketed skip, ON TOP of the flat
                       # ASSIGN_ABSTAIN an end-gap already pays, so a shift
                       # that fills the interior wins over bracketing it. Sized
                       # above the ~4.4 emit-cost gap on 089 between the
                       # bracketed read and the shifted all-filled read, but
                       # below 2x ASSIGN_ABSTAIN + a real geometry mismatch so
                       # a body that genuinely does not belong in the chain
                       # still abstains rather than being crammed in.
FIELD_Y_LO, FIELD_Y_HI = 0.0, 53.33  # bodies projecting out of bounds never
                                     # anchor a corner's lateral target
ASSIGN_ABSTAIN = 2.5   # cost of leaving a role unfilled / a body unassigned
                       # (a body+role pair worse than 2x this stays apart —
                       # low enough that clutter cannot be crammed into an
                       # otherwise-empty role)
DL_LAT = {"dl_ew": -4.5, "N": -0.9, "T": 1.6, "dl_es": 4.5}   # yd off DL center
LB_SPACING = 4.0       # lb_c / lb_a offset off the back-seven center
BC_CLIP = 1.5          # defensive refinement of the O-anchored box center
W_DL_LAT = 0.55        # per yd from the DL lateral slot
W_DL_DEPTH = 1.0       # per yd outside the DL row depth band
# ---- LOS-anchored depth gate (the D-line-rigidity prior) --------------------
# A 4-down front is a near-straight row AT the LOS. side-of-LOS / row-centroid
# depth alone can be fooled: when the ONLY shallow bodies tracked are walked-up
# linebackers, the DL "row" anchors on THEM (dl_d rides up to ~1.3 yд) and the
# row-band depth term stops discriminating — an off-the-line LB then wins a
# T/N/E slot (coach failure mode #1: a 2-point-stance body at 3-5 yд labeled
# DL). The fix is a SECOND depth term measured against the LOS ITSELF, immune
# to a contaminated row centroid: a body's DL cost ramps with its ABSOLUTE
# depth off the LOS beyond DL_LOS_OK. Kept SOFT (a lineman genuinely at the
# line, and even a walked-up lineman a shade off it, still reads DL) but strong
# enough that a back-seven body at 3-5 yд effectively cannot take a DL role —
# its LB/deep cost undercuts the inflated DL cost. Real film: manually-labeled
# DL sit at depth <=~1.7 yд; back-seven bodies at >=~2.7 yд, so the gate opens
# free through the true-DL band and bites hard in the linebacker band.
DL_LOS_OK = 1.3        # depth off the LOS (yд) a DL may reach for free; a body
                       # shallower than this pays no LOS-depth gate at all
                       # (covers the true-DL band: manual DL sit at <=~1.7 yд,
                       # so a lineman at 1.7 pays only a soft 1.4)
W_DL_LOS_DEPTH = 3.5   # per yд beyond DL_LOS_OK: soft near the line, steep by
                       # linebacker depth (a body 3.5 yд off pays ~7.7, far
                       # above ASSIGN_ABSTAIN, on top of the row-band term —
                       # its LB / deep roles decisively win the duel). Tuned so
                       # an all-linebacker shallow front (sample-009: bodies at
                       # 1-1.7 yд) yields DL roles to the back seven where it
                       # should, without pricing a genuine walked-up lineman
                       # (uniform gate across a real front keeps its ranking).
DL_WIDE_PEN = 4.0      # on-LOS body wider than DL_MAX_DY is not a lineman
DL_HARD_SLACK = 0.7    # beyond the row band by more than this...
DL_HARD_PEN = 6.0      # ...is never a lineman (backs can't cram into the DL)
EDGE_PAIR_SPREAD = 6.5 # a 2-body row straddling the center this wide is the
EDGE_PAIR_PEN = 1.5    # two edges (interior occluded) — N/T pay this
W_BACK_ONLINE = 1.2    # per yd a back role reaches into the DL row
W_LB_LAT = 0.5
W_LB_DTGT = 0.15       # per yd off the LB-row depth (breaks lateral duels)
W_LB_DEEP = 0.25       # per yd beyond LB_DEEP_OK (umpire resist)
LB_DEEP_OK = 7.0
W_C_LAT = 0.35         # corners pull hard to their side's wide anchor
W_C_DEEP = 0.25
C_DEEP_OK = 12.0
W_WHIP_LAT = 0.5       # weak overhang chain slot (hypothesis one)...
W_WS_DEPTH = 0.3       # ...or the shallower of two deep safeties
W_WS_LAT = 0.05        # (hypothesis two, the old depth carve-out)
W_WHIP_DEPTH = 0.2     # mild pull off the LB row
W_F_DEPTH = 1.0        # F is the deepest back when depth is informative...
W_F_LAT = 0.08         # ...and the strong chain slot when depth compresses
WIDE_ANCHOR_MIN = 8.0  # a side's wide anchor must be at least this far out
CORNER_FALLBACK = 12.0 # corner target when a side has no wide body at all
MARGIN_SCALE = 2.0     # cost-margin -> confidence squash scale
COLOR_LEAN_W = 14.0    # assignment cost per unit of jersey vote share leaning
                       # the offense's color beyond 0.5. SOFT, not the hard
                       # COLOR_CONF bar: a 0.62-wrong body pays 1.7 (loses any
                       # duel against a same-geometry right-color body, and
                       # abstains unless nobody else fits); a 0.95-wrong body
                       # pays 6.3 > 2x ASSIGN_ABSTAIN = effectively forbidden
                       # (goal-line play 093: a 0.62-white body won the F).
SNAP_SIM_WIN = 15      # frames around the snap that count as "simultaneous":
                       # two at-snap bodies never share a role; the same
                       # player re-tracked later may inherit one (intended)
BLOCK_NEAR = 4.0       # a fragment may not inherit a fitted defender's role
                       # when that defender is alive at the same time AND
                       # within this many yards (a different body stacked
                       # beside him); a far holder (play 079's 12-yd cb_s)
                       # does not block inheritance
FALLBACK_DEPTH_SLACK = 3.0  # inherited names never reach deeper than the
                            # deepest core-fitted defender + this

# ---- participant filter (unified "is this body in THIS play?") -------------
# One gate, computed once over field yards at the snap, applied BEFORE the O/D
# split so nothing downstream (LOS, strength, the Hungarian DL solve, the
# order-preserving back-seven DP, fallback inheritance) ever sees a non-player.
# A body is a PARTICIPANT unless rejected. Three signals; the combination rule
# (below) is deliberately conservative so real players are never over-rejected.
PART_WIN_S = 2.0        # motion window [snap, snap + this] in seconds
PART_Y_MARGIN = 2.0     # yd past a sideline (0 / 53.33) that still counts as
                        # plausibly in bounds (homography stretches the edges)
PART_STRANDED_DEPTH = 22.0  # a body this far onto the DEFENSIVE side of the LOS
                            # is deeper than any plausible safety = not in the
                            # play (calibration-generous; a CONTRIBUTOR gate,
                            # so the margin is wide — a deep safety pedaling to
                            # ~18 yд on depth-compressed film stays). Only the
                            # DEFENSIVE side is gated on depth: real offensive
                            # backs sit yards BEHIND the LOS legitimately.
PART_FAR = 14.0         # yd from the participant-cluster centroid that reads as
                        # "outside the ~22-man blob" (candidate for rejection)
PART_MOVE_MIN = 1.2     # path length (yd) below which a body is stationary
                        # environment over the whole play window. A press
                        # corner in tight coverage barely travels (~1.8 yд in
                        # the first 2s), so the bar sits below that; a still
                        # coach (~0.4 yд) is well under it.
PART_MIN_MOTION_FRAMES = 8  # a window shorter than this can't give a reliable
                            # motion estimate -> fall back to signals 1+2 only
PART_DENSITY_R = 12.0   # neighbourhood radius (yд) for the k-NN density used
                        # to find the participant-cluster centroid robustly

# ---- sideline-BENCH detection (the home/visitor bench in team colors) -------
# A running/kneeling bench of ~11-25 substitutes stands shoulder-to-shoulder in
# ONE team's jersey along a sideline the whole play. The tracker picks them up
# and team_colors clusters them with that team, so the team's confident-color
# count balloons past 11 (sample-077: 34 "dark" bodies for an 11-man side). For
# the O/D-ROLE decision that inflated, diffuse team then reads as the DEFENSE
# (its real 7-on-line front is drowned out) and the compact opponent is mistaken
# for the offense. The bench has a sharp signature distinct from any real player:
#   * a DENSE ROW — more than BENCH_MIN same-color bodies packed into
#   * a FIELD-EDGE y BAND — hugging / just past a sideline (extreme y), and
#   * OFF THE FORMATION — its own tight y-cluster, mostly LOW-MOTION over the
#     play (subs don't run the route the 22 do).
# Detected bench bodies are stripped BEFORE the core / color-split / O/D scorer
# and never earn a team or position — the same total exclusion refs already get.
# The gate is strict (a big same-color pack right on an edge line) so a legit
# bunch formation or a cluster of real wide receivers is never mistaken for it.
BENCH_MIN = 6           # a same-color edge row longer than this is a bench, not
                        # a formation cluster (an offense fields at most ~5 WR/TE
                        # to a side, and never all in a 1-2 yд y-band at an edge)
BENCH_Y_LO = -14.0      # low-y (near sideline / off field) edge band the bench
BENCH_Y_HI = 1.5        # packs into (homography stretches past the 0 sideline)
BENCH_Y_LO2 = 51.8      # ...or the far-sideline band (the 53.33 boundary)
BENCH_Y_HI2 = 68.0
BENCH_STILL_FRAC = 0.5  # at least this fraction of the row must read low-motion
                        # (stationary environment) for the row to be a bench —
                        # a moving pack of real players is not stripped
BENCH_PACK_R = 3.0      # a genuine bench body is shoulder-to-shoulder: it has
                        # at least BENCH_PACK_MIN same-color neighbours inside
                        # this radius. A lone receiver split wide to the sideline
                        # (near the edge band but well clear of the pack, and
                        # running the play) has none and is never stripped
                        # (sample-013: a real WR at the sideline, 6+ yд off the
                        # bench cluster, moving 11 yд — kept as a player).
BENCH_PACK_MIN = 3      # ...this many packed neighbours confirm membership
BENCH_DOMINANCE = 3.0   # a real bench is ONE team's subs: its color must
                        # outnumber the OTHER color in the edge band by at least
                        # this ratio. A mixed end-zone / far-sideline scrum of
                        # BOTH teams (sample-048: 15 dark + 15 light behind the
                        # far boundary) is not a bench — neither color dominates,
                        # so it is left alone (stripping it perturbed a real
                        # defender's role in the formation the scrum sat behind).

# ---- PIXEL-SPACE participant pre-filter (runs FIRST, before O/D split) ------
# Field yards come from a planar homography calibrated on the yard lines; it
# EXTRAPOLATES badly for bodies far off those lines. A sideline coach 20+ yд
# behind the play projects to shallow, in-formation-looking field coords AND
# drags the formation centroid off — so the field-space participant gate above
# cannot see him. Pixel feet-positions (from tracks.json boxes) carry no
# homography error, so a pixel cluster of the ~22 on-field bodies separates the
# sideline fringe cleanly. A body is rejected here iff it is, ALL AT ONCE:
#   FAR   — its pixel feet sit outside the dense play blob (robust centroid),
#   SMALL — its box is markedly shorter than the play cluster's median box
#           (a perspective/distance proxy: bodies behind the field are smaller),
#   HIGH  — it sits well ABOVE the cluster in the frame (smaller feet-y = higher
#           = further from camera = behind the play; the camera shoots down the
#           field so off-field staff pile up at the top edge).
# The three-way AND is deliberately strict so real deep safeties / wide corners
# (HIGH+SMALL-ish but contiguous with the blob, i.e. NOT far) are never dropped.
# Rejected bodies are removed BEFORE the O/D split, LOS, strength and the
# formation centroid, and never earn a defensive position.
PIX_DENSITY_R_FRAC = 0.12   # k-NN density radius as a fraction of the frame
                            # diagonal (finds the dense play blob robustly)
PIX_FAR_FRAC = 0.14         # pixel distance to the blob centroid, as a fraction
                            # of the frame diagonal, that reads as "outside it"
PIX_SMALL_FRAC = 0.78       # box height below this fraction of the cluster
                            # median box height reads as a distance outlier
PIX_HIGH_FRAC = 0.26        # feet-y this fraction of the frame height ABOVE the
                            # cluster centroid reads as "behind the field"
PIX_PRE_SNAP_WIN = 10       # frames before the snap averaged for pixel feet
PIX_MIN_BODIES = 8          # below this the pixel cluster is too thin to trust;
                            # skip the pixel filter (field-space gate still runs)
PIX_SHALLOW_DEPTH = 10.0    # the pixel filter only OVERRIDES field space for
                            # bodies whose field projection lands SHALLOW (this
                            # far or less onto the defensive side of the LOS) —
                            # exactly the danger case the mission targets: a
                            # sideline body whose homography projection collapses
                            # to shallow, in-formation-looking yards that field
                            # space cannot reject. A pixel outlier that ALSO
                            # projects deep is already handled by the field-space
                            # depth pruning (BACK_MAX_DEPTH etc.), so honouring
                            # the pixel reject there only risks perturbing a good
                            # fit (play 006: deep clutter at ~15-18 yд). Both
                            # sides of the LOS shallow-and-inward qualify.


# ---------------------------------------------------------------- helpers


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _track_positions(field: dict) -> dict[int, dict[int, tuple[float, float]]]:
    """{track_id: {frame_i: (x, y)}} from field_coords.json (defensive parsing)."""
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


def _smooth_positions(
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


def _mean_speeds(
    positions: dict[int, dict[int, tuple[float, float]]], fps: float
) -> list[float]:
    """Team-wide mean field-space speed (yd/s) per frame index."""
    max_i = max((max(fr) for fr in positions.values()), default=-1)
    if max_i < 0:
        return []
    sums = [0.0] * (max_i + 1)
    counts = [0] * (max_i + 1)
    for frames in positions.values():
        idxs = sorted(frames)
        for prev, cur in zip(idxs, idxs[1:]):
            (x0, y0), (x1, y1) = frames[prev], frames[cur]
            dt = (cur - prev) / fps
            if dt <= 0:
                continue
            v = math.hypot(x1 - x0, y1 - y0) / dt
            sums[cur] += v
            counts[cur] += 1
    return [s / c if c else 0.0 for s, c in zip(sums, counts)]


def _find_snap(speeds: list[float]) -> tuple[int, bool, float]:
    """(snap_frame, confident, low_thr) from a team-wide mean-speed series.

    Legacy team-wide detector: the first sustained mean-speed spike after a
    low-motion stretch, else the largest single jump. Kept as the fallback
    when the interior-cluster signal in :func:`auto_snap` is unavailable, and
    still used by :func:`run` to compute the still-period ``low_thr`` that
    anchors the alignment window. The interior-line-burst detector below is
    the primary automatic snap source.
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


def _track_speeds(
    positions: dict[int, dict[int, tuple[float, float]]], fps: float
) -> dict[int, dict[int, float]]:
    """{track_id: {frame_i: field-space speed (yd/s)}} from smoothed positions.

    Speed at frame ``cur`` is the displacement from the previous recorded
    frame divided by elapsed time (mirrors :func:`_mean_speeds`, but keeps the
    per-track breakdown so the interior cluster can be measured on its own).
    """
    out: dict[int, dict[int, float]] = {}
    for tid, frames in positions.items():
        idxs = sorted(frames)
        sp: dict[int, float] = {}
        for prev, cur in zip(idxs, idxs[1:]):
            (x0, y0), (x1, y1) = frames[prev], frames[cur]
            dt = (cur - prev) / fps
            if dt <= 0:
                continue
            sp[cur] = math.hypot(x1 - x0, y1 - y0) / dt
        out[tid] = sp
    return out


def _interior_line_speed(
    positions: dict[int, dict[int, tuple[float, float]]], fps: float
) -> tuple[list[float], list[bool]]:
    """(mean_speed, present) per frame for the DENSE INTERIOR CLUSTER at the LOS.

    All-22 mean speed is noisy (pre-snap motion men, DBs bailing early,
    tracking jitter). The stack of interior linemen at the ball is DEAD STILL
    in stance pre-snap, then explodes together at the snap — a far cleaner
    still->burst signal. We locate that cluster snap-independently: take each
    track's MEDIAN x over the whole clip (robust to the post-snap play), then
    the densest ``SNAP_INTERIOR_WIN``-yard x-window (the offense+defense
    interior stack). ``mean_speed`` is the mean speed of only those bodies per
    frame; ``present[i]`` is False when NO interior body had a speed at frame i
    (a genuine data gap, distinct from a genuinely still 0.0 yd/s).
    """
    max_i = max((max(fr) for fr in positions.values()), default=-1)
    if max_i < 0:
        return [], []
    med_x: dict[int, float] = {}
    for tid, frames in positions.items():
        if len(frames) < SNAP_MIN_TRACK:
            continue
        xs = sorted(p[0] for p in frames.values())
        med_x[tid] = xs[len(xs) // 2]
    if not med_x:
        med_x = {
            tid: sum(p[0] for p in fr.values()) / len(fr)
            for tid, fr in positions.items()
        }
    xs_sorted = sorted(med_x.values())
    best_lo, best_cnt = xs_sorted[0], 0
    j = 0
    for lo in xs_sorted:
        while j < len(xs_sorted) and xs_sorted[j] <= lo + SNAP_INTERIOR_WIN:
            j += 1
        # bisect-style count within [lo, lo+WIN]
        cnt = sum(1 for x in xs_sorted if lo <= x <= lo + SNAP_INTERIOR_WIN)
        if cnt > best_cnt:
            best_cnt, best_lo = cnt, lo
    interior = [
        tid for tid, x in med_x.items()
        if best_lo <= x <= best_lo + SNAP_INTERIOR_WIN
    ]
    if len(interior) < 4:
        interior = list(med_x)
    speeds = _track_speeds(positions, fps)
    out = [0.0] * (max_i + 1)
    present = [False] * (max_i + 1)
    for i in range(max_i + 1):
        vs = [speeds[t][i] for t in interior if i in speeds[t]]
        if vs:
            out[i] = sum(vs) / len(vs)
            present[i] = True
    return out, present


def _burst_snap(
    ms: list[float], present: list[bool] | None = None
) -> tuple[int, bool]:
    """(snap_frame, confident) from an interior-cluster mean-speed series.

    The snap is a STEP CHANGE: a sustained low-motion SET period, then a
    synchronized coordinated burst. We require the set run first (this rejects
    mid-play false spikes), find the first sustained burst above an adaptive
    threshold, and return the ONSET (the frame the still period ended), backed
    off by ``SNAP_LAG`` frames to undo the centered-smoothing / acceleration
    lag (the human marks the snap a few frames before smoothed coords show the
    line clearly moving). Two burst thresholds are tried: a strong absolute
    burst first (clean real film), then a gentler relative burst (softer or
    noisier clips); the last resort is the largest sustained window after a set.

    ``present[i]`` (if given) marks frames with interior data; the adaptive
    floor/peak are computed over present frames only, so a genuinely still 0.0
    counts toward the floor while true data gaps do not.
    """
    n = len(ms)
    if n == 0:
        return 0, False
    # Adaptive floor/peak from frames with real data. Frame 0 is a structural
    # 0.0 (no previous frame) and is skipped; a genuinely still 0.0 elsewhere
    # is kept, so the floor tracks the pre-snap set even when the stance is
    # dead still (idealized/clean clips), not just noisy real film.
    if present is not None:
        body = sorted(ms[i] for i in range(1, n) if present[i])
    else:
        body = sorted(ms[1:]) if n >= 2 else sorted(ms)
    if not body:
        return n // 3, False
    floor = body[int(0.2 * (len(body) - 1))]
    peak = body[int(0.90 * (len(body) - 1))]
    rng = max(0.0, peak - floor)
    setcap = floor + SNAP_SET_SLACK
    strong = max(floor + SNAP_HI_FRAC * rng, floor + SNAP_HI_MIN)
    gentle = floor + SNAP_FB_FRAC * rng

    def onset(i: int) -> int:
        k = i
        while k > 0 and ms[k - 1] > setcap:
            k -= 1
        return max(0, k - SNAP_LAG)

    def scan(hi: float) -> int | None:
        armed = False
        set_run = 0
        for i in range(n):
            if not armed:
                set_run = set_run + 1 if ms[i] <= setcap else 0
                if set_run >= SNAP_SET_RUN:
                    armed = True
                continue
            hemi = range(i, min(i + SNAP_BURST_RUN, n))
            if i + SNAP_BURST_RUN <= n and all(ms[j] >= hi for j in hemi):
                return onset(i)
        return None

    hit = scan(strong)
    if hit is not None:
        return hit, True
    hit = scan(gentle)
    if hit is not None:
        return hit, False
    # last resort: largest sustained SNAP_BURST_RUN-frame window (no confidence)
    best_v, best_i = -1.0, None
    for i in range(1, max(1, n - SNAP_BURST_RUN)):
        v = sum(ms[i:i + SNAP_BURST_RUN]) / SNAP_BURST_RUN
        if v > best_v:
            best_v, best_i = v, i
    return (onset(best_i) if best_i is not None else n // 3), False


def auto_snap(
    positions: dict[int, dict[int, tuple[float, float]]], fps: float
) -> tuple[int, bool, float]:
    """(snap_frame, confident, low_thr): automatic snap when no human snap.

    Primary signal is the interior-line still->burst (:func:`_interior_line_speed`
    + :func:`_burst_snap`). ``low_thr`` is the still-period speed threshold on
    the TEAM-WIDE mean series (:func:`_find_snap`), which :func:`run` uses to
    anchor the pre-snap alignment window; that team-wide detector is also the
    fallback if the interior series is empty/degenerate.

    Accepts RAW positions and smooths once internally (noise suppression).
    """
    smoothed = _smooth_positions(positions)
    speeds = _mean_speeds(smoothed, fps)
    _, _, low_thr = _find_snap(speeds)
    ms, present = _interior_line_speed(smoothed, fps)
    if not ms:
        snap, confident, low_thr = _find_snap(speeds)
        return snap, confident, low_thr
    snap, confident = _burst_snap(ms, present)
    return snap, confident, low_thr


def _alignment_positions(
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


def _clean_od_frame(
    positions: dict[int, dict[int, tuple[float, float]]],
    core: set[int],
    los_x: float,
    dir_sign: float,
    snap: int,
) -> dict[int, tuple[float, float]] | None:
    """Per-core-body position at the CLEANEST formation frame near the snap.

    Robustness to a late auto-snap: a snap that fires a few frames into the rep
    catches a scrambled formation (bodies streaming downfield), and the O/D
    signals then read from junk. Scan the PRE-snap window and score each sampled
    frame by how FORMATION-LIKE it is — the tightest two fronts with the fewest
    bodies leaked far onto the defensive side — and return the core bodies'
    positions at that frame. The window ENDS at the snap, so a human snap.json is
    honoured exactly (we only ever look at or before it). Returns None when the
    window has no usable frame (caller falls back to the snap-time alignment).

    "Cleanest" = fewest core bodies past OD_STRAY_DEPTH onto the defensive side
    (mid-play leakage) and, as a tie-break, the smallest total front spread in x
    (a set formation stacks two tight fronts; a live play smears them out)."""
    if not core:
        return None
    lo = max(0, snap - OD_WINDOW)

    def strays(frame: dict[int, tuple[float, float]]) -> int:
        # Bodies past the LOS on the DEFENSE side beyond OD_STRAY_DEPTH are
        # streaming downfield — the mid-play signature. Measured against the
        # SETTLED (color-informed) LOS, not a re-derived one (a lone downfield
        # straggler would drag a widest-gap split far past the real formation
        # and make every frame look equally dirty, sample-053).
        return sum(1 for x, _ in frame.values()
                   if (x - los_x) * dir_sign > OD_STRAY_DEPTH)

    snap_frame = {t: positions[t][snap] for t in core
                  if t in positions and snap in positions[t]}
    # The snap frame is CLEAN unless it shows downfield leakage — only then do
    # we hunt an earlier, tighter frame. This keeps well-behaved plays exactly
    # on their snap alignment (no perturbation) and reserves the earlier-frame
    # search for the mid-play case the robustness targets.
    if len(snap_frame) >= 4 and strays(snap_frame) == 0:
        return None
    best: tuple[int, int, dict[int, tuple[float, float]]] | None = None
    for f in range(snap, lo - 1, -OD_WINDOW_STEP):
        frame = {t: positions[t][f] for t in core
                 if t in positions and f in positions[t]}
        if len(frame) < 4:
            continue
        s = strays(frame)
        # Prefer the fewest strays; among equals, the frame CLOSEST to the snap
        # (largest f) — the latest still-clean set point.
        key = (s, -f)
        if best is None or key < (best[0], -best[1]):
            best = (s, f, frame)
    if best is None:
        return None
    # Only override the snap alignment when the chosen frame is meaningfully
    # cleaner than the snap frame itself.
    if len(snap_frame) >= 4 and best[0] >= strays(snap_frame):
        return None
    return best[2]


# ---------------------------------------------------------------- pixel filter


def _pixel_feet(
    tracks: dict[int, list[dict[str, Any]]], snap: int
) -> dict[int, tuple[float, float, float]]:
    """{track_id: (feet_x, feet_y, box_height)} in PIXELS at the snap.

    A body's pixel "feet" = ((x1+x2)/2, y2) of its detection box; box height
    = y2-y1 (a perspective/distance proxy — bodies further from the camera are
    smaller and higher in the frame). Averaged over the pre-snap window when
    available, else the nearest recorded frame (mirrors _alignment_positions).
    Boxes carry NO homography error, so these are immune to the yard-line
    extrapolation that corrupts field coords for off-field bodies."""
    out: dict[int, tuple[float, float, float]] = {}
    for tid, frames in tracks.items():
        win = [
            fr for fr in frames
            if snap - PIX_PRE_SNAP_WIN <= int(fr.get("i", -1)) < snap
            and isinstance(fr.get("box"), (list, tuple)) and len(fr["box"]) == 4
        ]
        if not win:
            cand = [
                fr for fr in frames
                if isinstance(fr.get("box"), (list, tuple)) and len(fr["box"]) == 4
            ]
            if not cand:
                continue
            win = [min(cand, key=lambda fr: abs(int(fr.get("i", 0)) - snap))]
        n = len(win)
        fx = sum((fr["box"][0] + fr["box"][2]) / 2.0 for fr in win) / n
        fy = sum(fr["box"][3] for fr in win) / n
        h = sum(fr["box"][3] - fr["box"][1] for fr in win) / n
        out[tid] = (float(fx), float(fy), float(h))
    return out


def _pixel_reject(
    feet: dict[int, tuple[float, float, float]],
    width: float,
    height: float,
) -> set[int]:
    """Track ids that are pixel-space outliers of THIS play (sideline staff).

    Runs FIRST, before any field-space analysis. Cluster the tracked bodies by
    pixel feet-position to find the dense ~22-body play blob (a robust k-NN
    density centroid — a plain mean is dragged toward the sideline fringe), then
    reject a body only when it is FAR from the blob AND has a much SMALLER box
    than the cluster median AND sits well HIGH above the blob in the frame. The
    conjunction is what makes it safe: a deep safety or wide corner is high and
    smallish but stays CONTIGUOUS with the blob (not far); a blocked lineman is
    near the blob; only a body behind the field is far + small + high at once.

    Too few bodies (< PIX_MIN_BODIES) means the cluster is unreliable -> reject
    nothing and let the field-space participant gate do its job."""
    ids = list(feet)
    if len(ids) < PIX_MIN_BODIES or width <= 0 or height <= 0:
        return set()
    diag = math.hypot(width, height)
    pts = {t: (feet[t][0], feet[t][1]) for t in ids}
    r2 = (PIX_DENSITY_R_FRAC * diag) ** 2
    dens = {
        t: sum(
            1 for q in pts.values()
            if (pts[t][0] - q[0]) ** 2 + (pts[t][1] - q[1]) ** 2 <= r2
        )
        for t in ids
    }
    order = sorted(ids, key=lambda t: dens[t], reverse=True)
    keep = order[: max(1, len(ids) // 2)]   # densest half = the play blob
    cx = sum(pts[t][0] for t in keep) / len(keep)
    cy = sum(pts[t][1] for t in keep) / len(keep)
    cluster_h = sorted(feet[t][2] for t in keep)
    med_h = cluster_h[len(cluster_h) // 2]
    far_thr = PIX_FAR_FRAC * diag
    small_thr = PIX_SMALL_FRAC * med_h
    high_thr = PIX_HIGH_FRAC * height
    rej: set[int] = set()
    for t in ids:
        px, py = pts[t]
        far = math.hypot(px - cx, py - cy) > far_thr
        small = feet[t][2] < small_thr
        high = (cy - py) > high_thr   # smaller feet-y = higher in frame
        if far and small and high:
            rej.add(t)
    return rej


# ---------------------------------------------------------------- participant


def _cluster_centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Robust centroid of the participant blob via local point density.

    The ~22 players form ONE dense cluster around the ball; sideline coaches,
    subs and spectators are sparse and can themselves form loose clumps far
    from the field, so a plain mean/median (or even a trimmed mean) drifts
    toward them when the clutter is heavy. Instead score each body by how
    many others sit within PART_DENSITY_R yд (a k-nearest-neighbour density),
    then take the mean position of the densest half — the field blob, by far
    the densest region, wins and the sparse fringe is ignored."""
    n = len(pts)
    if n == 0:
        return 0.0, 0.0
    if n <= 3:
        return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    r2 = PART_DENSITY_R * PART_DENSITY_R
    dens = [
        sum(1 for q in pts if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= r2)
        for p in pts
    ]
    order = sorted(range(n), key=lambda i: dens[i], reverse=True)
    keep = order[: max(1, n // 2)]
    return (sum(pts[i][0] for i in keep) / len(keep),
            sum(pts[i][1] for i in keep) / len(keep))


def _play_window_path(
    frames: dict[int, tuple[float, float]], snap: int, win: int
) -> tuple[float, int]:
    """(path_length_yd, n_frames) over [snap, snap+win] for one track.

    Path length (sum of per-step displacement) is the participation signal:
    every real player rushes / drops / pursues and racks up yards; a still
    coach barely moves. n_frames < PART_MIN_MOTION_FRAMES means the estimate
    is unreliable and motion is treated as unknown by the caller."""
    seg = [frames[i] for i in range(snap, snap + win + 1) if i in frames]
    if len(seg) < 2:
        return 0.0, len(seg)
    path = sum(
        math.hypot(seg[k + 1][0] - seg[k][0], seg[k + 1][1] - seg[k][1])
        for k in range(len(seg) - 1)
    )
    return path, len(seg)


def _participants(
    align: dict[int, tuple[float, float]],
    positions: dict[int, dict[int, tuple[float, float]]],
    los_x: float,
    dir_sign: float,
    snap: int,
    fps: float,
) -> set[int]:
    """Track ids that participate in THIS play (the pre-O/D-split filter).

    A body is a PARTICIPANT unless rejected. Three signals over field yards:

    1. IN-BOUNDS (geometry, calibration-dependent -> generous margins):
       clearly past a sideline, or stranded far behind the play on either
       side of the LOS. Failing this badly drops the body outright.
    2. NEAR THE FORMATION (spatial): distance to the robust cluster centroid
       (trimmed mean, so outliers don't move it). Far = candidate to reject.
    3. PARTICIPATION (motion, the decisive signal): path length over
       [snap, snap+~2s]. Under PART_MOVE_MIN yд across the whole window is
       stationary environment. A track too short for a reliable estimate
       falls back to signals 1+2 only.

    COMBINATION (do NOT over-reject real players): a body failing IN-BOUNDS
    badly is dropped. Otherwise drop ONLY IF far-from-formation AND
    non-participating — so a blocked (still) D-lineman inside the box stays
    (near formation), a real deep safety stays (participates / near
    formation), and a still coach at midfield drops (far + still)."""
    if not align:
        return set()
    cx, cy = _cluster_centroid(list(align.values()))
    win = max(1, int(round(PART_WIN_S * fps)))
    keep: set[int] = set()
    for tid, (x, y) in align.items():
        # 1. in-bounds / not stranded (generous, calibration-dependent).
        depth = (x - los_x) * dir_sign  # + = defensive side, - = offensive
        out_of_bounds = (
            y < FIELD_Y_LO - PART_Y_MARGIN or y > FIELD_Y_HI + PART_Y_MARGIN
            or depth > PART_STRANDED_DEPTH
        )
        if out_of_bounds:
            continue
        # 2. distance to the participant blob.
        far = math.hypot(x - cx, y - cy) > PART_FAR
        if not far:
            keep.add(tid)
            continue
        # 3. motion over the play window (decisive for far bodies).
        frames = positions.get(tid, {})
        path, nseg = _play_window_path(frames, snap, win)
        if nseg < PART_MIN_MOTION_FRAMES:
            keep.add(tid)   # motion unknown: fall back to 1+2 (in bounds) -> keep
            continue
        if path >= PART_MOVE_MIN:
            keep.add(tid)   # participates: real player who lined up wide
            continue
        # far AND non-participating -> reject (the coach signature).
    return keep


def _detect_bench(
    align: dict[int, tuple[float, float]],
    positions: dict[int, dict[int, tuple[float, float]]],
    colors: dict[int, tuple[str, float]],
    snap: int,
    fps: float,
) -> set[int]:
    """Track ids that form a sideline BENCH (a dense same-color edge row).

    The home / visitor bench stands packed in one team's colors along a
    sideline the whole play; the tracker counts them as that team and inflates
    its confident-color total past 11, which flips the O/D-role read (the
    inflated team's real front is diffused, so the compact opponent looks like
    the offense). A bench is the ONLY place on the field where MANY same-color
    bodies sit in a tight y-band right at an edge line while mostly NOT moving.

    We scan each field-edge y band for a run of > ``BENCH_MIN`` bodies of ONE
    jersey color, then confirm most of them are low-motion over the play window
    (subs don't run routes). Confirmed rows are returned for total exclusion.
    A real bunch / a stack of wide receivers fails at least one gate: too few,
    not at an edge, or moving with the play. Gate order matches the coach's
    description: dense row -> field-edge y band -> off-formation low motion."""
    if not align:
        return set()
    win = max(1, int(round(PART_WIN_S * fps)))
    bench: set[int] = set()
    bands = ((BENCH_Y_LO, BENCH_Y_HI), (BENCH_Y_LO2, BENCH_Y_HI2))
    for lo_y, hi_y in bands:
        # a dense row of ONE color inside this edge y-band, on the field in x.
        by_color: dict[str, list[int]] = {"A": [], "B": []}
        for tid, (x, y) in align.items():
            if not (lo_y <= y <= hi_y and 0.0 <= x <= 120.0):
                continue
            entry = colors.get(tid)
            if entry and entry[0] in ("A", "B") and entry[1] >= TEAM_COLOR_CONF:
                by_color[entry[0]].append(tid)
        for col, members in by_color.items():
            if len(members) <= BENCH_MIN:
                continue
            # ONE-TEAM dominance: a bench is a single team's subs. When the OTHER
            # color also crowds this band (a mixed scrum behind the boundary),
            # it is not a bench and is left alone.
            other = "B" if col == "A" else "A"
            if len(members) < BENCH_DOMINANCE * max(1, len(by_color[other])):
                continue
            # PACKED: keep only bodies shoulder-to-shoulder with the row — a
            # lone receiver split wide to the sideline sits in the band but
            # clear of the pack, so it fails this and stays a player
            # (sample-013's real WR).
            pts = {tid: align[tid] for tid in members}
            packed = [
                tid for tid in members
                if sum(
                    1 for other in members
                    if other != tid
                    and math.hypot(pts[tid][0] - pts[other][0],
                                   pts[tid][1] - pts[other][1]) <= BENCH_PACK_R
                ) >= BENCH_PACK_MIN
            ]
            if len(packed) <= BENCH_MIN:
                continue
            # off-formation low motion: subs stand / mill, they don't run the
            # play. A track too short to measure is treated as still (a barely
            # tracked bench body is still not a participant). A packed body that
            # IS running the play (a real bunch) keeps the row from stripping.
            still = 0
            for tid in packed:
                path, nseg = _play_window_path(positions.get(tid, {}), snap, win)
                if nseg < PART_MIN_MOTION_FRAMES or path < PART_MOVE_MIN:
                    still += 1
            if still >= BENCH_STILL_FRAC * len(packed):
                bench.update(packed)
    return bench


# ---------------------------------------------------------------- LOS / split


def _split_and_los(
    pts: list[tuple[float, float]],
    lo_bound: float | None = None,
    hi_bound: float | None = None,
) -> tuple[int, float, float, float]:
    """(split_k, los_x, left_front, right_front) for pts sorted by x.

    Optional bounds restrict candidate boundaries to midpoints inside
    [lo_bound, hi_bound] — used to let jersey colors pick the region and
    geometry pick the exact neutral zone within it.

    The LOS is the gap where many bodies crowd both facing edges (the two
    fronts stack on the ball), the gap looks like a neutral zone
    (~0.5-2.5 yd), and — decisively on real film, where inner defensive row
    gaps (corners|LBs, LBs|DL) can crowd just as hard — the offense side of
    the split has a QB/backfield: a body 3-8 yd behind its front near the
    formation's middle. No balance term: clutter makes cluster sizes
    meaningless.
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
        # LOS puts a backfield behind that front.
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


CORNER_DY = 8.0            # |dy| off the side's center that reads wide
CORNER_DEPTH_LO = 1.5      # a corner presses off its front by 1.5-9 yd;
CORNER_DEPTH_HI = 9.0      # split WRs sit ON the line, backs sit central


def _wide_deep_count(pts: list[tuple[float, float]], front: float,
                     behind_sign: float) -> int:
    """Bodies split wide AND off their own front line = corners = defense.

    The offense's wide bodies (receivers) align on/near the LOS and its deep
    bodies (QB/backs) align centrally — only a defense posts wide bodies
    yards behind its front. The most reliable O/D tell on cluttered film."""
    if len(pts) < 3:
        return 0
    center = _median([p[1] for p in pts])
    return sum(
        1 for x, y in pts
        if abs(y - center) >= CORNER_DY
        and CORNER_DEPTH_LO <= (front - x) * behind_sign <= CORNER_DEPTH_HI
    )


# ---------------------------------------------------------------- O/D role


def _line_count(
    pts: list[tuple[float, float]], front: float, center_y: float | None = None
) -> int:
    """Bodies within LINE_BAND of a side's front edge = on the LOS.

    Signal 1 (7-on-the-line): the offense stacks ~7 bodies on the ball (OL +
    on-line receivers/TEs); a defense fronts only 3-4 down linemen. The side
    with clearly more bodies at its own front is the offense — the strongest,
    purely rule-based O/D tell, and it survives a mid-play frame better than any
    depth heuristic because the LINE band hugs the ball the whole rep.

    When a formation lateral center is given, a body counts only if it sits
    within LINE_MAX_DY of it — the real front lives in the tackle box; a knot
    of bodies parked at a sideline (a bunch / a track cluster) is not a front
    (sample-053: a dark sideline knot at y~5, ~13 yд off the formation center)."""
    return sum(
        1 for x, y in pts
        if abs(x - front) <= LINE_BAND
        and (center_y is None or abs(y - center_y) <= LINE_MAX_DY)
    )


def _backfield_cluster(
    pts: list[tuple[float, float]], front: float, behind_sign: float
) -> bool:
    """True when a side has a COMPACT central backfield behind its front.

    Signal 2: the offense hides a tight 1-3 body cluster (QB + 1-2 backs)
    3-8 yд directly behind its front's lateral center. A defense has no such
    thing — its deep bodies are the secondary, spread WIDE across the field.
    So we require a small count of bodies in the backfield depth band that also
    sit near the front's lateral center; a wide spread back there is a defense
    and fails this test (it is caught instead by _wide_deep_count)."""
    if len(pts) < 3:
        return False
    center = _median([p[1] for p in pts])
    behind = [
        y for x, y in pts
        if BACKFIELD_LO <= (front - x) * behind_sign <= BACKFIELD_HI
        and abs(y - center) <= BACKFIELD_CENTER_DY
    ]
    return BACKFIELD_MIN <= len(behind) <= BACKFIELD_MAX


def _ol_contiguity(
    pts: list[tuple[float, float]], front: float, center_y: float | None = None
) -> int:
    """Longest run of shoulder-to-shoulder bodies at the front (the OL block).

    Signal 3: the offense packs ~5 linemen in one contiguous lateral row at a
    common depth; a defensive front is gapped (interior gaps, edges set off the
    ball). Take the bodies near the front depth, sort by lateral, and return the
    longest run where each adjacent pair sits within CONTIG_GAP — the OL row is
    the only place on the field this many bodies line up shoulder to shoulder.

    A run is only counted when it spans a real lateral width (>= CONTIG_MIN_SPAN
    yд): a knot of bodies crammed into a couple of yards at a sideline is a track
    cluster / a bunch, not an OL that stretches across the formation
    (sample-053: 5 dark bodies pinned at y=3-7 falsely read as an OL block).
    When a formation center is given the run must also STRADDLE it (bodies on
    both sides of center) — an OL centers on the ball, a sideline knot does not."""
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


def _offense_likeness(
    pts: list[tuple[float, float]], front: float, behind_sign: float,
    center_y: float | None = None,
) -> float:
    """Weighted offense-likeness score for one side of the LOS.

    Combines the rule-based football signals, each weighted by reliability:
    7-on-the-line (strongest), a compact central backfield, and a contiguous
    OL block push a side toward OFFENSE; wide-deep corners (the secondary) pull
    it toward DEFENSE. The front-crowding fraction is folded in at low weight as
    the legacy corroborator. Higher score = more offense-like; the caller
    compares the two sides. center_y (the formation's lateral center, over BOTH
    teams) anchors the front near the tackle box so a sideline knot cannot pose
    as an offensive front."""
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    # A real formation-side spreads across the field (OL + split receivers span
    # tens of yards); a side whose bodies are all crammed into a few yards of
    # lateral width is a track cluster / bunch pinned at a sideline, never an
    # 11-man offense (sample-053: the dark side is 7 bodies inside a ~4-yд knot
    # at y~5, the white side spans the whole field). A side that narrow is
    # physically incapable of fielding an offensive FRONT, so it earns NONE of
    # the offense-front bonuses (7-on-line, OL block, backfield) — those would be
    # false positives on a knot (5 bodies within CONTIG_GAP read as an "OL") —
    # and takes a flat DEFENSE-likeness penalty. Zeroing the bonuses (rather than
    # adding then subtracting) is what makes the knot lose decisively: line(7) +
    # contig(6) can otherwise swamp any fixed penalty.
    span = (max(ys) - min(ys)) if len(ys) >= 2 else 0.0
    if span < NARROW_SPAN:
        score = -W_NARROW
        score -= W_CORNER * _wide_deep_count(pts, front, behind_sign)
        score += W_FRONT_FRAC * _front_fraction(xs, front, front)
        return score
    score = W_LINE * _line_count(pts, front, center_y)
    if _backfield_cluster(pts, front, behind_sign):
        score += W_BACKFIELD
    contig = _ol_contiguity(pts, front, center_y)
    if contig >= CONTIG_MIN:
        score += W_CONTIG * contig
    score -= W_CORNER * _wide_deep_count(pts, front, behind_sign)
    score += W_FRONT_FRAC * _front_fraction(xs, front, front)
    return score


# ---------------------------------------------------------------- jersey color


def _load_colors(pdir: Path) -> dict[int, tuple[str, float]]:
    """{track_id: (team, confidence)} from team_colors.json (missing -> {})."""
    path = pdir / "team_colors.json"
    if not path.exists():
        return {}
    try:
        assignments = _load_json(path).get("assignments", {}) or {}
    except (json.JSONDecodeError, AttributeError):
        return {}
    out: dict[int, tuple[str, float]] = {}
    for tid_s, entry in assignments.items():
        try:
            out[int(tid_s)] = (
                str(entry.get("team", "unknown")),
                float(entry.get("confidence", 0.0)),
            )
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _confident_color(colors: dict[int, tuple[str, float]], tid: int) -> str | None:
    """'A' / 'B' when the jersey cluster vote is confident, else None."""
    entry = colors.get(tid)
    if entry and entry[0] in ("A", "B") and entry[1] >= COLOR_CONF:
        return entry[0]
    return None


def _cluster_color(colors: dict[int, tuple[str, float]], tid: int) -> str | None:
    """'A' / 'B' when a body CLEARLY belongs to a color cluster, else None.

    Relaxed sibling of _confident_color for the GLOBAL per-color TEAM tag: the
    O-vs-D decision is made once per color (which cluster is offense), so an
    individual body only needs to sit clearly in cluster A or B — not clear the
    strict COLOR_CONF bar the LOS split / formation-fit cost use. The floor
    still sits above the ref vote band so officials never inherit a team."""
    entry = colors.get(tid)
    if entry and entry[0] in ("A", "B") and entry[1] >= TEAM_COLOR_CONF:
        return entry[0]
    return None


def _near_los_color(colors: dict[int, tuple[str, float]], tid: int) -> str | None:
    """'A' / 'B' at the RELAXED neutral-zone bar, else None.

    Used ONLY for on-the-line bodies (|depth| <= NEAR_LOS_BAND), where side-of-
    LOS is the least reliable team signal and a moderate cluster vote is trusted
    over geometry. The bar (NEAR_LOS_COLOR_CONF) is looser than _cluster_color's
    global-team bar; a body below it is genuinely color-unknown and defers to
    side-of-LOS."""
    entry = colors.get(tid)
    if entry and entry[0] in ("A", "B") and entry[1] >= NEAR_LOS_COLOR_CONF:
        return entry[0]
    return None


def _color_lean(
    colors: dict[int, tuple[str, float]], tid: int, d_color: str | None
) -> float:
    """Vote share in (0, 0.5] by which a body's jersey leans AWAY from the
    defense's color; 0.0 when unknown, a ref, or leaning with the defense."""
    if d_color is None:
        return 0.0
    entry = colors.get(tid)
    if not entry or entry[0] not in ("A", "B") or entry[0] == d_color:
        return 0.0
    return max(0.0, entry[1] - 0.5)


def _color_split(
    core: dict[int, tuple[float, float]], colors: dict[int, tuple[str, float]]
) -> tuple[float, float, str] | None:
    """(gap_lo, gap_hi, left_color): the jersey-color-primary LOS region.

    Geometry alone mis-splits compressed formations (a jumbo set has no
    neutral-zone signature) — but the two teams' jersey clusters are known,
    so the LOS lives in the gap between the x-boundary that best separates
    confident-A from confident-B core bodies (fewest misclassified; widest
    gap breaks ties). The caller refines the exact line inside the gap with
    the geometric neutral-zone score (color-unknown bodies live there too).
    Returns None when either cluster is too thin to trust."""
    pts = sorted(
        (core[t][0], c)
        for t in core
        if (c := _confident_color(colors, t)) is not None
    )
    na = sum(1 for _, c in pts if c == "A")
    nb = len(pts) - na
    if na < COLOR_MIN_SIDE or nb < COLOR_MIN_SIDE:
        return None
    best: tuple[float, float, int, str] | None = None  # (err, -gap, k, left)
    for k in range(1, len(pts)):
        gap = pts[k][0] - pts[k - 1][0]
        a_right = sum(1 for _, c in pts[k:] if c == "A")
        b_left = sum(1 for _, c in pts[:k] if c == "B")
        for left, err in (
            ("A", a_right + b_left),
            ("B", (na - a_right) + (nb - b_left)),
        ):
            cand = (float(err), -gap, k, left)
            if best is None or cand < best:
                best = cand
    _, _, k, left = best
    return pts[k - 1][0], pts[k][0], left


# ---------------------------------------------------------------- strength


def _strength_sign(
    o_pos: dict[int, tuple[float, float]],
    los_x: float,
    dir_sign: float,
    center_y: float,
) -> tuple[float, float]:
    """(+1 high-y strong | -1 low-y strong, confidence).

    Strength (CONTRACT) = side with more offensive players outside the
    tackles. The OL is under-tracked on real film, so count split-out
    receivers (|dy| beyond the formation core) at/near the LOS instead;
    fall back to a wider net, then to the low-y side (deterministic)."""
    def odepth(x: float) -> float:
        return (los_x - x) * dir_sign

    hi = lo = 0
    for x, y in o_pos.values():
        dy = y - center_y
        if abs(dy) <= WIDE_DY:
            continue
        if -1.0 <= odepth(x) <= WIDE_ON_LOS:
            hi, lo = hi + (dy > 0), lo + (dy < 0)
    if hi != lo:
        return (1.0 if hi > lo else -1.0), 1.0
    hi = lo = 0
    for x, y in o_pos.values():
        dy = y - center_y
        if abs(dy) <= WIDE_FALLBACK_DY:
            continue
        if -1.0 <= odepth(x) <= WIDE_ANY:
            hi, lo = hi + (dy > 0), lo + (dy < 0)
    if hi != lo:
        return (1.0 if hi > lo else -1.0), 0.8
    return -1.0, 0.6


# ---------------------------------------------------------------- defense


def _role_slots(
    dl_d: float, lb_d: float, deep_d: float,
    bc: float, cw_t: float, cs_t: float,
) -> dict[str, tuple[float, float]]:
    """Template (depth, lateral) slot per role from robust formation stats.

    Laterals live on the coach-verified chain cb_w-saf_w-lb_c-lb_b-lb_a-
    F-cb_s (weak -> strong; positive u = strong side): the LB trio brackets
    the back-seven center bc, corners sit on their side's wide anchor, and
    saf_w / F interpolate the chain between corner and trio. All stats are
    medians / per-side extremes over the pool, so removing any one body
    barely moves the template — the cascade-prevention property."""
    slots: dict[str, tuple[float, float]] = {}
    for r, t in DL_LAT.items():
        slots[r] = (dl_d, t)
    b_t, m_t, s_t = bc - LB_SPACING, bc, bc + LB_SPACING
    slots["lb_c"] = (lb_d, b_t)
    slots["lb_b"] = (lb_d, m_t)
    slots["lb_a"] = (lb_d, s_t)
    slots["saf_w"] = (lb_d + 1.0, (cw_t + b_t) / 2.0)
    slots["F"] = (deep_d, (s_t + cs_t) / 2.0)
    slots["cb_w"] = (max(lb_d, 3.0), cw_t)
    slots["cb_s"] = (max(lb_d, 3.0), cs_t)
    return slots


def _role_cost(
    role: str, d: float, u: float, wide: float,
    slots: dict[str, tuple[float, float]], dl_d: float, lb_d: float,
    deep_d: float, deep2: float,
    thin_dl: tuple[float, bool] | None,
) -> float:
    """Soft cost of putting a body at (depth d, signed lateral u) in a role.

    The same signals the old sequential rules used, expressed as costs:
    depth-from-LOS priors per role group (DL shallow, backs off the line),
    cross-field chain consistency via the slot laterals, edge/interior width
    for the DL, corner wideness, and depth attraction for the F (adaptive:
    when homography compresses depth the term shrinks for every candidate
    and the chain lateral priors decide instead)."""
    d_t, u_t = slots[role]
    row_hi = dl_d + DL_ROW_TOL
    pen_line = W_BACK_ONLINE * max(0.0, row_hi + 0.3 - d)
    if role in DL_LAT:
        if thin_dl is None:
            lat = abs(u - u_t)
        else:
            thin_dl_delta, edge_pair = thin_dl
            # Thin DL row (<3 tracked): the row's own centroid says nothing,
            # so measure the shade off the whole defense's median (the old
            # measured-win rule): nose band, edge hinge, 3-tech between.
            dy = u - thin_dl_delta
            if role == "N":
                lat = max(0.0, abs(dy) - NOSE_WIDTH)
            elif role == "dl_es":
                lat = max(0.0, EDGE_DY - dy)
            elif role == "dl_ew":
                lat = max(0.0, EDGE_DY + dy)
            else:  # T: the 3-tech band between nose and strong edge
                lat = abs(dy - (NOSE_WIDTH + EDGE_DY) / 2.0)
            if edge_pair and role in ("N", "T"):
                lat += EDGE_PAIR_PEN / W_DL_LAT
        c = W_DL_LAT * lat
        c += W_DL_DEPTH * max(0.0, d - row_hi, (dl_d - DL_BEHIND) - d)
        # LOS-anchored depth gate: independent of the (LB-contaminable) row
        # centroid, this ramps with ABSOLUTE depth off the LOS so an off-the-
        # line body cannot win a DL role even when it is the shallowest thing
        # tracked. Soft through the true-DL band (<= DL_LOS_OK), steep by
        # linebacker depth.
        c += W_DL_LOS_DEPTH * max(0.0, d - DL_LOS_OK)
        if wide > DL_MAX_DY:
            c += DL_WIDE_PEN
        if d > row_hi + DL_HARD_SLACK or d < dl_d - DL_BEHIND - DL_HARD_SLACK:
            c += DL_HARD_PEN   # clearly off the line: never crammed into DL
        return c
    if role in ("lb_c", "lb_b", "lb_a"):
        return (W_LB_LAT * abs(u - u_t) + pen_line
                + W_LB_DTGT * abs(d - lb_d)
                + W_LB_DEEP * max(0.0, d - LB_DEEP_OK))
    if role == "saf_w":
        c = (W_WHIP_LAT * abs(u - u_t) + pen_line
             + W_WHIP_DEPTH * max(0.0, lb_d - d))
        if d >= lb_d + UMPIRE_MARGIN and deep_d - d > 0.05 and (
            deep_d - d >= F_DEPTH_MARGIN
            or d >= lb_d + SAFETY_DEPTH
        ):
            # Two-deep-safeties look: the SECOND-deepest deep body is the
            # saf_w whatever its side (the deepest is always the F). A body
            # at true safety depth qualifies even when the pair is nearly
            # level; the umpire (just off the LB row) never does.
            c = min(c, W_WS_DEPTH * abs(deep2 - d) + W_WS_LAT * abs(u)
                    + pen_line)
        return c
    if role == "F":
        return (W_F_DEPTH * (deep_d - min(d, deep_d))
                + W_F_LAT * abs(u - u_t) + pen_line)
    return W_C_LAT * abs(u - u_t) + W_C_DEEP * max(0.0, d - C_DEEP_OK)


def _order_align(
    pool: list[int], chain: list[str], emit: dict[int, dict[str, float]]
) -> tuple[float, dict[int, str]]:
    """Min-cost monotone alignment of chain roles to u-sorted bodies (DP).

    pool: candidate indices sorted weak -> strong; emit[i][role]: soft cost.
    Matched pairs are strictly increasing on BOTH sides, so every pairwise
    chain-order constraint (cb_w outside saf_w outside lb_c ... inside cb_s)
    holds by construction — the property the independent Hungarian cannot
    express. Skipping a body (unassigned) or a role (unfilled) costs
    ASSIGN_ABSTAIN, the same dummy semantics as before: a missing body
    leaves exactly its role empty with no neighbor drift.

    Adjacency terms (also inexpressible per-cell): matching saf_w while cb_w
    was skipped pays ORDER_PEN (an overhang with no corner outside him —
    every weaker body is then unassigned, or the saf_w body is outermost);
    skipping cb_s after matching an in-chain F pays the same on the strong
    side. A one-bit "previous role filled" flag in the state carries both.

    Interior-gap term (the one the coach flagged on sample-089): leaving a
    chain role UNFILLED with filled roles on BOTH sides of it is football
    nonsense — the outermost weak second-level body IS the saf_w, so when a
    body is short you push the fill OUTWARD and leave an END role empty, not
    a middle one. A bare role-skip at ASSIGN_ABSTAIN charges an interior hole
    the same as an end hole, so the DP happily brackets a gap. GAP_PEN fixes
    that: a role-skip that lands BETWEEN a fill and a later fill pays extra.
    "Later fill" is unknowable at skip time in a forward DP, so the penalty
    is deferred — the state carries a small counter `sg` of bracket-pending
    skips (roles skipped after the first fill and not yet closed by a later
    fill), and every MATCH commits GAP_PEN * sg for the skips it just
    bracketed. Trailing skips after the last fill (a strong-END gap) are
    never closed by a match, so they never pay — end gaps stay free, only
    interior gaps cost, and the fill shifts out toward the weak end.

    Returns (total_cost, {candidate index: role})."""
    nb, nr = len(pool), len(chain)
    inf = float("inf")
    # State: dp[i][j][pf][af][sg]
    #   pf = immediate-previous role filled (carries the ORDER_PEN adjacency)
    #   af = ANY role filled so far (monotone; gates gap accrual)
    #   sg = count of bracket-pending role-skips since the last fill (roles
    #        skipped while af=1, awaiting a later fill to bill GAP_PEN)
    def _cell() -> list[list[list[float]]]:
        return [[[inf] * (nr + 1) for _ in range(2)] for _ in range(2)]

    dp = [[_cell() for _ in range(nr + 1)] for _ in range(nb + 1)]
    par: dict[
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int, str],
    ] = {}
    dp[0][0][0][0][0] = 0.0

    def relax(i: int, j: int, pf: int, af: int, sg: int, v: float,
              fi: int, fj: int, fpf: int, faf: int, fsg: int, act: str) -> None:
        if v < dp[i][j][pf][af][sg]:
            dp[i][j][pf][af][sg] = v
            par[(i, j, pf, af, sg)] = (fi, fj, fpf, faf, fsg, act)

    for i in range(nb + 1):
        for j in range(nr + 1):
            for pf in (0, 1):
                for af in (0, 1):
                    for sg in range(nr + 1):
                        cur = dp[i][j][pf][af][sg]
                        if cur == inf:
                            continue
                        if i < nb:   # body unassigned
                            relax(i + 1, j, pf, af, sg, cur + ASSIGN_ABSTAIN,
                                  i, j, pf, af, sg, "b")
                        if j < nr:   # role unfilled
                            pen = ORDER_PEN if (
                                chain[j] == "cb_s" and j > 0
                                and chain[j - 1] == "F" and pf
                            ) else 0.0
                            # A skip after any fill is a bracket candidate: it
                            # only pays once (if) a later fill closes it.
                            nsg = sg + af
                            relax(i, j + 1, 0, af, nsg,
                                  cur + ASSIGN_ABSTAIN + pen,
                                  i, j, pf, af, sg, "r")
                        if i < nb and j < nr:   # match
                            pen = ORDER_PEN if (
                                chain[j] == "saf_w" and j > 0
                                and chain[j - 1] == "cb_w" and not pf
                            ) else 0.0
                            # This fill closes every pending bracketed skip.
                            pen += GAP_PEN * sg
                            relax(i + 1, j + 1, 1, 1, 0,
                                  cur + emit[pool[i]][chain[j]] + pen,
                                  i, j, pf, af, sg, "m")

    best = None  # (total, pf, af, sg)
    for pf in (0, 1):
        for af in (0, 1):
            for sg in range(nr + 1):
                v = dp[nb][nr][pf][af][sg]
                if best is None or v < best[0]:
                    best = (v, pf, af, sg)
    total, pf, af, sg = best
    matches: dict[int, str] = {}
    i, j = nb, nr
    while (i, j, pf, af, sg) != (0, 0, 0, 0, 0):
        fi, fj, fpf, faf, fsg, act = par[(i, j, pf, af, sg)]
        if act == "m":
            matches[pool[fi]] = chain[fj]
        i, j, pf, af, sg = fi, fj, fpf, faf, fsg
    return total, matches


def _solve_roles(
    cands: list[tuple[int, float, float, float]],
    center: float, s: float,
    slots: dict[str, tuple[float, float]],
    dl_d: float, lb_d: float, deep_d: float, deep2: float,
    thin_dl: tuple[float, bool] | None = None,
    row_pen: dict[int, float] | None = None,
) -> tuple[dict[int, str], dict[int, float], set[str]]:
    """Global one-to-one assignment of ROLES to candidates.

    Two solvers over ONE soft-cost matrix: the joint Hungarian (augmented
    with dummy rows/cols at ASSIGN_ABSTAIN so <11 candidates leave roles
    unfilled and >11 leave bodies unassigned) fixes the DL group — it
    weighs each lineman against every back-role alternative exactly as
    before. The back seven is then re-solved by the order-preserving DP
    (_order_align) over the remaining bodies: roles in coach-chain order x
    bodies sorted weak -> strong, with the F and a two-high saf_w matched
    OUT of the chain when a body reads truly deep (deep safeties are the
    one legitimate order exemption; the global minimum over those picks x
    the DP decides). row_pen adds a flat per-body cost to EVERY role
    (jersey-color lean): it never changes which role a body prefers, only
    whether it deserves one at all — and it loses duels to a same-geometry
    unpenalized body. Returns ({tid: role},
    {tid: cost margin to its second-best role}, unfilled role names)."""
    n, m = len(cands), len(ROLES)
    if n == 0:
        return {}, {}, set(ROLES)
    # A body whose jersey-color lean is "effectively forbidden" (row_pen at or
    # beyond the 2x ASSIGN_ABSTAIN bar the color penalty was tuned against) is
    # not a defensive-role candidate AT ALL — it is removed from BOTH solvers
    # so it can never claim a role. This keeps the color forbiddance a hard
    # floor that GAP_PEN cannot buy through: leaving a chain role empty
    # (ASSIGN_ABSTAIN + GAP_PEN) must never look cheaper than cramming a white
    # jersey into it, and the only robust way to guarantee that as GAP_PEN
    # grows is to deny the forbidden body a seat outright (it still earns a
    # low-confidence fallback name from the caller iff its color allows).
    forbid_bar = 2.0 * ASSIGN_ABSTAIN
    forbidden = {
        i for i in range(n)
        if (row_pen.get(cands[i][0], 0.0) if row_pen else 0.0) >= forbid_bar
    }
    cost = np.zeros((n, m))
    for i, (tid, _, y, d) in enumerate(cands):
        u = (y - center) * s
        wide = abs(y - center)
        pen = row_pen.get(tid, 0.0) if row_pen else 0.0
        for j, role in enumerate(ROLES):
            cost[i, j] = pen + _role_cost(role, d, u, wide, slots, dl_d, lb_d,
                                          deep_d, deep2, thin_dl)
    big = 4.0 * ASSIGN_ABSTAIN + 100.0
    size = n + m
    mat = np.full((size, size), big)
    mat[:n, :m] = cost
    if forbidden:   # a forbidden body's every real role costs more than its
        for i in forbidden:   # dummy: the Hungarian always leaves it unassigned
            mat[i, :m] = big
    mat[np.arange(n), m + np.arange(n)] = ASSIGN_ABSTAIN  # body unassigned
    mat[n + np.arange(m), np.arange(m)] = ASSIGN_ABSTAIN  # role unfilled
    mat[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(mat)
    assigned: dict[int, str] = {}   # candidate index -> role
    for r, c in zip(rows, cols):
        if r < n and c < m and ROLES[c] in DL_LAT:
            assigned[r] = ROLES[c]

    # ---- back seven: order-preserving DP over the coach chain. Forbidden
    # bodies are excluded from the pool entirely (never a chain seat).
    back_idx = [i for i in range(n) if i not in assigned and i not in forbidden]
    us = {i: (cands[i][2] - center) * s for i in back_idx}
    order = sorted(back_idx, key=lambda i: (us[i], cands[i][3], cands[i][0]))
    emit = {
        i: {role: float(cost[i, ROLES.index(role)]) for role in BACK_CHAIN}
        for i in back_idx
    }
    # Deep safeties may sit anywhere laterally (deep-middle F, two-high
    # saf_w): bodies that read TRULY deep are eligible to take those roles
    # out of band. The saf_w gate is hypothesis two's own predicate; the F
    # gate is "clearly beyond the LB row, middle of field".
    deep_f = [
        i for i in back_idx
        if abs(us[i]) <= F_SIDE_DY_MAX and cands[i][3] >= lb_d + UMPIRE_MARGIN
    ]
    deep_w = [
        i for i in back_idx
        if cands[i][3] >= lb_d + UMPIRE_MARGIN
        and deep_d - cands[i][3] > 0.05
        and (deep_d - cands[i][3] >= F_DEPTH_MARGIN
             or cands[i][3] >= lb_d + SAFETY_DEPTH)
    ]
    best: tuple[float, dict[int, str], int | None, int | None] | None = None
    for f_i in [None] + deep_f:
        for w_i in [None] + deep_w:
            if w_i is not None and w_i == f_i:
                continue
            chain = [
                r for r in BACK_CHAIN
                if not (r == "F" and f_i is not None)
                and not (r == "saf_w" and w_i is not None)
            ]
            pool = [i for i in order if i != f_i and i != w_i]
            base = (emit[f_i]["F"] if f_i is not None else 0.0) + (
                emit[w_i]["saf_w"] if w_i is not None else 0.0
            )
            total, matches = _order_align(pool, chain, emit)
            if best is None or base + total < best[0]:
                best = (base + total, matches, f_i, w_i)
    _, matches, f_i, w_i = best
    assigned.update(matches)
    if f_i is not None:
        assigned[f_i] = "F"
    if w_i is not None:
        assigned[w_i] = "saf_w"

    roles_of: dict[int, str] = {}
    margins: dict[int, float] = {}
    for r, role in assigned.items():
        tid = cands[r][0]
        c = ROLES.index(role)
        roles_of[tid] = role
        others = [cost[r, j] for j in range(m) if j != c]
        margins[tid] = (min(others) - cost[r, c]) if others else 1.0
    return roles_of, margins, set(ROLES) - set(roles_of.values())


def _label_defense(
    d_pos: dict[int, tuple[float, float]],
    o_pos: dict[int, tuple[float, float]],
    los_x: float,
    dir_sign: float,
    base_conf: float,
    colors: dict[int, tuple[str, float]] | None = None,
    d_color: str | None = None,
) -> tuple[dict[str, str], dict[str, float], list[tuple[float, float, str]]]:
    """Fit the formation structure to the core defenders.

    colors / d_color (the defense's jersey cluster, when known) make jersey
    color a SOFT assignment cost: a body whose cluster vote leans the
    offense's color pays COLOR_LEAN_W * (share - 0.5) for ANY defensive role.

    Returns (positions, confidence, anchors) where anchors are
    (x, y, role) points — the fitted defenders plus interpolated "virtual"
    slots for unfilled back-seven roles — used to classify non-core
    defenders (track fragments) by nearest neighbor.
    """
    colors = colors or {}
    positions: dict[str, str] = {}
    confidence: dict[str, float] = {}
    anchors: list[tuple[float, float, str]] = []
    if not d_pos:
        return positions, confidence, anchors

    def depth(x: float) -> float:
        return (x - los_x) * dir_sign

    items = [(tid, x, y, depth(x)) for tid, (x, y) in d_pos.items()]
    center0 = _median([y for _, _, y, _ in items])

    # Sideline officials stand on the LOS at the boundary; drop them from the
    # formation fit (they still get a team + fallback label from the caller).
    items = [
        it for it in items
        if not (it[3] <= SIDELINE_OFFICIAL_DEPTH
                and (it[2] <= SIDELINE_Y or it[2] >= 53.33 - SIDELINE_Y))
    ]

    # A body ON the LOS way out wide with no offensive player to cover is a
    # sideline sub in team colors, not a press corner — drop it from the fit.
    o_ys = [p[1] for p in o_pos.values()]
    idle_wide = [
        it for it in items
        if it[3] <= WIDE_IDLE_DEPTH and abs(it[2] - center0) >= WIDE_IDLE_DY
        and not any(abs(it[2] - oy) <= WIDE_COVER_DY for oy in o_ys)
    ]
    items = [it for it in items if it not in idle_wide]

    # ---- DL: the shallow row on the ball. This team's front is ALWAYS the
    # four dl_es/T/N/dl_ew — a fifth body at DL-ish depth is a walked-up LB, so
    # the row is the shallow depth cluster, capped at four.
    cand = sorted(
        (
            it for it in items
            if -DL_BEHIND <= it[3] <= DL_DEPTH
            and abs(it[2] - center0) <= DL_MAX_DY
        ),
        key=lambda it: it[3],
    )
    dl = []
    if cand:
        row = cand[0][3]  # anchored at the shallowest body on the ball
        dl = [it for it in cand if it[3] <= row + DL_ROW_TOL][:4]

    # ---- back seven (never across the LOS; deep central bodies are clutter)
    back = sorted(
        (it for it in items if it not in dl and it[3] >= BACK_MIN_DEPTH),
        key=lambda it: it[2],
    )
    unassigned = [
        it for it in items if it not in dl and it not in back
    ] + idle_wide
    deep = [
        it for it in back
        if it[3] > BACK_MAX_DEPTH
        and (abs(it[2] - center0) <= WIDE_EXEMPT_DY or it[3] > WIDE_EXEMPT_MAX)
    ]
    if deep and len(back) - len(deep) >= BACK_KEEP_MIN:
        back = [it for it in back if it not in deep]
        unassigned.extend(deep)

    # Formation center: the DL row when tracked, else the (pruned) back seven
    # — the raw items median drags toward sideline clutter.
    if len(dl) >= 2:
        center = sum(it[2] for it in dl) / len(dl)
    elif back:
        center = _median([it[2] for it in back])
    else:
        center = center0

    # ---- strength: the defense declares it — the F (a clearly-deeper
    # interior back, off-center but not corner-wide) aligns to the strong
    # side. When no such body reads cleanly, fall back to counting split-out
    # offensive receivers (CONTRACT definition), which survives clutter less
    # well but needs no defensive structure.
    s = s_conf = None
    interior = back[1:-1] if len(back) >= 5 else back
    if len(interior) >= 3:
        deepest = max(interior, key=lambda it: it[3])
        med = _median([it[3] for it in interior])
        dy = deepest[2] - center
        if (deepest[3] >= med + F_CLEAR_MARGIN
                and F_SIDE_DY_MIN <= abs(dy) <= F_SIDE_DY_MAX):
            s, s_conf = (1.0 if dy > 0 else -1.0), 0.9
    if s is None:
        s, s_conf = _strength_sign(o_pos, los_x, dir_sign, center)

    def assign(tid: int, name: str, conf: float) -> None:
        positions[str(tid)] = name
        confidence[str(tid)] = max(0.05, min(1.0, conf))

    # ---- global one-to-one assignment: 11 roles x all candidate defenders.
    # Robust template stats (medians / extremes over the pool) parameterize
    # the per-role costs; the Hungarian solve picks the jointly-best map.
    dl_d = _median([it[3] for it in dl]) if dl else 0.7
    lb_src = interior if interior else back
    lb_d = min(7.0, max(1.0, _median([it[3] for it in lb_src]))) if lb_src else 4.0

    def u_of(y: float) -> float:
        return (y - center) * s

    # Deep stats come from CENTRAL backs only — the F is the deepest
    # middle-of-field safety; corner-wide bodies are never the F.
    central_backs = [it for it in back if abs(u_of(it[2])) <= F_SIDE_DY_MAX]
    c_ds = sorted((min(it[3], 14.0) for it in central_backs), reverse=True)
    deep_d = max(lb_d + 3.0, c_ds[0] if c_ds else 0.0)
    deep2 = c_ds[1] if len(c_ds) > 1 else deep_d

    # ---- umpire prune (ported measured win): with more box-width bodies
    # than LB slots, clear depth deviants are officials (umpire deep middle,
    # down judge shallow) — never the deepest central back (the F) nor a
    # second-deep saf_w candidate.
    deepest = (max(central_backs, key=lambda it: it[3])
               if central_backs else None)
    box = [
        it for it in back
        if abs(u_of(it[2])) <= 2 * LB_SPACING
        and it is not deepest
        and not (it[3] >= lb_d + UMPIRE_MARGIN
                 and (deep_d - it[3] >= F_DEPTH_MARGIN
                      or it[3] >= lb_d + SAFETY_DEPTH))
    ]
    while len(box) > 3:
        med = _median([it[3] for it in box])
        worst = max(box, key=lambda it: abs(it[3] - med))
        if abs(worst[3] - med) < BOX_PRUNE_DEV:
            break
        box.remove(worst)
        back = [it for it in back if it is not worst]
        unassigned.append(worst)

    cands = dl + [it for it in back if it not in dl]

    # Back-seven center: anchored on the DL center (stable when any back's
    # track drops — the cascade-prevention anchor), refined by the mean
    # lateral of box-width interior backs but only within a small clip.
    # Wide anchors per side: the widest defensive back, else the widest
    # offensive receiver (a corner aligns over his man), else a fixed
    # fallback — gated so a box body never reads as a side's anchor, and
    # never a body projecting out of bounds (play 019: a sideline body at
    # y < 0 anchored the weak corner target on itself and stole cb_w, which
    # slid the real cb_w -> saf_w -> lb_c chain one seat inward).
    o_us = [u_of(y) for _, y in o_pos.values()
            if FIELD_Y_LO <= y <= FIELD_Y_HI]
    pool = list(lb_src)
    if len(pool) > 3:   # the deepest interior back is the F candidate,
        pool.remove(max(pool, key=lambda it: it[3]))   # not a box body
    pool_us = [u for it in pool if abs(u := u_of(it[2])) <= 2 * LB_SPACING]
    d_bc = sum(pool_us) / len(pool_us) if pool_us else 0.0
    bc = max(-BC_CLIP, min(BC_CLIP, d_bc))
    back_us = [u_of(it[2]) for it in back
               if FIELD_Y_LO <= it[2] <= FIELD_Y_HI]

    def wide_anchor(sign: float) -> float:
        d_side = [u * sign for u in back_us if u * sign >= WIDE_ANCHOR_MIN]
        if d_side:
            return max(d_side) * sign
        o_side = [u * sign for u in o_us if u * sign >= WIDE_ANCHOR_MIN]
        if o_side:
            return max(o_side) * sign
        return CORNER_FALLBACK * sign

    cw_t, cs_t = wide_anchor(-1.0), wide_anchor(1.0)
    slots = _role_slots(dl_d, lb_d, deep_d, bc, cw_t, cs_t)
    thin_dl: tuple[float, bool] | None = None
    if len(dl) < 3:
        delta = (center0 - center) * s
        edge_pair = False
        if len(dl) == 2:
            dy0, dy1 = (u_of(it[2]) - delta for it in dl)
            edge_pair = (dy0 * dy1 < 0
                         and abs(dy0 - dy1) >= EDGE_PAIR_SPREAD)
        thin_dl = (delta, edge_pair)
    row_pen = {
        it[0]: COLOR_LEAN_W * lean
        for it in cands
        if (lean := _color_lean(colors, it[0], d_color)) > 0.0
    }
    roles_of, margins, unfilled = _solve_roles(
        cands, center, s, slots, dl_d, lb_d, deep_d, deep2, thin_dl, row_pen
    )
    xy_of = {it[0]: (it[1], it[2]) for it in cands}
    for tid, role in roles_of.items():
        factor = max(0.3, min(0.95, 0.45 + 0.5 * margins[tid] / MARGIN_SCALE))
        assign(tid, role, base_conf * s_conf * factor)

    # ---- anchors: assigned defenders at their real spots + template slots
    # for unfilled roles (fragments near a missing slot — e.g. an
    # untracked-at-snap saf_w — still land on the right name).
    anchors = [(xy_of[t][0], xy_of[t][1], r) for t, r in roles_of.items()]
    for role in unfilled:
        d_t, u_t = slots[role]
        anchors.append((los_x + d_t * dir_sign, center + u_t * s, role))

    # Best-effort names for pruned / abstained bodies (officials, extras)
    # at low confidence. Same gates as the caller's fragment fallback (play
    # 093: end-zone officials 10+ yd behind the whole defense held "F", and
    # an abstained wrong-color body would re-enter here): never deeper than
    # the deepest fitted defender + slack, never past the end lines, never a
    # body whose jersey leans the offense's color.
    unassigned.extend(it for it in cands if it[0] not in roles_of)
    fit_depths = [it[3] for it in cands if it[0] in roles_of]
    max_fit_depth = max(fit_depths) if fit_depths else POSITION_RANGE
    for it in unassigned:
        if str(it[0]) in positions:
            continue
        if it[3] > max_fit_depth + FALLBACK_DEPTH_SLACK:
            continue  # behind the entire fitted defense = clutter
        if not (0.0 <= it[1] <= 120.0):
            continue  # past an end line: no position claims
        if _color_lean(colors, it[0], d_color) > 0.0:
            continue  # jersey leans the offense's color
        name = _nearest_anchor(anchors, it[1], it[2])
        if name:
            assign(it[0], name, 0.2)

    return positions, confidence, anchors


def _nearest_anchor(
    anchors: list[tuple[float, float, str]], x: float, y: float,
    exclude: set[str] | None = None,
) -> str | None:
    pool = [a for a in anchors if not exclude or a[2] not in exclude]
    if not pool:
        return None
    ax, ay, name = min(pool, key=lambda a: (a[0] - x) ** 2 + (a[1] - y) ** 2)
    return name


# ------------------------------------------------- generic offense classifier
# The defense gets full role logic above; the offense is tagged generically from
# alignment (on-line = OL, split wide = WR, deep-centered = QB, else RB).
_OFF_LINE_BAND = 1.8    # within this of the LOS reads as on the line
_OFF_WIDE_YD = 12.0     # this far from the formation center reads as split wide
_OFF_BACKFIELD_YD = 3.0  # bodies deeper than this off the line are backs / QB
_OFF_QB_LAT = 3.0       # a back within this of center (and deepest) is the QB


def _classify_offense(off_ids, pts, los_x):
    """{track_id: OL/QB/RB/WR} for the offense from field-relative alignment."""
    if not off_ids:
        return {}
    center_y = _median([pts[t][1] for t in off_ids])
    out: dict[int, str] = {}
    for t in off_ids:
        x, y = pts[t]
        depth = abs(x - los_x)
        lateral = abs(y - center_y)
        if lateral >= _OFF_WIDE_YD:
            out[t] = "WR"
        elif depth <= _OFF_LINE_BAND:
            out[t] = "OL"
        elif depth >= _OFF_BACKFIELD_YD:
            out[t] = "QB" if lateral <= _OFF_QB_LAT else "RB"
        else:
            out[t] = "WR"
    # exactly one QB: keep the deepest centered back, demote the rest to RB
    qbs = [t for t in out if out[t] == "QB"]
    if len(qbs) > 1:
        deepest = max(qbs, key=lambda t: abs(pts[t][0] - los_x))
        for t in qbs:
            if t != deepest:
                out[t] = "RB"
    return out


# ---------------------------------------------------------------- entry point


def run(play_id: str, data_dir: Path) -> None:
    """Compute labels.json for one play from field_coords.json (+ meta.json)."""
    pdir = play_dir(play_id, data_dir)
    field = _load_json(pdir / "field_coords.json")
    fps = 30.0
    meta_path = pdir / "meta.json"
    if meta_path.exists():
        try:
            fps = float(_load_json(meta_path).get("fps", 30.0)) or 30.0
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    positions = _track_positions(field)
    labels: dict[str, Any] = {
        "play_id": play_id,
        "snap_frame": 0,
        "los_x": 60.0,
        "offense_direction": "+x",
        "teams": {},
        "positions": {},
        "position_confidence": {},
    }
    if not positions:
        _write(pdir, labels)
        return

    # Human-marked snap (snap.json) beats motion detection.
    snap_path = pdir / "snap.json"
    human_snap: int | None = None
    if snap_path.exists():
        try:
            human_snap = int(_load_json(snap_path)["frame"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            human_snap = None

    speeds = _mean_speeds(_smooth_positions(positions), fps)
    if human_snap is not None:
        snap, snap_confident = human_snap, True
        labels["snap_frame"] = snap
        labels["snap_source"] = "human"
        align_anchor = snap
    else:
        # Interior-line still->burst detector (auto_snap smooths internally).
        snap, snap_confident, low_thr = auto_snap(positions, fps)
        labels["snap_frame"] = snap
        labels["snap_source"] = "auto"
        # Anchor the alignment window at the end of the pre-snap still period.
        align_anchor = snap
        for i in range(snap - 1, 0, -1):
            if i < len(speeds) and speeds[i] <= low_thr:
                align_anchor = min(snap, i + 1)
                break
    base_conf = 0.9 if snap_confident else 0.55

    align = _alignment_positions(positions, align_anchor)

    # Jersey-color clusters (team_colors.json): the PRIMARY team signal.
    # Confident striped officials are excluded from everything — they never
    # join the formation fit, never get a team, never get a position.
    colors = _load_colors(pdir)
    refs = {tid for tid, (team, _) in colors.items() if team == "ref"}
    refs_hard = {
        tid for tid, (team, conf) in colors.items()
        if team == "ref" and conf >= REF_CONF
    }

    # ---- sideline BENCH: a dense same-color edge row of low-motion subs. It
    # inflates its team's confident-color count past 11 and, left in, flips the
    # O/D-role read (the compact opponent looks like the offense). Excluded from
    # EVERYTHING downstream — the core, the color-split LOS region, the O/D
    # scorer, the participant set, team tags and positions — exactly like refs.
    bench = _detect_bench(align, positions, colors, snap, fps)
    refs = refs | bench
    refs_hard = refs_hard | bench

    # ---- PIXEL-SPACE pre-filter (runs FIRST, before the O/D split / LOS /
    # strength / formation centroid). Field yards are corrupted by homography
    # extrapolation for bodies far off the calibrated yard lines (a sideline
    # coach projects to shallow, in-formation-looking coords AND drags the
    # centroid off), so field space cannot reject them. Pixel feet-positions
    # from tracks.json boxes carry no homography error: cluster them and drop
    # the sideline fringe (far + small box + high in frame) before anything
    # field-space ever sees it. tracks.json shares track_ids with
    # field_coords.json; a missing/broken tracks.json simply skips the filter.
    pix_reject: set[int] = set()
    tracks_path = pdir / "tracks.json"
    if tracks_path.exists():
        try:
            raw_tracks = _load_json(tracks_path).get("tracks", []) or []
            meta = _load_json(meta_path) if meta_path.exists() else {}
            width = float(meta.get("width", 0.0) or 0.0)
            height = float(meta.get("height", 0.0) or 0.0)
            pix_tracks: dict[int, list[dict[str, Any]]] = {}
            for t in raw_tracks:
                try:
                    pix_tracks[int(t["track_id"])] = t.get("frames", []) or []
                except (KeyError, TypeError, ValueError):
                    continue
            feet = _pixel_feet(pix_tracks, snap)
            pix_reject = _pixel_reject(feet, width, height)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pix_reject = set()

    # ---- core selection: alive near the snap, plausibly on the field,
    # inside the densest formation x-window; never an official; never a
    # pixel-space sideline outlier.
    core: dict[int, tuple[float, float]] = {}
    for tid, frames in positions.items():
        if tid in refs or len(frames) < MIN_CORE_FRAMES:
            continue
        lo, hi = min(frames), max(frames)
        if hi < snap - CORE_BEFORE or lo > snap + CORE_AFTER:
            continue
        x, y = align[tid]
        if CORE_Y_MIN <= y <= CORE_Y_MAX and 0.0 <= x <= 120.0:
            core[tid] = (x, y)

    if len(core) >= 8:
        xs_all = sorted(x for x, _ in core.values())
        best_lo, best_cnt = xs_all[0], 0
        j = 0
        for i, lo in enumerate(xs_all):
            while j < len(xs_all) and xs_all[j] <= lo + DENSE_WIN:
                j += 1
            if j - i > best_cnt:
                best_cnt, best_lo = j - i, lo
        windowed = {
            t: p for t, p in core.items() if best_lo <= p[0] <= best_lo + DENSE_WIN
        }
        if len(windowed) >= 8:
            core = windowed

    tids = sorted(core, key=lambda t: core[t][0])
    xs = [core[t][0] for t in tids]
    if len(xs) < 4:
        labels["los_x"] = xs[len(xs) // 2] if xs else 60.0
        labels["teams"] = {str(t): "O" for t in align if t not in refs}
        _write(pdir, labels)
        return

    # ---- LOS: jersey color first — the boundary that best separates the two
    # confident color groups (geometry only breaks ties / fills gaps). Pure
    # geometry remains the fallback when either cluster is under-sampled.
    split = _color_split(core, colors)
    if split is not None:
        gap_lo, gap_hi, left_color = split
        k, los_x, left_front, right_front = _split_and_los(
            [core[t] for t in tids], gap_lo, gap_hi
        )
        if not (gap_lo <= los_x <= gap_hi):
            # No scoreable geometric boundary inside the color gap.
            los_x = (gap_lo + gap_hi) / 2.0
            k = sum(1 for v in xs if v < los_x)
            left_front = max((v for v in xs if v < los_x), default=los_x)
            right_front = min((v for v in xs if v >= los_x), default=los_x)
    else:
        left_color = None
        k, los_x, left_front, right_front = _split_and_los(
            [core[t] for t in tids]
        )

    # Confident color on a pre-snap track parked well past the LOS on its
    # color's WRONG side is sideline staff in team colors (coaches, subs) —
    # drop it from the formation and make no claims about it at all.
    conflicted: set[int] = set()
    if left_color is not None:
        right_color = "B" if left_color == "A" else "A"
        for t in list(core):
            c = _confident_color(colors, t)
            if c is None:
                continue
            x = core[t][0]
            side_color = left_color if x < los_x else right_color
            wrong_depth = abs(x - los_x)
            if (c != side_color and wrong_depth > CONFLICT_DEPTH
                    and min(positions[t]) <= snap):
                conflicted.add(t)
                del core[t]
        tids = sorted(core, key=lambda t: core[t][0])
        xs = [core[t][0] for t in tids]
        k = sum(1 for v in xs if v < los_x)

    left_ids, right_ids = tids[:k], tids[k:]
    left_xs, right_xs = xs[:k], xs[k:]

    # Offense = the side with the larger fraction of bodies on its front.
    # A human-marked direction.json (dashboard "swap O/D") beats the heuristic,
    # same trust model as snap.json.
    human_dir: str | None = None
    dir_path = pdir / "direction.json"
    if dir_path.exists():
        try:
            v = _load_json(dir_path).get("offense_direction")
            human_dir = v if v in ("+x", "-x") else None
        except (json.JSONDecodeError, AttributeError):
            human_dir = None

    # ---- offense-vs-defense ROLE: which side has the ball. Multiple football
    # signals are corroborated into a per-side offense-likeness score (7-on-the-
    # line, a compact central backfield, a contiguous OL block; wide-deep
    # corners pull toward defense) and the higher-scoring side is the offense.
    # This replaces the single fragile front-crowding heuristic that flipped the
    # role backwards when an auto-snap fired mid-play (sample-053).
    #
    # SNAP ROBUSTNESS: the signals are read from the CLEANEST formation frame in
    # the pre-snap window (fewest bodies streaming downfield, tightest fronts)
    # rather than one possibly-mid-play snap frame. The window ends at the snap,
    # so a human snap.json is still honoured exactly; a per-side FRONT for the
    # clean frame is re-derived from that frame's own widest interior x-gap.
    left_pts = [core[t] for t in left_ids]
    right_pts = [core[t] for t in right_ids]
    lfr, rfr = left_front, right_front
    # Direction hint for the clean-frame stray count: the tentative offense side
    # (higher front fraction). Only used to score frame cleanliness; the final
    # role is decided by the scores below.
    _lf0 = _front_fraction(left_xs, left_front, right_front)
    _rf0 = _front_fraction(right_xs, right_front, left_front)
    hint_sign = 1.0 if _lf0 >= _rf0 else -1.0
    clean = _clean_od_frame(positions, set(core), los_x, hint_sign, snap)
    if clean:
        # Keep each body on its ORIGINAL (color-informed) side; the clean frame
        # only refines each body's x within that side and re-derives the fronts.
        cleft = {t: clean[t] for t in left_ids if t in clean}
        cright = {t: clean[t] for t in right_ids if t in clean}
        if len(cleft) >= 4 and len(cright) >= 4:
            left_pts = list(cleft.values())
            right_pts = list(cright.values())
            lfr = max(x for x, _ in left_pts)
            rfr = min(x for x, _ in right_pts)

    # Formation lateral center = the y where the two interior FRONTS meet at the
    # ball. Offense and defense linemen line up nose-to-nose, so the true center
    # is the y-band holding on-line bodies from BOTH sides. Taking it from the
    # OVERLAP of the two fronts rejects a one-sided sideline knot: a bunch of
    # bodies crammed at a sideline (sample-053's dark knot at y~5) has no
    # opposing on-line body near it, so it never anchors the center; the real
    # front, where both teams stack, does. Falls back to the near-line median,
    # then the all-bodies median, when the overlap is too thin to trust.
    l_on = sorted(y for x, y in left_pts if lfr - x <= FRONT_BAND)
    r_on = sorted(y for x, y in right_pts if x - rfr <= FRONT_BAND)
    overlap = [
        y for y in l_on
        if any(abs(y - ry) <= LINE_MAX_DY for ry in r_on)
    ] + [
        y for y in r_on
        if any(abs(y - ly) <= LINE_MAX_DY for ly in l_on)
    ]
    if len(overlap) >= 3:
        center_y = _median(overlap)
    elif len(l_on) + len(r_on) >= 3:
        center_y = _median(l_on + r_on)
    else:
        center_y = _median([y for _, y in left_pts + right_pts])

    ol = _offense_likeness(left_pts, lfr, 1.0, center_y)
    orr = _offense_likeness(right_pts, rfr, -1.0, center_y)
    # Decisive 7-on-the-line lead overrides everything short of a human dir.
    ll = _line_count(left_pts, lfr, center_y)
    rl = _line_count(right_pts, rfr, center_y)
    # The combined score is only trusted when at least one side shows a REAL
    # front (>= LINE_PRESENT bodies on the line, or a contiguous OL block): a
    # scrambled frame where neither side has a recognizable front carries no
    # reliable O/D structure, so it defers to the legacy ordering the old
    # heuristic used (never regress a play that read fine on side-of-LOS clutter,
    # sample-048's sparse mostly-one-color mess).
    front_present = (
        max(ll, rl) >= LINE_PRESENT
        or max(_ol_contiguity(left_pts, lfr, center_y),
               _ol_contiguity(right_pts, rfr, center_y)) >= CONTIG_MIN + 1
    )
    wl = _wide_deep_count(left_pts, lfr, 1.0)
    wr = _wide_deep_count(right_pts, rfr, -1.0)
    # BENCH-CORRUPTED front: a big single-color sideline bench (of one team's
    # subs) was stripped above, but its team's ON-FIELD front is still poorly
    # sampled at the snap — its offensive line under-tracked while the compact
    # opponent's DEFENSIVE line stacks tight at the ball. The 7-on-line / front
    # count then reads the OPPONENT (the defense) as the offense (sample-077:
    # the dark offense's OL is a handful of at-snap tracks, the white 4-3 front
    # is dense, so front-count wrongly calls white the offense). Right here the
    # rule-based front tell is the thing the bench corrupts, so we fall to the
    # WIDE-DEEP corner tell instead — the secondary posts wide bodies yards off
    # its front only on DEFENSE, a structure the bench cannot fake. The side
    # with MORE wide-deep corners is the defense; the other is the offense.
    bench_dir: str | None = None
    if bench and (ll >= LINE_PRESENT or rl >= LINE_PRESENT) and wl != wr:
        bench_dir = "-x" if wl > wr else "+x"
    if bench_dir is not None:
        heuristic_dir = bench_dir
    elif abs(ll - rl) >= LINE_MARGIN + 1 and (ll >= 6 or rl >= 6):
        heuristic_dir = "+x" if ll > rl else "-x"
    elif front_present and abs(ol - orr) >= OD_DEADBAND:
        heuristic_dir = "+x" if ol > orr else "-x"
    else:
        # Near-tie / no clear front: defer to the legacy wide-deep / front-
        # fraction ordering so a play the old heuristic already got right is
        # never disturbed.
        if wl != wr:
            heuristic_dir = "-x" if wl > wr else "+x"
        else:
            lf = _front_fraction([x for x, _ in left_pts], lfr, rfr)
            rf = _front_fraction([x for x, _ in right_pts], rfr, lfr)
            heuristic_dir = "+x" if lf >= rf else "-x"
    if (human_dir or heuristic_dir) == "+x":
        o_ids, d_ids = left_ids, right_ids
        offense_direction, dir_sign = "+x", 1.0
    else:
        o_ids, d_ids = right_ids, left_ids
        offense_direction, dir_sign = "-x", -1.0
    if human_dir is not None:
        labels["direction_source"] = "human"

    # ---- O/D confidence: how sure is the offense-vs-defense (which side has the
    # ball) call? A normalized margin between the two sides' offense-likeness
    # scores, squashed to (0, 1]: a runaway front (7-on-line + backfield vs a
    # spread secondary) scores far apart -> high; a near-tie (the dashboard's
    # one-click-confirm case) scores close -> low. A human direction.json is an
    # explicit human decision, so its confidence is 1.0 and its source "human"
    # (the dashboard shows it as confirmed, not a guess). A bench-corrupted
    # front (the wide-deep fallback fired) is inherently shakier, so its auto
    # confidence is capped low to surface it for confirmation.
    if human_dir is not None:
        od_confidence = 1.0
        labels["od_source"] = "human"
    else:
        margin = abs(ol - orr) / (abs(ol) + abs(orr) + OD_CONF_SCALE)
        od_confidence = max(0.05, min(1.0, margin))
        if bench_dir is not None:
            od_confidence = min(od_confidence, OD_CONF_BENCH_CAP)
        labels["od_source"] = "auto"
    labels["od_confidence"] = round(od_confidence, 3)

    labels["los_x"] = round(los_x, 3)
    labels["offense_direction"] = offense_direction

    # ---- teams for every track (fragments included). Jersey color is the
    # primary signal (a fragment tracked only downfield still wears its
    # team's color); side of the LOS covers color-unknown tracks. No claims
    # for officials, conflicted sideline staff, or bodies off the field and
    # far from the formation (spectators beyond the sideline).
    o_color = None
    if left_color is not None:
        o_color = left_color if dir_sign > 0 else (
            "B" if left_color == "A" else "A"
        )

    # Jersey color is the PRIMARY team signal. When the two clusters separate
    # cleanly (o_color is set), the O-vs-D decision is made ONCE per color:
    # a body that belongs to cluster A or B inherits that color's team, so all
    # A bodies get one tag and all B the other. The only question is HOW clearly
    # a body must belong to its cluster before color overrides geometry — the
    # per-body confidence bar:
    #
    #   * STRICT (COLOR_CONF) when the LOS reads clean relative to color — the
    #     offense color's CONFIDENT bodies sit predominantly on the offense
    #     side. Geometry is trustworthy here, so a 0.6-0.8 body whose (possibly
    #     mis-clustered) color disagrees with its side defers to geometry
    #     (a mis-colored D-lineman polling the offense's color stays D).
    #   * RELAXED (TEAM_COLOR_CONF) when the LOS is MIS-READ relative to color —
    #     the offense color's confident bodies land mostly on the DEFENSE side
    #     (a mid-play snap / heavy clutter split). Geometry cannot be trusted,
    #     so every clear cluster member follows its color and "everyone is
    #     defense" becomes structurally impossible (play 049: low-confidence
    #     dark bodies stranded on the wrong geometric side follow their color).
    #
    # Refs stay excluded at both bars. Color-unknown / ref-band bodies always
    # fall back to side-of-LOS. When colors are NOT trustworthy at all
    # (o_color is None) the whole split is the current geometric one, unchanged
    # (synthetic e2e / no team_colors.json).
    #
    # NEUTRAL-ZONE OVERRIDE (independent of los_clean). Exactly ON the line
    # (|depth| <= NEAR_LOS_BAND) side-of-LOS is the WORST team signal: an OL and
    # a DL are inches apart across the ball, so a late auto-snap / LOS error
    # flips them (sample-080: penetrated dark D-linemen stamped OFFENSE). There
    # jersey color decides at the RELAXED NEAR_LOS_COLOR_CONF bar even when the
    # los_clean gate would otherwise trust geometry — because right at the ball
    # geometry is never trustworthy. Only genuinely color-unknown on-line bodies
    # (below that bar) fall back to side-of-LOS. This runs before the far-field
    # los_clean logic and never touches far-from-LOS bodies.
    los_clean = True
    if o_color is not None:
        off_side = def_side = 0
        for tid, (x, _) in align.items():
            if tid in bench:
                continue
            if _confident_color(colors, tid) == o_color:
                if (x - los_x) * dir_sign < 0:
                    off_side += 1
                else:
                    def_side += 1
        los_clean = off_side >= def_side

    teams: dict[str, str] = {}
    for tid, (x, y) in align.items():
        if tid in refs_hard or tid in conflicted:
            continue
        if not (TEAM_Y_MIN <= y <= TEAM_Y_MAX) and abs(x - los_x) > TEAM_X_SLACK:
            continue
        depth = (x - los_x) * dir_sign  # + = defense side, - = offense side
        # Neutral-zone bind fires for a body sitting ON the OFFENSE side of the
        # line (a defender that penetrated across a late-snap / mis-read LOS) —
        # the one case where side-of-LOS is actively wrong for a real player.
        # A body on the DEFENSE side stays with geometry: an OFFENSIVE lineman
        # essentially never lines up across the ball, so a defense-side body
        # polling the offense color at moderate confidence is a muddy-jersey
        # D-lineman, and geometry (D) is the reliable read (sample-005: a real
        # dl_es polled the offense color at 0.72 but sat 0.8 yд on its own side).
        near_los = -NEAR_LOS_BAND <= depth < 0.0
        if o_color is None:
            c = None
        elif near_los:
            # On the line, offense side: bind to color at the relaxed bar,
            # ignoring los_clean (geometry is least reliable here).
            c = _near_los_color(colors, tid)
        elif los_clean:
            c = _confident_color(colors, tid)
        else:
            c = _cluster_color(colors, tid)
        if c is not None:
            teams[str(tid)] = "O" if c == o_color else "D"
        else:
            teams[str(tid)] = "O" if (x - los_x) * dir_sign < 0 else "D"
    labels["teams"] = teams

    # Formation groups: the color-informed teams, not the raw x-split — a
    # confident defender projecting slightly across the LOS still fits with
    # the defense.
    o_ids = [t for t in tids if teams.get(str(t)) == "O"]
    d_ids = [t for t in tids if teams.get(str(t)) == "D"]
    if len(o_ids) != 11 or len(d_ids) != 11:
        base_conf *= 0.85

    # Defense's jersey cluster: the opposite of the offense's when the color
    # split held; else the confident-color majority of the core defenders.
    # Feeds the soft color cost in the role assignment and the fallback gates.
    d_color: str | None = None
    if o_color is not None:
        d_color = "B" if o_color == "A" else "A"
    else:
        d_votes = [c for t in d_ids if (c := _confident_color(colors, t))]
        a_n = sum(1 for c in d_votes if c == "A")
        if a_n >= len(d_votes) - a_n + 2:
            d_color = "A"
        elif len(d_votes) - a_n >= a_n + 2:
            d_color = "B"

    # ---- unified participant filter: "is this body playing in THIS snap?"
    # Computed once over field yards with the settled LOS + direction, then
    # applied BEFORE the formation fit and the defensive fallback so nothing
    # that names a defender (the Hungarian DL solve, the order-preserving
    # back-seven DP, fallback inheritance) ever sees a sideline coach / sub /
    # spectator and steals a system role for one (the coach-reported "saf_w"
    # that was a motionless body isolated near the far goal line). Team
    # membership above is untouched — a non-participant simply never earns a
    # defensive POSITION. Signals + combination live in _participants().
    participants = _participants(align, positions, los_x, dir_sign, snap, fps)
    # Pixel-space outliers never earn a defensive position — but only override
    # field space for SHALLOW-projecting bodies (the danger case: a sideline
    # coach whose homography projection collapses into the formation, which
    # field space cannot reject). A pixel outlier projecting DEEP is already an
    # obvious field-space outlier the fit prunes, so leaving it to field space
    # avoids perturbing a good defensive fit.
    pix_reject_shallow = {
        tid for tid in pix_reject
        if tid in align
        and (align[tid][0] - los_x) * dir_sign <= PIX_SHALLOW_DEPTH
    }
    participants -= pix_reject_shallow

    o_pos = {t: core[t] for t in o_ids}
    d_pos = {t: core[t] for t in d_ids if t in participants}
    pos_map, conf_map, anchors = _label_defense(
        d_pos, o_pos, los_x, dir_sign, base_conf, colors, d_color
    )

    # ---- fallback: non-core defenders (fragments, officials) inherit the
    # nearest fitted defender's / virtual slot's role. Gated hard (goal-line
    # play 093 exposed the holes): (1) a jersey leaning the offense's color
    # never inherits a defensive name; (2) bodies clearly BEHIND the entire
    # fitted defense (end-zone officials, subs past the end line when the
    # defense is backed up) don't inherit; (3) never past the end lines;
    # (4) an at-snap body wearing the fit's own sideline-official signature
    # (on the LOS at the boundary) doesn't inherit — the fit already ruled
    # it out, and re-claiming it duplicates a role held at the snap.
    # Time-separated fragments (the same player re-tracked later in the
    # play) still inherit freely — that duplication is intended.
    core_depths = [
        abs(align[int(t)][0] - los_x)
        for t in pos_map
        if conf_map.get(t, 0.0) >= 0.5 and int(t) in align
    ]
    max_core_depth = max(core_depths) if core_depths else POSITION_RANGE

    def _at_snap(tid: int) -> bool:
        fr = positions.get(tid)
        return (bool(fr) and min(fr) <= snap + SNAP_SIM_WIN
                and max(fr) >= snap - SNAP_SIM_WIN)

    # A fragment standing NEXT TO a fitted defender whose track is alive at
    # the same time is a DIFFERENT body stacked beside him — it must not
    # inherit his name (play 006: the saf_w's re-track stood 0.85 yd from
    # the fitted cb_w, whose own track spanned the fragment's entire life,
    # and took "cb_w"; the next-nearest anchor was his own saf_w). Spans
    # sharing >= SNAP_SIM_WIN frames count as simultaneous; the same player
    # re-tracked later shares no span and still inherits freely. The block
    # only reaches BLOCK_NEAR yards: a far-away fragment is not confusing
    # itself with anyone (play 079: the real cb_s — excluded from the fit —
    # must keep inheriting cb_s from 12 yd away even though a wrong body
    # holds it).
    role_block: list[tuple[str, int, int, float, float]] = []
    for t_s, role in pos_map.items():
        fr = positions.get(int(t_s))
        if fr and conf_map.get(t_s, 0.0) >= 0.25 and int(t_s) in align:
            hx, hy = align[int(t_s)]
            role_block.append((role, min(fr), max(fr), hx, hy))

    for tid_s, team in teams.items():
        if team != "D" or tid_s in pos_map:
            continue
        if int(tid_s) not in participants:
            continue  # non-participant (sideline coach/sub/spectator): a real
            # player never earns a defensive position via inheritance
        x, y = align[int(tid_s)]
        if abs(x - los_x) > POSITION_RANGE or not (
            -6.0 <= y <= 60.0 and 0.0 <= x <= 120.0
        ):
            continue
        if abs(x - los_x) > max_core_depth + FALLBACK_DEPTH_SLACK:
            continue  # behind the whole fitted defense = clutter
        if _color_lean(colors, int(tid_s), d_color) > 0.0:
            continue  # jersey leans the offense's color
        if (_at_snap(int(tid_s))
                and (x - los_x) * dir_sign <= SIDELINE_OFFICIAL_DEPTH
                and (y <= SIDELINE_Y or y >= 53.33 - SIDELINE_Y)):
            continue  # sideline official at the snap (fit pruned it too)
        fr = positions.get(int(tid_s))
        taken: set[str] = set()
        if fr:
            lo, hi = min(fr), max(fr)
            taken = {
                role for role, rlo, rhi, hx, hy in role_block
                if min(hi, rhi) - max(lo, rlo) + 1 >= SNAP_SIM_WIN
                and math.hypot(hx - x, hy - y) <= BLOCK_NEAR
            }
        name = _nearest_anchor(anchors, x, y, taken)
        if name:
            pos_map[tid_s] = name
            conf_map[tid_s] = 0.3

    # ---- human play-type override: on special teams (kicks/punts) there is
    # no scrimmage formation — emit no team/position claims; the dashboard
    # then colors purely by jersey cluster, the only reliable signal there.
    pt_path = pdir / "play_type.json"
    if pt_path.exists():
        try:
            ptype = _load_json(pt_path).get("type")
        except json.JSONDecodeError:
            ptype = None
        if ptype == "special":
            labels["play_type"] = "special"
            labels["teams"] = {}
            labels["positions"] = {}
            labels["position_confidence"] = {}
            _write(pdir, labels)
            return

    # ---- whose defense is this? The system names (lb_a/lb_b/lb_c/saf_w…)
    # describe OUR defense only (coach-confirmed: the opponent runs their own
    # system). When the defense is the opponent's, emit generic position
    # groups instead.
    us = None
    settings_path = Path(data_dir) / "settings.json"
    if settings_path.exists():
        try:
            us = _load_json(settings_path).get("us_color")
        except json.JSONDecodeError:
            us = None
    if us in ("dark", "light"):
        us_cluster = "A" if us == "dark" else "B"
        if o_color is not None:
            labels["defense_is_us"] = o_color != us_cluster
        else:
            d_clusters = [
                c for t in pos_map
                if (c := _confident_color(colors, int(t))) is not None
            ]
            if len(d_clusters) >= 4:
                ours = sum(1 for c in d_clusters if c == us_cluster)
                labels["defense_is_us"] = ours >= len(d_clusters) / 2
        if labels.get("defense_is_us") is False:
            generic = {
                "dl_es": "DL", "dl_ew": "DL", "T": "DL", "N": "DL",
                "lb_a": "LB", "lb_b": "LB", "lb_c": "LB",
                "cb_s": "CB", "cb_w": "CB", "F": "S", "saf_w": "S",
            }
            pos_map = {t: generic.get(n, n) for t, n in pos_map.items()}

    labels["positions"] = pos_map
    labels["position_confidence"] = {k: round(v, 3) for k, v in conf_map.items()}
    _write(pdir, labels)

    # football-cv generic output: collapse the defensive slot ids to standard
    # groups, classify the offense generically, and write positions.json.
    _generic = {
        "dl_es": "DL", "dl_ew": "DL", "T": "DL", "N": "DL",
        "lb_a": "LB", "lb_b": "LB", "lb_c": "LB",
        "cb_s": "CB", "cb_w": "CB", "F": "S", "saf_w": "S",
    }
    out = {str(t): _generic.get(n, n) for t, n in pos_map.items()}
    try:
        teams = labels.get("teams", {})
        off_ids = [t for t in core if teams.get(str(t)) == "O"]
        for t, tok in _classify_offense(off_ids, core, los_x).items():
            out[str(t)] = tok
    except (NameError, KeyError, TypeError):
        pass
    (pdir / "positions.json").write_text(json.dumps(out))


def _write(pdir: Path, labels: dict) -> None:
    with open(pdir / "labels.json", "w") as f:
        json.dump(labels, f, indent=1)
