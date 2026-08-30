// Generic placement of defensive markers against an offensive look. Each
// marker resolves to a field spot from its generic position band, plus an
// optional keyed receiver number. No named-position logic.
import type { Assignment, Call, Formation, FormationPlayer } from "./model";
import { DL_DEPTH, DEFAULT_LB_DEPTH, DEFAULT_SAFETY_DEPTH } from "./constants";

export type ResolvedAlignment = {
  x: number;
  y: number;
  /** True when the spot is a band default, not an exact placement. */
  approximate: boolean;
};

export type Placement = {
  assignment: Assignment;
  resolved: ResolvedAlignment | null;
  /** The offensive player this marker keys (highlighted on render). */
  keyPlayer: FormationPlayer | null;
};

/** Depth (yards off LOS) for each generic defensive position band. */
function bandDepth(label: string): number {
  switch (label) {
    case "DL":
    case "CB":
      return DL_DEPTH;
    case "LB":
      return DEFAULT_LB_DEPTH;
    case "S":
      return DEFAULT_SAFETY_DEPTH;
    default:
      return DEFAULT_LB_DEPTH;
  }
}

/**
 * Resolve an optional numbered key ("#1"/"#2"/"#3") to a receiver, preferring
 * the side indicated by sideSign. Named-role keys are not supported here.
 */
export function resolveKeyPlayer(
  key: string | null,
  formation: Formation,
  sideSign: 1 | -1
): FormationPlayer | null {
  if (!key) return null;
  const m = /^#([123])$/.exec(key);
  if (!m) return null;
  const num = Number(m[1]);
  const candidates = formation.players.filter((p) => p.receiverNumber === num);
  return candidates.find((p) => Math.sign(p.x) === sideSign) ?? candidates[0] ?? null;
}

/**
 * Place every marker in a call against its formation using generic position
 * bands. The lateral spot spreads markers across the front; the depth comes
 * from the marker's generic label band.
 */
export function placeDefense(call: Call, formation: Formation): Placement[] {
  const n = call.assignments.length;
  return call.assignments.map((assignment, i) => {
    const spread = n > 1 ? (i / (n - 1) - 0.5) * 16 : 0;
    const resolved: ResolvedAlignment = {
      x: spread,
      y: bandDepth(assignment.label),
      approximate: true,
    };
    return { assignment, resolved, keyPlayer: null };
  });
}
