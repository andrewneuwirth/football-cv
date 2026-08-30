import type { AlignmentSpec, Formation, FormationPlayer } from "./model";
import {
  APEX_DEPTH,
  DEFAULT_LB_DEPTH,
  DEFAULT_SAFETY_DEPTH,
  DL_DEPTH,
  INSIDE_SHADE_X,
  OLINE_X,
  TECHNIQUE_X,
} from "./constants";

export type ResolvedAlignment = {
  x: number;
  y: number;
  /**
   * Approximate placements render with a dashed outline and the raw string
   * beneath the board. Approximate and honest beats confidently wrong.
   */
  approximate: boolean;
};

/**
 * Resolve an AlignmentSpec to field coordinates against a formation.
 * Returns null when nothing can be placed — the caller falls back to
 * showing `raw` as text. Text always works.
 *
 * The third parameter is the computed side sign; +1 = the field's left.
 *
 * Deliberately NO width/depth disambiguation heuristics: observed data
 * mixes "1x6" and "8x1" orderings. We render a best guess and mark it
 * approximate rather than trying to out-guess the input.
 */
export function resolveAlignment(
  spec: AlignmentSpec,
  formation: Formation,
  sideSign: 1 | -1
): ResolvedAlignment | null {
  switch (spec.anchor) {
    case "OLINE":
      return resolveOline(spec, sideSign);
    case "RECEIVER":
      return resolveReceiver(spec, formation, sideSign);
    case "LANDMARK":
      return resolveLandmark(spec);
    default:
      return null;
  }
}

function resolveOline(spec: AlignmentSpec, sideSign: 1 | -1): ResolvedAlignment | null {
  if (spec.technique === undefined) return null;
  const tech = spec.technique;
  // Off-ball levels use the tens convention: 10/30/50 share the lateral
  // landmark of 1/3/5 but play off the ball with depth given separately.
  const offBall = tech >= 10;
  const laneTech = offBall ? tech / 10 : tech;
  let x: number | undefined;
  if (spec.leverage === "INSIDE" && INSIDE_SHADE_X[laneTech] !== undefined) {
    x = INSIDE_SHADE_X[laneTech];
  } else {
    x = TECHNIQUE_X[laneTech];
  }
  if (x === undefined) return null;
  const y = offBall
    ? spec.depthYards ?? DEFAULT_LB_DEPTH
    : spec.depthYards ?? DL_DEPTH;
  return {
    x: laneTech === 0 ? 0 : x * sideSign,
    y,
    approximate: false,
  };
}

function findAnchorPlayer(
  spec: AlignmentSpec,
  formation: Formation,
  sideSign: 1 | -1
): FormationPlayer | null {
  const ap = spec.anchorPlayer;
  if (!ap) return null;
  if (ap === "TE") {
    return formation.players.find((p) => p.role === "TE") ?? null;
  }
  const m = ap.match(/^#([123])$/);
  if (m) {
    const num = Number(m[1]) as 1 | 2 | 3;
    const candidates = formation.players.filter(
      (p) => p.receiverNumber === num
    );
    if (candidates.length === 0) return null;
    // Prefer the receiver on the computed side; fall back to any.
    return (
      candidates.find((p) => Math.sign(p.x) === sideSign) ?? candidates[0]
    );
  }
  return null;
}

function resolveReceiver(
  spec: AlignmentSpec,
  formation: Formation,
  sideSign: 1 | -1
): ResolvedAlignment | null {
  const anchor = findAnchorPlayer(spec, formation, sideSign);
  if (!anchor) {
    // Bare "apex" rows anchor to structure, not a specific receiver:
    // halfway between the end of the line and the nearest slot/edge target.
    if (spec.landmark === "APEX") return resolveBareApex(spec, formation, sideSign);
    return null;
  }
  const side = (Math.sign(anchor.x) || sideSign) as 1 | -1;

  if (spec.landmark === "APEX") {
    // Halfway between the anchor receiver and the end of the line.
    const eol =
      (formation.players.some((p) => p.role === "TE" && Math.sign(p.x) === side)
        ? OLINE_X.TE
        : OLINE_X.T) * side;
    return { x: (anchor.x + eol) / 2, y: spec.depthYards ?? APEX_DEPTH, approximate: false };
  }

  const depth = spec.depthYards;
  const width = spec.widthYards ?? 0;
  if (depth === undefined) return null;
  // Leverage shifts toward (INSIDE) or away from (OUTSIDE) the ball.
  const towardBall = -side;
  let dx = 0;
  if (spec.leverage === "INSIDE") dx = width * towardBall;
  else if (spec.leverage === "OUTSIDE") dx = width * -towardBall;
  // Width-by-depth ordering is not consistent in the source cards ("1x6" vs
  // "8x1") — this is a best guess, marked approximate on render.
  return { x: anchor.x + dx, y: depth, approximate: true };
}

function resolveBareApex(
  spec: AlignmentSpec,
  formation: Formation,
  sideSign: 1 | -1
): ResolvedAlignment | null {
  // Apex without an anchor player: split the end of the line and the
  // nearest receiver on that side; no receiver → hang off the edge.
  const eolAbs = formation.players.some(
    (p) => p.role === "TE" && Math.sign(p.x) === sideSign
  )
    ? OLINE_X.TE
    : OLINE_X.T;
  const receivers = formation.players.filter(
    (p) => p.receiverNumber !== undefined && Math.sign(p.x) === sideSign
  );
  const nearest = receivers.length
    ? receivers.reduce((a, b) => (Math.abs(a.x) < Math.abs(b.x) ? a : b))
    : null;
  const x = nearest
    ? (eolAbs * sideSign + nearest.x) / 2
    : (eolAbs + 2.5) * sideSign;
  return { x, y: spec.depthYards ?? APEX_DEPTH, approximate: nearest === null };
}

function resolveLandmark(spec: AlignmentSpec): ResolvedAlignment | null {
  switch (spec.landmark) {
    case "MOF":
      return { x: 0, y: spec.depthYards ?? DEFAULT_SAFETY_DEPTH, approximate: false };
    case "FIELD":
    case "BOUNDARY":
      // Field/boundary calls need hash info we don't model.
      return { x: 0, y: spec.depthYards ?? DEFAULT_SAFETY_DEPTH, approximate: true };
    default:
      return null;
  }
}
