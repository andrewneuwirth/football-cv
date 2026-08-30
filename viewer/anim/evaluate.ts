// evaluatePose — pure UI-thread function of (track, tMs): segment scan +
// easing lerp + facing lerp + flatten ramp. No React, no state, no clocks.
import type { CompiledSeg, EasingId, Pose, Track } from "./types";
import { CUT_FLATTEN_RAMP_MS, FACING_STEP_MS } from "./types";

function sampleEase(id: EasingId, u: number): number {
  "worklet";
  if (id === "linear") return u;
  if (id === "easeOut") return 1 - Math.pow(1 - u, 3);
  // easeInOut
  return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
}

/**
 * Position on a segment chain at tMs. Segments chain positionally, so holds
 * between actions are free.
 */
export function positionOnSegs(
  segs: CompiledSeg[],
  x0: number,
  y0: number,
  tMs: number
): { x: number; y: number } {
  "worklet";
  let x = x0;
  let y = y0;
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i];
    if (tMs <= s.t0) return { x, y };
    if (tMs >= s.t1) {
      x = s.x1;
      y = s.y1;
      continue;
    }
    const u = (tMs - s.t0) / (s.t1 - s.t0);
    const e = sampleEase(s.ease, u);
    return { x: s.x0 + (s.x1 - s.x0) * e, y: s.y0 + (s.y1 - s.y0) * e };
  }
  return { x, y };
}

/** Pose at tMs. t < 0 (snap hold) clamps to the start pose. */
export function evaluatePose(track: Track, tMs: number): Pose {
  "worklet";
  const t = tMs < 0 ? 0 : tMs;
  const p = positionOnSegs(track.segs, track.x0, track.y0, t);

  const f = track.facing;
  const n = f.length;
  let facingDeg = track.restFacingDeg;
  if (n === 1) {
    facingDeg = f[0];
  } else if (n > 1) {
    const pos = t / FACING_STEP_MS;
    if (pos <= 0) {
      facingDeg = f[0];
    } else if (pos >= n - 1) {
      facingDeg = f[n - 1];
    } else {
      const i = Math.floor(pos);
      const frac = pos - i;
      facingDeg = f[i] + (f[i + 1] - f[i]) * frac;
    }
  }

  let flatten = 0;
  if (track.flattenAtMs !== undefined) {
    const raw = (t - track.flattenAtMs) / CUT_FLATTEN_RAMP_MS;
    flatten = raw <= 0 ? 0 : raw >= 1 ? 1 : raw;
  }

  return { x: p.x, y: p.y, facingDeg, flatten };
}
