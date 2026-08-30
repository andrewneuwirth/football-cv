// Neutral playback types for the field renderer's optional live tracks.
export type EasingId = "linear" | "easeOut" | "easeInOut";

export type CompiledSeg = {
  t0: number;
  t1: number; // absolute ms from snap
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  ease: EasingId;
  prim?: string;
};

export type Track = {
  actorId: string; // e.g. "RB", "ball", "ghost"
  x0: number;
  y0: number;
  segs: CompiledSeg[];
  endMs: number;
  facing: Float32Array; // sampled every FACING_STEP_MS
  restFacingDeg: number;
  flattenAtMs?: number;
  key: boolean;
};

export type CompiledLook = {
  mirrored: boolean;
  tracks: Track[];
  totalMs: number;
};

export type Pose = { x: number; y: number; facingDeg: number; flatten: number };

export type SpotlightSpec = { actors: string[] };

export const FACING_STEP_MS = 50;
/** Cut squash ramp length (ms). */
export const CUT_FLATTEN_RAMP_MS = 120;
