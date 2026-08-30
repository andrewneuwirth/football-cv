// Field renderer constants + screen mapping — DESIGN §5, SPEC §8.
// All tunable numbers live here so they can be adjusted in one place after
// the coach looks at it. Coordinates: x lateral yards (+x = offense's right
// = defense's left = screen left, unmirrored), y yards off LOS (+y =
// defensive backfield).

import type { Crop, FieldPoint } from "./types";
import { QUANT_YD } from "./types";
import { darkTheme } from "../theme/colors";

/** Default crop: the box. Do NOT draw the full field width. */
export const CROP: Crop = {
  xMin: -14,
  xMax: 14,
  yMin: -4, // offensive backfield
  yMax: 15, // defensive backfield
};

/** HS hashes are 53'4" apart → ±8.9 yd. */
export const HASH_X = 8.9;

/** O-line spacing (yards from center). */
export const OLINE_X = {
  C: 0,
  G: 1.6,
  T: 3.4,
  TE: 5.4,
} as const;

/**
 * Technique → |x| offset in yards. Approximations — technique numbers are
 * landmarks relative to the line and real spacing varies with splits.
 */
export const TECHNIQUE_X: Record<number, number> = {
  0: 0.0,
  1: 0.8,
  2: 1.6,
  3: 2.2,
  4: 3.4,
  5: 4.0,
  6: 5.4,
  7: 4.9,
  9: 6.0,
};

/** Inside-shade ("2i", "4i") pull-in from the head-up technique x. */
export const INSIDE_SHADE_X: Record<number, number> = {
  2: 1.2,
  4: 2.9,
};

/** Default depth (yards off LOS) for on-the-line defenders. */
export const DL_DEPTH = 0.7;
/** Default LB depth when the card gives none. */
export const DEFAULT_LB_DEPTH = 4.5;
/** Default depth for apex players. */
export const APEX_DEPTH = 4.5;
/** Default deep-safety depth when the card gives none. */
export const DEFAULT_SAFETY_DEPTH = 12;

/**
 * Body sizing in yards (converted to px by the mapper's scale).
 * DESIGN §5.3: one-flat-edge rounded rect ≈ 1.6 × 1.1 yd.
 */
export const BODY = {
  width: 1.6,
  height: 1.1,
  cornerRadius: 0.32,
} as const;

/** Ball: small chalk-stroked ellipse at the LOS. */
export const BALL = {
  spot: { x: 0, y: -0.2 } as FieldPoint,
  rx: 0.3,
  ry: 0.18,
} as const;

/** Outline weights (px) — the fill grammar of DESIGN §5.3. */
export const STROKE = {
  offense: 1.5,
  defender: 1.5,
  key: 2,
  youRing: 2,
} as const;

/**
 * Board pigments never theme (DESIGN §5.1) — the board stays dark in light
 * mode, so its interactive colors are pinned to the dark-legible token set.
 * A saturated accent on a dark surface can wash out; these are tuned values.
 */
export const BOARD = {
  accent: darkTheme.accent, // other defenders, selection, zone tint
  actionFill: darkTheme.actionFill, // YOU — the only solid body
  correct: darkTheme.correct,
  wrong: darkTheme.wrong,
} as const;

// ————— Screen mapping —————

export type FieldMapper = {
  /** Field x (yd) → screen x (px). Mirror-aware. */
  sx: (xYd: number) => number;
  /** Field y (yd) → screen y (px). */
  sy: (yYd: number) => number;
  pxPerYd: number;
  width: number;
  height: number;
  /** Screen px → field yards (inverse of fromField; mirror-aware). */
  toField: (px: { x: number; y: number }) => FieldPoint;
  /** Field yards → screen px. */
  fromField: (p: FieldPoint) => { x: number; y: number };
};

/**
 * One mapper shared by canvas, gestures, and grading so taps and drawings
 * can never disagree — including under `mirrored` (view-space x flip only).
 */
export function makeMapper(crop: Crop, width: number, mirrored = false): FieldMapper {
  const cropW = crop.xMax - crop.xMin;
  const cropH = crop.yMax - crop.yMin;
  const pxPerYd = width / cropW;
  const height = cropH * pxPerYd;
  const xMin = crop.xMin;
  const xMax = crop.xMax;
  const yMin = crop.yMin;

  const sx = (xYd: number): number => {
    "worklet";
    return mirrored ? (xYd - xMin) * pxPerYd : (xMax - xYd) * pxPerYd;
  };
  const sy = (yYd: number): number => {
    "worklet";
    return (yYd - yMin) * pxPerYd;
  };
  const toField = (px: { x: number; y: number }): FieldPoint => {
    "worklet";
    return {
      x: mirrored ? xMin + px.x / pxPerYd : xMax - px.x / pxPerYd,
      y: yMin + px.y / pxPerYd,
    };
  };
  const fromField = (p: FieldPoint): { x: number; y: number } => {
    "worklet";
    return { x: sx(p.x), y: sy(p.y) };
  };

  return { sx, sy, pxPerYd, width, height, toField, fromField };
}

/** Board height in px for a crop at a width (layout before Skia mounts). */
export function boardHeight(crop: Crop, width: number): number {
  return ((crop.yMax - crop.yMin) / (crop.xMax - crop.xMin)) * width;
}

/** Quantize a field point to 0.25 yd — judging and rendering share it. */
export function quantizePoint(p: FieldPoint): FieldPoint {
  return {
    x: Math.round(p.x / QUANT_YD) * QUANT_YD,
    y: Math.round(p.y / QUANT_YD) * QUANT_YD,
  };
}

/** Clamp a field point inside a crop (drag can't leave the board). */
export function clampToCrop(p: FieldPoint, crop: Crop, marginYd = 0.25): FieldPoint {
  return {
    x: Math.min(crop.xMax - marginYd, Math.max(crop.xMin + marginYd, p.x)),
    y: Math.min(crop.yMax - marginYd, Math.max(crop.yMin + marginYd, p.y)),
  };
}
