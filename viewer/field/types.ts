// Generic field-geometry types shared by the renderer + screen mapping.
export type FieldPoint = { x: number; y: number };
export type Crop = { xMin: number; xMax: number; yMin: number; yMax: number };

/** Quantization step (yards) — judging and rendering share it. */
export const QUANT_YD = 0.25;
