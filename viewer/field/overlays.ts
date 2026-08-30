// Neutral overlay-marker types the field renderer draws on top of the board:
// ghost teammates, run-fit targets, zone hints, and post-answer truth.
import type { FieldPoint } from "./types";
import type { ZoneId } from "./model";

/** A target id, e.g. "FORCE_L", "ALLEY_R", "A_L". Opaque string. */
export type RunTargetId = string;

export type GhostMarker = { slot: number; label: string; spot: FieldPoint };

export type RunTargetMarker = { id: RunTargetId; center: FieldPoint };

export type ZoneHint = { zone: ZoneId; center: FieldPoint; tolYd: number };

export type TruthOverlay =
  | { kind: "SPOT"; truth: FieldPoint; yours: FieldPoint | null; distanceLabel: string | null }
  | { kind: "RUN_TARGET"; correct: RunTargetId; yours: RunTargetId | null }
  | { kind: "ZONE"; landmark: FieldPoint; tolYd: number; yours: FieldPoint | null }
  | { kind: "KEY"; correctKey: string; yoursKey: string | null };
