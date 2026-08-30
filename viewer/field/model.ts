// Neutral domain shapes for the field renderer. Generic positions only —
// no named-position assignment logic.
import type { FieldPoint } from "./types";

/** Generic position tokens (defense DL/LB/CB/S, offense OL/QB/RB/WR). */
export type PositionId =
  | "DL"
  | "LB"
  | "CB"
  | "S"
  | "OL"
  | "QB"
  | "RB"
  | "WR";

export type ZoneId =
  | "flat"
  | "curl"
  | "hook"
  | "deep_third"
  | "deep_half"
  | "deep_middle";

/** One offensive body at its field spot. */
export type FormationPlayer = {
  actorId: string;
  x: number;
  y: number;
  role: string;
  receiverNumber?: number;
};

export type Formation = {
  name: string;
  players: FormationPlayer[];
  /** +1 / -1 lateral sign used to resolve numbered receivers. */
  strengthSign: 1 | -1;
};

/** One defender's alignment description. */
export type Alignment = { raw: string };

/** Where a defender's alignment is measured from. */
export type AlignmentAnchor = "OLINE" | "RECEIVER" | "LANDMARK";

/**
 * A structured alignment: generic field geometry only (technique number,
 * depth/width in yards, leverage, and neutral landmarks). Generic only.
 */
export type AlignmentSpec = {
  raw: string;
  anchor: AlignmentAnchor;
  technique?: number; // 0,1,2,3,4,5,6,7,9 | 10,30,50 off-ball
  depthYards?: number;
  widthYards?: number;
  leverage?: "INSIDE" | "OUTSIDE" | "HEAD_UP";
  anchorPlayer?: string; // "TE", "#1", "#2", "#3"
  landmark?: "MOF" | "APEX" | "FIELD" | "BOUNDARY";
};

export type Assignment = {
  slot: number;
  label: PositionId;
  alignment: Alignment;
};

/** A defensive call: a set of assignments over an offensive look. */
export type Call = {
  assignments: Assignment[];
};
