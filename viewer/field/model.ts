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

export type Assignment = {
  slot: number;
  label: PositionId;
  alignment: Alignment;
};

/** A defensive call: a set of assignments over an offensive look. */
export type Call = {
  assignments: Assignment[];
};
