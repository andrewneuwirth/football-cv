// The gesture layer over the board. One tap/drag decision implemented once:
// Gesture.Race(tap, pan), an 8dp threshold, and 0.25 yd quantization before
// anything leaves this file. Generic positions only — no named-role logic.
import React, { useMemo, useRef, useState } from "react";
import { Platform, StyleSheet, Text, View } from "react-native";
import type { ViewStyle } from "react-native";
import { Gesture, GestureDetector } from "react-native-gesture-handler";
import type { Call, Formation, FormationPlayer } from "./model";
import type { Crop, FieldPoint } from "./types";
import type {
  GhostMarker,
  RunTargetId,
  RunTargetMarker,
  TruthOverlay,
  ZoneHint,
} from "./overlays";
import { useTheme, chalk } from "../theme/colors";
import {
  BOARD,
  BODY,
  CROP,
  clampToCrop,
  makeMapper,
  quantizePoint,
} from "./constants";
import { FieldRenderer } from "./FieldRenderer";

export type InteractiveFieldProps = {
  formation: Formation;
  /** Draw the defense too (study board) — forwarded to the canvas. */
  call?: Call;
  /** The slot rendered as YOU (the only solid body) — forwarded to the canvas. */
  highlightSlot?: number;
  crop?: Crop;
  width: number;
  ghosts?: GhostMarker[];
  selfSpot?: FieldPoint | null;
  /** Controlled placement marker — parent owns it, we report field points. */
  marker: FieldPoint | null;
  onTap?: (p: FieldPoint) => void;
  onDragMove?: (p: FieldPoint) => void;
  onDragEnd?: (p: FieldPoint) => void;
  /** false during feedback — every gesture disabled. */
  interactive: boolean;
  truth?: TruthOverlay | null;
  /** Animated crop interpolation + minimap strip. */
  zoom?: { center: FieldPoint; widthYd: number } | null;
  zoneHints?: ZoneHint[];
  targets?: { marker: RunTargetMarker; selected: boolean }[];
  onTargetTap?: (id: RunTargetId) => void;
  /** Offense bodies become tappable, reporting a numbered receiver key. */
  offenseTappable?: boolean;
  onOffenseTap?: (key: string) => void;
  mirrored?: boolean;
  reducedMotion?: boolean;
  /**
   * Side sign for resolving numbered KEY truth ("#1"/"#2"/"#3"). Defaults to
   * formation.strengthSign.
   */
  keySideSign?: 1 | -1;
  children?: React.ReactNode;
};

const FINGER_OFFSET_PX = 40; // marker floats above the touch — finger never occludes it
const DRAG_GRAB_PX = 24; // drag starting this close to the marker moves it
const TAP_MAX_DIST = 8;
const OFFENSE_SLOP_PX = 44; // minimum effective target, nearest-center resolution
const MINIMAP_H = 48;

type HoverSpot =
  | { kind: "target"; id: string; x: number; y: number; r: number }
  | { kind: "offense"; id: string; x: number; y: number };

/** Nearest tap-target within its own radius, else null. Generic geometry. */
function resolveTargetTap(
  p: FieldPoint,
  markers: RunTargetMarker[]
): RunTargetId | null {
  let best: RunTargetId | null = null;
  let bestD = Infinity;
  for (const m of markers) {
    const d = Math.hypot(m.center.x - p.x, m.center.y - p.y);
    if (d < bestD) {
      bestD = d;
      best = m.id;
    }
  }
  return best && bestD <= 2.0 ? best : null;
}

export default function InteractiveField({
  formation,
  call,
  highlightSlot,
  crop: cropProp,
  width,
  ghosts,
  selfSpot,
  marker,
  onTap,
  onDragMove,
  onDragEnd,
  interactive,
  truth,
  zoom,
  zoneHints,
  targets,
  onTargetTap,
  offenseTappable,
  onOffenseTap,
  mirrored,
  reducedMotion,
  keySideSign,
  children,
}: InteractiveFieldProps) {
  const { theme } = useTheme();
  const crop = cropProp ?? CROP;
  const map = useMemo(
    () => makeMapper(crop, width, !!mirrored),
    [crop.xMin, crop.xMax, crop.yMin, crop.yMax, width, mirrored]
  );
  const height = map.height;

  // ————— zoom window (transform-based; the mapping stays put) —————
  const zt = useMemo(() => {
    if (!zoom || width <= 0) {
      return { k: 1, tx: 0, ty: 0, cx: 0, cy: 0, wYd: 0 };
    }
    const cropW = crop.xMax - crop.xMin;
    const cropH = crop.yMax - crop.yMin;
    const wYd = Math.min(zoom.widthYd, cropW);
    const hYd = wYd * (cropH / cropW);
    const cx = Math.min(crop.xMax - wYd / 2, Math.max(crop.xMin + wYd / 2, zoom.center.x));
    const cy = Math.min(crop.yMax - hYd / 2, Math.max(crop.yMin + hYd / 2, zoom.center.y));
    const k = cropW / wYd;
    const wpx = map.fromField({ x: cx, y: cy });
    return {
      k,
      tx: -k * (wpx.x - width / 2),
      ty: -k * (wpx.y - height / 2),
      cx,
      cy,
      wYd,
    };
  }, [zoom, crop.xMin, crop.xMax, crop.yMin, crop.yMax, map, width, height]);

  const zoomStyle: ViewStyle = {
    transform: [
      { translateX: zt.tx },
      { translateY: zt.ty },
      { scale: zt.k },
    ],
  };

  /** viewport px → base (unzoomed) px, using the settled zoom target. */
  const toBasePx = (vx: number, vy: number) => ({
    x: width / 2 + (vx - width / 2 - zt.tx) / zt.k,
    y: height / 2 + (vy - height / 2 - zt.ty) / zt.k,
  });
  /** base px → viewport px. */
  const toViewportPx = (bx: number, by: number) => ({
    x: width / 2 + zt.tx + zt.k * (bx - width / 2),
    y: height / 2 + zt.ty + zt.k * (by - height / 2),
  });

  const fieldFromViewport = (vx: number, vy: number): FieldPoint => {
    const b = toBasePx(vx, vy);
    return clampToCrop(quantizePoint(map.toField(b)), crop);
  };

  // ————— hit resolution —————
  const resolveOffense = (
    basePx: { x: number; y: number }
  ): { key: string; player: FormationPlayer; cx: number; cy: number } | null => {
    let best: { key: string; player: FormationPlayer; cx: number; cy: number } | null =
      null;
    let bestD = Infinity;
    for (const p of formation.players) {
      const key = receiverKeyFor(p);
      if (!key) continue;
      const c = map.fromField({ x: p.x, y: p.y });
      const d = Math.hypot(c.x - basePx.x, c.y - basePx.y);
      if (d < bestD) {
        bestD = d;
        best = { key, player: p, cx: c.x, cy: c.y };
      }
    }
    const slop = Math.max(OFFENSE_SLOP_PX, BODY.width * map.pxPerYd);
    return best && bestD <= slop ? best : null;
  };

  const handleTap = (vx: number, vy: number) => {
    if (!interactive) return;
    const basePx = toBasePx(vx, vy);
    if (offenseTappable && onOffenseTap) {
      const hit = resolveOffense(basePx);
      if (hit) onOffenseTap(hit.key);
      return;
    }
    const p = clampToCrop(quantizePoint(map.toField(basePx)), crop);
    if (targets && targets.length > 0 && onTargetTap) {
      const id = resolveTargetTap(p, targets.map((t) => t.marker));
      if (id) onTargetTap(id);
      return;
    }
    onTap?.(p);
  };

  // ————— gestures —————
  const [dragging, setDragging] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0 });
  const dragValidRef = useRef(false);
  const lastDragRef = useRef<FieldPoint | null>(null);

  const tap = Gesture.Tap()
    .enabled(interactive)
    .maxDistance(TAP_MAX_DIST)
    .maxDuration(10000)
    .runOnJS(true)
    .onEnd((e, success) => {
      if (success) handleTap(e.x, e.y);
    });

  const pan = Gesture.Pan()
    .enabled(interactive && !!marker && !!(onDragMove || onDragEnd))
    .minDistance(TAP_MAX_DIST)
    .runOnJS(true)
    .onBegin((e) => {
      panStartRef.current = { x: e.x, y: e.y };
    })
    .onStart(() => {
      if (!marker) return;
      const start = toBasePx(panStartRef.current.x, panStartRef.current.y);
      const mpx = map.fromField(marker);
      const grab = Math.hypot(start.x - mpx.x, start.y - mpx.y) <= DRAG_GRAB_PX + 8;
      dragValidRef.current = grab;
      if (grab) {
        lastDragRef.current = null;
        setDragging(true);
      }
    })
    .onUpdate((e) => {
      if (!dragValidRef.current || !onDragMove) return;
      const p = fieldFromViewport(e.x, e.y - FINGER_OFFSET_PX);
      const last = lastDragRef.current;
      if (!last || last.x !== p.x || last.y !== p.y) {
        lastDragRef.current = p;
        onDragMove(p);
      }
    })
    .onEnd((e) => {
      if (dragValidRef.current && onDragEnd) {
        onDragEnd(fieldFromViewport(e.x, e.y - FINGER_OFFSET_PX));
      }
    })
    .onFinalize(() => {
      dragValidRef.current = false;
      setDragging(false);
    });

  // Hover outline (web frame mode) — RNGH's hover gesture, 44pt-equivalent.
  const [hover, setHover] = useState<HoverSpot | null>(null);
  const hoverEnabled =
    Platform.OS === "web" && interactive && !!(targets?.length || offenseTappable);
  const hoverG = Gesture.Hover()
    .enabled(hoverEnabled)
    .runOnJS(true)
    .onUpdate((e) => {
      const basePx = toBasePx(e.x, e.y);
      let next: HoverSpot | null = null;
      if (offenseTappable) {
        const hit = resolveOffense(basePx);
        if (hit) {
          next = { kind: "offense", id: hit.player.actorId, x: hit.cx, y: hit.cy };
        }
      } else if (targets && targets.length > 0) {
        const p = quantizePoint(map.toField(basePx));
        const id = resolveTargetTap(p, targets.map((t) => t.marker));
        if (id) {
          const t = targets.find((x) => x.marker.id === id);
          if (t) {
            const c = map.fromField(t.marker.center);
            next = {
              kind: "target",
              id,
              x: c.x,
              y: c.y,
              r: 0.62 * map.pxPerYd,
            };
          }
        }
      }
      setHover((prev) =>
        prev?.id === next?.id && prev?.kind === next?.kind ? prev : next
      );
    })
    .onEnd(() => setHover(null))
    .onFinalize(() => setHover(null));

  const gesture = Gesture.Simultaneous(Gesture.Race(tap, pan), hoverG);

  // ————— crosshair while dragging (viewport space, crisp at any zoom) —————
  const crosshair = useMemo(() => {
    if (!dragging || !marker) return null;
    const mBase = map.fromField(marker);
    const mv = toViewportPx(mBase.x, mBase.y);
    const losV = toViewportPx(mBase.x, map.sy(0));
    const line = formation.players.filter(
      (p) => p.y > -1.5 && p.receiverNumber === undefined
    );
    const nearest = line.length
      ? line.reduce((a, b) =>
          Math.abs(a.x - marker.x) < Math.abs(b.x - marker.x) ? a : b
        )
      : null;
    const olV = nearest ? toViewportPx(map.sx(nearest.x), mBase.y) : null;
    const label = `${Math.abs(marker.x).toFixed(1)} yd from ball · ${Math.abs(
      marker.y
    ).toFixed(1)} off LOS`;
    return { mv, losV, olV, label };
  }, [dragging, marker, map, formation, zt]);

  const webCursor: ViewStyle | null =
    Platform.OS === "web" && interactive
      ? ({
          cursor: targets?.length || offenseTappable ? "pointer" : "crosshair",
        } as unknown as ViewStyle)
      : null;

  const bodyW = BODY.width * map.pxPerYd;
  const bodyH = BODY.height * map.pxPerYd;

  return (
    <View style={{ width }}>
      {zoom && (
        <Minimap
          width={width}
          crop={crop}
          formation={formation}
          marker={marker}
          winCx={zt.cx}
          winWYd={zt.wYd}
          mirrored={!!mirrored}
          dark={theme.dark}
        />
      )}
      <GestureDetector gesture={gesture}>
        <View
          collapsable={false}
          style={[{ width, height, overflow: "hidden" }, webCursor]}
        >
          <View style={[{ width, height }, zoomStyle]}>
            <FieldRenderer
              formation={formation}
              call={call}
              highlightSlot={highlightSlot}
              width={width}
              crop={crop}
              ghosts={ghosts}
              selfSpot={selfSpot}
              marker={marker}
              truth={truth}
              zoneHints={zoneHints}
              targets={targets}
              mirrored={mirrored}
              reducedMotion={reducedMotion}
              keySideSign={keySideSign}
            />
            {/* Hover outline — base space, scales with the board. */}
            {hover && hover.kind === "target" && (
              <View
                pointerEvents="none"
                style={{
                  position: "absolute",
                  left: hover.x - hover.r - 4,
                  top: hover.y - hover.r - 4,
                  width: (hover.r + 4) * 2,
                  height: (hover.r + 4) * 2,
                  borderRadius: hover.r + 4,
                  borderWidth: 1.5,
                  borderColor: BOARD.accent,
                }}
              />
            )}
            {hover && hover.kind === "offense" && (
              <View
                pointerEvents="none"
                style={{
                  position: "absolute",
                  left: hover.x - bodyW / 2 - 4,
                  top: hover.y - bodyH / 2 - 4,
                  width: bodyW + 8,
                  height: bodyH + 8,
                  borderRadius: 6,
                  borderWidth: 1.5,
                  borderColor: BOARD.accent,
                }}
              />
            )}
          </View>

          {/* Crosshair + live yardage — teaches the coordinate vocabulary. */}
          {crosshair && (
            <View pointerEvents="none" style={StyleSheet.absoluteFill}>
              <View
                style={{
                  position: "absolute",
                  left: crosshair.mv.x - 0.5,
                  top: Math.min(crosshair.mv.y, crosshair.losV.y),
                  width: 1,
                  height: Math.abs(crosshair.losV.y - crosshair.mv.y),
                  backgroundColor: chalk.los,
                }}
              />
              {crosshair.olV && (
                <View
                  style={{
                    position: "absolute",
                    top: crosshair.mv.y - 0.5,
                    left: Math.min(crosshair.mv.x, crosshair.olV.x),
                    width: Math.abs(crosshair.olV.x - crosshair.mv.x),
                    height: 1,
                    backgroundColor: chalk.los,
                  }}
                />
              )}
              <View
                style={{
                  position: "absolute",
                  left: Math.min(Math.max(4, crosshair.mv.x - 90), width - 184),
                  top: Math.max(4, crosshair.mv.y - 34),
                  width: 180,
                  alignItems: "center",
                }}
              >
                <Text
                  maxFontSizeMultiplier={1}
                  style={{
                    color: chalk.full,
                    fontSize: 12,
                    letterSpacing: 0.1,
                  }}
                >
                  {crosshair.label}
                </Text>
              </View>
            </View>
          )}
        </View>
      </GestureDetector>

      {/* Extra layers passed by the host — viewport space. */}
      {children ? (
        <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
          {children}
        </View>
      ) : null}
    </View>
  );
}

/**
 * Numbered-receiver key for a body ("#1"/"#2"/"#3"), or null. Only receivers
 * carry keys here — there is no named-role resolution.
 */
function receiverKeyFor(p: FormationPlayer): string | null {
  if (p.receiverNumber) return `#${p.receiverNumber}`;
  return null;
}

/**
 * Dimmed full-board context above the zoomed stage: the line, the marker, and
 * where the window sits. 48px tall, chalk grammar, nothing interactive.
 */
function Minimap({
  width,
  crop,
  formation,
  marker,
  winCx,
  winWYd,
  mirrored,
  dark,
}: {
  width: number;
  crop: Crop;
  formation: Formation;
  marker: FieldPoint | null;
  winCx: number;
  winWYd: number;
  mirrored: boolean;
  dark: boolean;
}) {
  const cropW = crop.xMax - crop.xMin;
  const cropH = crop.yMax - crop.yMin;
  const px = width / cropW;
  const sx = (x: number) => (mirrored ? (x - crop.xMin) * px : (crop.xMax - x) * px);
  const yTo = (y: number) => ((y - crop.yMin) / cropH) * MINIMAP_H;
  const left = Math.min(sx(winCx - winWYd / 2), sx(winCx + winWYd / 2));
  return (
    <View
      style={{
        width,
        height: MINIMAP_H,
        marginBottom: 4,
        backgroundColor: dark ? chalk.boardDark : chalk.boardLight,
        opacity: 0.75,
        overflow: "hidden",
      }}
    >
      <View
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: yTo(0) - 0.5,
          height: 1,
          backgroundColor: chalk.los,
        }}
      />
      {formation.players.map((p, i) => (
        <View
          key={`mm${i}`}
          style={{
            position: "absolute",
            left: sx(p.x) - 2,
            top: yTo(p.y) - 2,
            width: 4,
            height: 4,
            borderRadius: 1,
            backgroundColor: chalk.body,
          }}
        />
      ))}
      {marker && (
        <View
          style={{
            position: "absolute",
            left: sx(marker.x) - 3,
            top: yTo(marker.y) - 3,
            width: 6,
            height: 6,
            borderRadius: 3,
            backgroundColor: BOARD.actionFill,
            borderWidth: 1,
            borderColor: chalk.full,
          }}
        />
      )}
      <View
        style={{
          position: "absolute",
          left,
          top: 0,
          bottom: 0,
          width: winWYd * px,
          borderWidth: 1,
          borderColor: chalk.full,
          backgroundColor: "rgba(242,245,250,0.08)",
        }}
      />
    </View>
  );
}
