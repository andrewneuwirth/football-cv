// The board (DESIGN §5) — chalk surface that never themes, chalk grammar
// bodies, optional live playback driven per-frame off the React tree.
// "You're the solid one. Yellow is what you read."
import React, { useEffect, useMemo } from "react";
import { StyleSheet, Text, View } from "react-native";
import {
  Canvas,
  Circle,
  DashPathEffect,
  Group,
  Line,
  Oval,
  Path,
  Rect,
  Skia,
  vec,
} from "@shopify/react-native-skia";
import type { SkPath } from "@shopify/react-native-skia";
import Animated, {
  Easing,
  cancelAnimation,
  useAnimatedStyle,
  useDerivedValue,
  useSharedValue,
  withDelay,
  withSequence,
  withTiming,
} from "react-native-reanimated";
import type { SharedValue } from "react-native-reanimated";
import { evaluatePose } from "../anim/evaluate";
import type { CompiledLook, SpotlightSpec, Track } from "../anim/types";
import type {
  Call,
  Formation,
  FormationPlayer,
  PositionId,
  ZoneId,
} from "./model";
import type { Crop, FieldPoint } from "./types";
import type {
  GhostMarker,
  RunTargetMarker,
  TruthOverlay,
  ZoneHint,
} from "./overlays";
import { AppText, positionTagLabel } from "../components/ui";
import { useTheme, chalk, radius, space } from "../theme/colors";
import {
  BALL,
  BOARD,
  BODY,
  CROP,
  HASH_X,
  STROKE,
  makeMapper,
} from "./constants";
import type { FieldMapper } from "./constants";
import { placeDefense, resolveKeyPlayer } from "./placeDefense";

export type FieldProps = {
  formation: Formation;
  /** Optional — formations render alone; reps pass ghosts/selfSpot instead. */
  call?: Call;
  /** Slot rendered as YOU — the only solid body on the board. */
  highlightSlot?: number;
  width: number;
  crop?: Crop;
  /** Live playback: offense + ball driven per-frame from compiled tracks. */
  live?: CompiledLook;
  playTime?: SharedValue<number>;
  /**
   * True when RiveTokenLayer is mounted above this canvas (native): the
   * animated player BODIES are drawn by Rive, so LiveBody chips are skipped
   * here. Trails, ball, ghost, and every static/graded element stay Skia.
   */
  riveBodies?: boolean;
  /**
   * Scrub mode (F5): a thin progress track under the board reflecting
   * playTime/totalMs. Reuses the live path for the actors; this only adds the
   * bar. Off unless supplied — static + trainer live modes render unchanged.
   * Reanimated-driven off the SharedValue; no per-frame React state.
   */
  scrubTrack?: { playTime: SharedValue<number>; totalMs: number };
  /** Replay dimming: non-key actors to 25%, keys ringed in accent. */
  spotlight?: SpotlightSpec | null;
  /** View-space x flip only — mapping handles both directions. */
  mirrored?: boolean;
  /** PASS_DROP hint regions — faint tint, never the answer. */
  zoneHints?: ZoneHint[];
  /** RUN_FIT tap targets, drawn as chalk circles with gap letters. */
  targets?: { marker: RunTargetMarker; selected: boolean }[];
  /** Post-commit truth rendering — the board always shows the right answer. */
  truth?: TruthOverlay | null;
  /** Controlled placement marker — YOU treatment. */
  marker?: FieldPoint | null;
  /** Teammates at their true spots, ghosted. */
  ghosts?: GhostMarker[];
  /** Pre-placed YOU (RUN_FIT, READ). */
  selfSpot?: FieldPoint | null;
  /** Teach mode only: A/B/C/D micro-chalk between the linemen. */
  gapLetters?: boolean;
  reducedMotion?: boolean;
  /**
   * Side sign for resolving numbered KEY truth ("#1"/"#2"/"#3") — pass
   * sideSignFor(rep.sideTag, formation.strengthSign) so a weak-side rep's
   * reveal outlines the weak-side receiver. Defaults to strengthSign.
   */
  keySideSign?: 1 | -1;
};

const EASE_OUT = Easing.bezier(0.16, 1, 0.3, 1);

const ZONE_NAMES: Record<ZoneId, string> = {
  flat: "Flat",
  curl: "Curl",
  hook: "Hook",
  deep_third: "Deep 1/3",
  deep_half: "Deep 1/2",
  deep_middle: "Deep middle",
};

function withAlpha(hex: string, a: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

/** Plain-number mapping constants — safe to capture inside worklets. */
type WorkletMap = { xMin: number; xMax: number; yMin: number; ppy: number; mir: boolean };

export default function FieldCanvas({
  formation,
  call,
  highlightSlot,
  width,
  crop: cropProp,
  live,
  playTime,
  riveBodies,
  scrubTrack,
  spotlight,
  mirrored,
  zoneHints,
  targets,
  truth,
  marker,
  ghosts,
  selfSpot,
  gapLetters,
  reducedMotion,
  keySideSign,
}: FieldProps) {
  const { theme } = useTheme();
  const crop = cropProp ?? CROP;
  const map = useMemo(
    () => makeMapper(crop, width, !!mirrored),
    [crop.xMin, crop.xMax, crop.yMin, crop.yMax, width, mirrored]
  );
  const height = map.height;
  const ppy = map.pxPerYd;
  const bodyW = BODY.width * ppy;
  const bodyH = BODY.height * ppy;
  const cornerR = Math.min(BODY.cornerRadius * ppy, bodyW / 2, bodyH / 2);
  const wm: WorkletMap = useMemo(
    () => ({ xMin: crop.xMin, xMax: crop.xMax, yMin: crop.yMin, ppy, mir: !!mirrored }),
    [crop.xMin, crop.xMax, crop.yMin, ppy, mirrored]
  );

  const placements = useMemo(
    () => (call ? placeDefense(call, formation) : []),
    [call, formation]
  );
  const highlighted =
    highlightSlot === undefined
      ? undefined
      : placements.find((p) => p.assignment.slot === highlightSlot);
  const keyPlayer = highlighted?.keyPlayer ?? null;

  // ————— live partition —————
  const liveOn = !!live && !!playTime;
  const tracks = liveOn ? live!.tracks : [];
  const trackedIds = useMemo(() => new Set(tracks.map((t) => t.actorId)), [tracks]);
  const spotSet = useMemo(
    () => (spotlight ? new Set(spotlight.actors) : null),
    [spotlight]
  );
  const isSpotlit = (t: Track) =>
    !spotSet ? true : spotSet.size > 0 ? spotSet.has(t.actorId) : t.key;

  const anim = useTruthAnim(truth, !!reducedMotion);

  const boardColor = theme.dark ? chalk.boardDark : chalk.boardLight;

  // Yard lines every 5 within the crop (LOS drawn separately).
  const yardLines: number[] = [];
  for (let y = Math.ceil(crop.yMin / 5) * 5; y <= crop.yMax; y += 5) {
    if (y !== 0) yardLines.push(y);
  }
  const hashYs: number[] = [];
  for (let y = Math.ceil(crop.yMin); y <= Math.floor(crop.yMax); y += 1) {
    hashYs.push(y);
  }
  const hashInCrop = HASH_X >= crop.xMin && HASH_X <= crop.xMax;

  const approximate = placements.filter((p) => p.resolved?.approximate);
  const unresolved = placements.filter((p) => p.resolved === null);

  const gapSpots = useMemo(() => gapLetterSpots(formation), [formation]);

  // Truth helpers resolved once. Numbered keys ("#1"/"#2") exist on both
  // sides of most formations — the rep's authored side decides, never a
  // strong-side guess (CONTRACT decisions 10/11).
  const truthSideSign = keySideSign ?? formation.strengthSign;
  const truthKeyCorrect =
    truth?.kind === "KEY"
      ? resolveKeyPlayer(truth.correctKey, formation, truthSideSign)
      : null;
  const truthKeyYours =
    truth?.kind === "KEY" && truth.yoursKey && truth.yoursKey !== truth.correctKey
      ? resolveKeyPlayer(truth.yoursKey, formation, truthSideSign)
      : null;

  if (width <= 0) return <View />;

  return (
    <View style={{ width }}>
      <View style={{ width, height }}>
        <Canvas style={{ width, height }}>
          <Rect x={0} y={0} width={width} height={height} color={boardColor} />
          {/* Board band rules — 1px chalk.line top and bottom. */}
          <Line p1={vec(0, 0.5)} p2={vec(width, 0.5)} color={chalk.line} strokeWidth={1} />
          <Line
            p1={vec(0, height - 0.5)}
            p2={vec(width, height - 0.5)}
            color={chalk.line}
            strokeWidth={1}
          />

          {/* Yard lines every 5 · LOS 2px · hash ticks ±8.9. */}
          {yardLines.map((y) => (
            <Line
              key={`yl${y}`}
              p1={vec(0, map.sy(y))}
              p2={vec(width, map.sy(y))}
              color={chalk.line}
              strokeWidth={1}
            />
          ))}
          <Line
            p1={vec(0, map.sy(0))}
            p2={vec(width, map.sy(0))}
            color={chalk.los}
            strokeWidth={2}
          />
          {hashInCrop &&
            hashYs.map((y) =>
              [HASH_X, -HASH_X].map((hx) => (
                <Line
                  key={`h${y}:${hx}`}
                  p1={vec(map.sx(hx) - 3, map.sy(y))}
                  p2={vec(map.sx(hx) + 3, map.sy(y))}
                  color={chalk.line}
                  strokeWidth={1}
                />
              ))
            )}

          {/* Zone hints — faint accent tint, dashed edge (all equal; no leak). */}
          {zoneHints?.map((z) => {
            const c = map.fromField(z.center);
            const r = z.tolYd * ppy;
            return (
              <Group key={`zh${z.zone}:${z.center.x}:${z.center.y}`}>
                <Circle cx={c.x} cy={c.y} r={r} color={withAlpha(BOARD.accent, 0.09)} />
                <Circle
                  cx={c.x}
                  cy={c.y}
                  r={r}
                  style="stroke"
                  strokeWidth={1.5}
                  color={withAlpha(BOARD.accent, 0.28)}
                >
                  <DashPathEffect intervals={[5, 5]} />
                </Circle>
              </Group>
            );
          })}

          {/* Drop path — selfSpot → marker, dotted (PASS_DROP look). */}
          {selfSpot && marker && (
            <Line
              p1={vec(map.sx(selfSpot.x), map.sy(selfSpot.y))}
              p2={vec(map.sx(marker.x), map.sy(marker.y))}
              color={withAlpha(chalk.full, 0.55)}
              strokeWidth={2}
            >
              <DashPathEffect intervals={[2, 6]} />
            </Line>
          )}

          {/* Offense — chalk outlines, facing down. Live tracks take over. */}
          {formation.players.map((p, i) => {
            if (liveOn && trackedIds.has(p.actorId)) return null;
            const isKey = keyPlayer === p;
            const path = bodyPath(map.sx(p.x), map.sy(p.y), bodyW, bodyH, "down", cornerR);
            return (
              <Path
                key={`o${i}`}
                path={path}
                style="stroke"
                strokeWidth={isKey ? STROKE.key : STROKE.offense}
                color={isKey ? chalk.yellow : chalk.body}
              />
            );
          })}

          {/* Ball — chalk ellipse at the LOS (live ball comes from its track). */}
          {!(liveOn && trackedIds.has("ball")) && (
            <Oval
              x={map.sx(BALL.spot.x) - BALL.rx * ppy}
              y={map.sy(BALL.spot.y) - BALL.ry * ppy}
              width={BALL.rx * 2 * ppy}
              height={BALL.ry * 2 * ppy}
              style="stroke"
              strokeWidth={1.5}
              color={chalk.body}
            />
          )}

          {/* Defense — outline grammar. YOU is the only solid body. */}
          {placements.map((p) => {
            if (!p.resolved) return null;
            const isYou = p.assignment.slot === highlightSlot;
            const path = bodyPath(
              map.sx(p.resolved.x),
              map.sy(p.resolved.y),
              bodyW,
              bodyH,
              "up",
              cornerR
            );
            if (isYou) {
              return (
                <Group key={`d${p.assignment.slot}`}>
                  <Path path={path} color={BOARD.actionFill} />
                  <Path
                    path={path}
                    style="stroke"
                    strokeWidth={STROKE.youRing}
                    color={chalk.full}
                  >
                    {p.resolved.approximate ? <DashPathEffect intervals={[4, 4]} /> : null}
                  </Path>
                </Group>
              );
            }
            return (
              <Path
                key={`d${p.assignment.slot}`}
                path={path}
                style="stroke"
                strokeWidth={STROKE.defender}
                color={BOARD.accent}
              >
                {p.resolved.approximate ? <DashPathEffect intervals={[4, 4]} /> : null}
              </Path>
            );
          })}

          {/* Ghost teammates — 30% outline at their true spots. */}
          {ghosts?.map((g) => (
            <Group key={`g${g.slot}`} opacity={0.3}>
              <Path
                path={bodyPath(map.sx(g.spot.x), map.sy(g.spot.y), bodyW, bodyH, "up", cornerR)}
                style="stroke"
                strokeWidth={STROKE.defender}
                color={BOARD.accent}
              />
            </Group>
          ))}

          {/* Pre-placed YOU (RUN_FIT / READ). */}
          {selfSpot && (
            <YouBody
              x={map.sx(selfSpot.x)}
              y={map.sy(selfSpot.y)}
              w={bodyW}
              h={bodyH}
              r={cornerR}
            />
          )}

          {/* Controlled placement marker — YOU treatment. */}
          {marker && (
            <YouBody
              x={map.sx(marker.x)}
              y={map.sy(marker.y)}
              w={bodyW}
              h={bodyH}
              r={cornerR}
            />
          )}

          {/* Live playback — trails under bodies, everything off-React. */}
          {liveOn &&
            tracks
              .filter((t) => t.actorId !== "ball" && t.actorId !== "ghost" && t.segs.length > 0)
              .map((t) => (
                <LiveTrail key={`tr${t.actorId}`} track={t} m={wm} playTime={playTime!} />
              ))}
          {liveOn &&
            tracks.map((t) => {
              if (t.actorId === "ball") {
                return (
                  <LiveBall key="ball" track={t} m={wm} playTime={playTime!} ppy={ppy} />
                );
              }
              if (t.actorId === "ghost") {
                return (
                  <LiveGhost
                    key="ghost"
                    track={t}
                    m={wm}
                    playTime={playTime!}
                    w={bodyW}
                    h={bodyH}
                    r={cornerR}
                    ppy={ppy}
                  />
                );
              }
              if (riveBodies) return null; // body drawn by RiveTokenLayer above
              return (
                <LiveBody
                  key={`la${t.actorId}`}
                  track={t}
                  m={wm}
                  playTime={playTime!}
                  w={bodyW}
                  h={bodyH}
                  r={cornerR}
                  dim={!!spotlight && !isSpotlit(t)}
                  ring={!!spotlight && isSpotlit(t)}
                />
              );
            })}

          {/* Run-fit targets — tappable chalk circles. */}
          {targets?.map(({ marker: t, selected }) => (
            <TargetCircle
              key={t.id}
              cx={map.sx(t.center.x)}
              cy={map.sy(t.center.y)}
              r={0.62 * ppy}
              state={targetState(t.id, selected, truth)}
              anim={anim}
              reducedMotion={!!reducedMotion}
              bg={boardColor}
            />
          ))}

          {/* Truth — the board always shows the right answer. */}
          {truth?.kind === "SPOT" && (
            <SpotTruth truth={truth} map={map} ppy={ppy} anim={anim} />
          )}
          {truth?.kind === "ZONE" && (
            <ZoneTruth truth={truth} map={map} ppy={ppy} anim={anim} />
          )}
          {truth?.kind === "KEY" && (
            <>
              {truthKeyYours && (
                <ShakeGroup v={anim.shake}>
                  <Path
                    path={bodyPath(
                      map.sx(truthKeyYours.x),
                      map.sy(truthKeyYours.y),
                      bodyW,
                      bodyH,
                      "down",
                      cornerR
                    )}
                    style="stroke"
                    strokeWidth={STROKE.key}
                    color={BOARD.wrong}
                  />
                </ShakeGroup>
              )}
              {truthKeyCorrect && truth.yoursKey === truth.correctKey ? (
                <ScaleGroup
                  cx={map.sx(truthKeyCorrect.x)}
                  cy={map.sy(truthKeyCorrect.y)}
                  v={anim.pulse}
                  opacity={anim.appear}
                >
                  <Path
                    path={bodyPath(
                      map.sx(truthKeyCorrect.x),
                      map.sy(truthKeyCorrect.y),
                      bodyW,
                      bodyH,
                      "down",
                      cornerR
                    )}
                    style="stroke"
                    strokeWidth={STROKE.key}
                    color={BOARD.correct}
                  />
                </ScaleGroup>
              ) : truthKeyCorrect ? (
                <Group opacity={truthKeyYours ? anim.late : anim.appear}>
                  <Path
                    path={bodyPath(
                      map.sx(truthKeyCorrect.x),
                      map.sy(truthKeyCorrect.y),
                      bodyW,
                      bodyH,
                      "down",
                      cornerR
                    )}
                    style="stroke"
                    strokeWidth={STROKE.key}
                    color={chalk.yellow}
                  />
                </Group>
              ) : null}
            </>
          )}
        </Canvas>

        {/* ————— Label overlay — RN Text, diagram lettering, never scales. */}
        {formation.players.map((p, i) => {
          if (liveOn && trackedIds.has(p.actorId)) return null;
          let color = keyPlayer === p ? chalk.yellow : withAlpha(chalk.full, 0.7);
          if (truth?.kind === "KEY") {
            if (truthKeyCorrect === p) {
              color = truth.yoursKey === truth.correctKey ? BOARD.correct : chalk.yellow;
            } else if (truthKeyYours === p) {
              color = BOARD.wrong;
            }
          }
          return (
            <BoardLabel
              key={`ol${i}`}
              x={map.sx(p.x)}
              y={map.sy(p.y)}
              w={bodyW}
              h={bodyH}
              text={offenseLabel(p)}
              color={color}
            />
          );
        })}
        {liveOn &&
          tracks
            .filter((t) => t.actorId !== "ball" && t.actorId !== "ghost")
            .map((t) => (
              <LiveLabel
                key={`ll${t.actorId}`}
                track={t}
                m={wm}
                playTime={playTime!}
                w={bodyW}
                h={bodyH}
                text={actorLabel(t.actorId)}
                color={withAlpha(chalk.full, 0.7)}
                dim={!!spotlight && !isSpotlit(t)}
              />
            ))}
        {placements.map(
          (p) =>
            p.resolved && (
              <BoardLabel
                key={`dl${p.assignment.slot}`}
                x={map.sx(p.resolved.x)}
                y={map.sy(p.resolved.y)}
                w={bodyW}
                h={bodyH}
                text={positionTagLabel(p.assignment.label)}
                color={
                  p.assignment.slot === highlightSlot ? chalk.full : BOARD.accent
                }
              />
            )
        )}
        {ghosts?.map((g) => (
          <BoardLabel
            key={`gl${g.slot}`}
            x={map.sx(g.spot.x)}
            y={map.sy(g.spot.y)}
            w={bodyW}
            h={bodyH}
            text={positionTagLabel(g.label as PositionId)}
            color={withAlpha(BOARD.accent, 0.55)}
          />
        ))}
        {gapLetters &&
          gapSpots.map((g) => (
            <BoardLabel
              key={`gap${g.letter}${g.x}`}
              x={map.sx(g.x)}
              y={map.sy(g.y)}
              w={20}
              h={16}
              text={g.letter}
              color={withAlpha(chalk.full, 0.4)}
              size={11}
            />
          ))}
        {zoneHints?.map((z) => (
          <BoardLabel
            key={`zl${z.zone}:${z.center.x}`}
            x={map.sx(z.center.x)}
            y={map.sy(z.center.y)}
            w={90}
            h={16}
            text={ZONE_NAMES[z.zone]}
            color={withAlpha(chalk.full, 0.5)}
            size={11}
            family="Barlow_500Medium"
          />
        ))}
        {targets?.map(({ marker: t, selected }) => {
          const st = targetState(t.id, selected, truth);
          const c = targetLabelColor(st);
          const word = targetWord(t.id);
          if (word) {
            return (
              <BoardLabel
                key={`tl${t.id}`}
                x={map.sx(t.center.x)}
                y={map.sy(t.center.y) + 0.62 * ppy + 10}
                w={60}
                h={14}
                text={word}
                color={c}
                size={11}
                family="Barlow_500Medium"
              />
            );
          }
          return (
            <BoardLabel
              key={`tl${t.id}`}
              x={map.sx(t.center.x)}
              y={map.sy(t.center.y)}
              w={24}
              h={16}
              text={t.id[0]}
              color={c}
              size={12}
            />
          );
        })}
        {truth?.kind === "SPOT" && truth.yours && truth.distanceLabel && (
          <BoardLabel
            x={(map.sx(truth.yours.x) + map.sx(truth.truth.x)) / 2}
            y={(map.sy(truth.yours.y) + map.sy(truth.truth.y)) / 2 - 12}
            w={140}
            h={16}
            text={truth.distanceLabel}
            color={chalk.full}
            size={12}
            family="Barlow_500Medium"
          />
        )}
      </View>

      {/* Scrub track (F5) — a thin chalk progress bar under the board. Chalk
          grammar (hairline + chalk-white fill), not a new hue; the board
          pigments above are untouched. */}
      {scrubTrack ? (
        <ScrubTrack
          width={width}
          playTime={scrubTrack.playTime}
          totalMs={scrubTrack.totalMs}
        />
      ) : null}

      {/* Annotations live below the board, never on top of it. */}
      {(approximate.length > 0 || unresolved.length > 0) && (
        <View style={styles.notes}>
          {approximate.map((p) => (
            <AppText key={`a${p.assignment.slot}`} role="micro" color={theme.textSecondary}>
              {positionTagLabel(p.assignment.label)} ≈ “{p.assignment.alignment.raw}” (approximate)
            </AppText>
          ))}
          {unresolved.map((p) => (
            <AppText key={`u${p.assignment.slot}`} role="micro" color={theme.textSecondary}>
              {positionTagLabel(p.assignment.label)}: “{p.assignment.alignment.raw}” (not placed)
            </AppText>
          ))}
        </View>
      )}
    </View>
  );
}

// ————————————————————————————————————————————— truth animation

type TruthAnim = {
  appear: SharedValue<number>;
  pulse: SharedValue<number>;
  shake: SharedValue<number>;
  late: SharedValue<number>;
};

function truthDepKey(t: TruthOverlay | null | undefined): string {
  if (!t) return "none";
  switch (t.kind) {
    case "SPOT":
      return `S:${t.truth.x},${t.truth.y}:${t.yours ? `${t.yours.x},${t.yours.y}` : ""}`;
    case "RUN_TARGET":
      return `R:${t.correct}:${t.yours ?? ""}`;
    case "ZONE":
      return `Z:${t.landmark.x},${t.landmark.y}:${t.yours ? `${t.yours.x},${t.yours.y}` : ""}`;
    case "KEY":
      return `K:${t.correctKey}:${t.yoursKey ?? ""}`;
  }
}

function truthIsWrong(t: TruthOverlay): boolean {
  if (t.kind === "RUN_TARGET") return t.yours !== null && t.yours !== t.correct;
  if (t.kind === "KEY") return t.yoursKey !== null && t.yoursKey !== t.correctKey;
  return false;
}

function useTruthAnim(truth: TruthOverlay | null | undefined, reducedMotion: boolean): TruthAnim {
  const appear = useSharedValue(0);
  const pulse = useSharedValue(1);
  const shake = useSharedValue(0);
  const late = useSharedValue(0);
  const depKey = truthDepKey(truth);
  const wrong = truth ? truthIsWrong(truth) : false;

  useEffect(() => {
    cancelAnimation(appear);
    cancelAnimation(pulse);
    cancelAnimation(shake);
    cancelAnimation(late);
    shake.value = 0;
    pulse.value = 1;
    if (!truth) {
      appear.value = 0;
      late.value = 0;
      return;
    }
    if (reducedMotion) {
      // Reduce Motion: color changes only (DESIGN §5.4).
      appear.value = 1;
      late.value = 1;
      return;
    }
    appear.value = 0;
    appear.value = withTiming(1, { duration: 200, easing: EASE_OUT });
    if (wrong) {
      // ±3px horizontal, 3 cycles, 260ms; the right answer draws 200ms later.
      const step = 260 / 6;
      shake.value = withSequence(
        withTiming(-3, { duration: step }),
        withTiming(3, { duration: step }),
        withTiming(-3, { duration: step }),
        withTiming(3, { duration: step }),
        withTiming(-3, { duration: step }),
        withTiming(0, { duration: step })
      );
      late.value = 0;
      late.value = withDelay(200, withTiming(1, { duration: 200, easing: EASE_OUT }));
    } else {
      late.value = 1;
      pulse.value = withSequence(
        withTiming(1.12, { duration: 175, easing: EASE_OUT }),
        withTiming(1, { duration: 175, easing: EASE_OUT })
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depKey, reducedMotion]);

  return { appear, pulse, shake, late };
}

function ShakeGroup({ v, children }: { v: SharedValue<number>; children: React.ReactNode }) {
  const transform = useDerivedValue(() => [{ translateX: v.value }]);
  return <Group transform={transform}>{children}</Group>;
}

function ScaleGroup({
  cx,
  cy,
  v,
  opacity,
  children,
}: {
  cx: number;
  cy: number;
  v: SharedValue<number>;
  opacity?: SharedValue<number>;
  children: React.ReactNode;
}) {
  const transform = useDerivedValue(() => [{ scale: v.value }]);
  return (
    <Group origin={vec(cx, cy)} transform={transform} opacity={opacity}>
      {children}
    </Group>
  );
}

function BloomGroup({
  cx,
  cy,
  appear,
  children,
}: {
  cx: number;
  cy: number;
  appear: SharedValue<number>;
  children: React.ReactNode;
}) {
  const transform = useDerivedValue(() => [{ scale: 0.6 + 0.4 * appear.value }]);
  return (
    <Group origin={vec(cx, cy)} transform={transform} opacity={appear}>
      {children}
    </Group>
  );
}

// ————————————————————————————————————————————— truth layers

function SpotTruth({
  truth,
  map,
  ppy,
  anim,
}: {
  truth: Extract<TruthOverlay, { kind: "SPOT" }>;
  map: FieldMapper;
  ppy: number;
  anim: TruthAnim;
}) {
  const tp = map.fromField(truth.truth);
  const r = 1.0 * ppy;
  if (!truth.yours) {
    // Right where you stood — ring blooms under the marker.
    return (
      <BloomGroup cx={tp.x} cy={tp.y} appear={anim.appear}>
        <Circle cx={tp.x} cy={tp.y} r={r} style="stroke" strokeWidth={2} color={BOARD.correct} />
      </BloomGroup>
    );
  }
  const yp = map.fromField(truth.yours);
  return (
    <Group opacity={anim.appear}>
      <Line
        p1={vec(yp.x, yp.y)}
        p2={vec(tp.x, tp.y)}
        color={withAlpha(chalk.full, 0.75)}
        strokeWidth={1.5}
      >
        <DashPathEffect intervals={[6, 6]} />
      </Line>
      <Circle cx={tp.x} cy={tp.y} r={r} style="stroke" strokeWidth={2} color={chalk.yellow} />
    </Group>
  );
}

function ZoneTruth({
  truth,
  map,
  ppy,
  anim,
}: {
  truth: Extract<TruthOverlay, { kind: "ZONE" }>;
  map: FieldMapper;
  ppy: number;
  anim: TruthAnim;
}) {
  const lm = map.fromField(truth.landmark);
  const r = truth.tolYd * ppy;
  const yp = truth.yours ? map.fromField(truth.yours) : null;
  const far =
    truth.yours &&
    Math.hypot(truth.yours.x - truth.landmark.x, truth.yours.y - truth.landmark.y) > 0.75;
  return (
    <Group opacity={anim.appear}>
      <Circle cx={lm.x} cy={lm.y} r={r} color={withAlpha(BOARD.correct, 0.13)} />
      <Circle
        cx={lm.x}
        cy={lm.y}
        r={r}
        style="stroke"
        strokeWidth={1.5}
        color={withAlpha(BOARD.correct, 0.55)}
      >
        <DashPathEffect intervals={[5, 5]} />
      </Circle>
      {yp && far && (
        <Line
          p1={vec(yp.x, yp.y)}
          p2={vec(lm.x, lm.y)}
          color={withAlpha(chalk.full, 0.7)}
          strokeWidth={1.5}
        >
          <DashPathEffect intervals={[6, 6]} />
        </Line>
      )}
      <ScaleGroup cx={lm.x} cy={lm.y} v={anim.pulse}>
        <Circle
          cx={lm.x}
          cy={lm.y}
          r={0.5 * ppy}
          style="stroke"
          strokeWidth={2}
          color={BOARD.correct}
        />
      </ScaleGroup>
    </Group>
  );
}

// ————————————————————————————————————————————— run-fit targets

type TargetState = "normal" | "selected" | "hit" | "answer" | "wrong" | "dim";

function targetState(
  id: string,
  selected: boolean,
  truth: TruthOverlay | null | undefined
): TargetState {
  if (truth?.kind === "RUN_TARGET") {
    if (id === truth.correct) return truth.yours === truth.correct ? "hit" : "answer";
    if (id === truth.yours) return "wrong";
    return "dim";
  }
  return selected ? "selected" : "normal";
}

function targetWord(id: string): string | null {
  if (id.startsWith("FORCE")) return "Force";
  if (id.startsWith("ALLEY")) return "Alley";
  return null;
}

function targetLabelColor(state: TargetState): string {
  switch (state) {
    case "hit":
      return BOARD.correct;
    case "answer":
      return chalk.yellow;
    case "wrong":
      return BOARD.wrong;
    case "dim":
      return withAlpha(chalk.full, 0.35);
    case "selected":
      return BOARD.accent;
    default:
      return withAlpha(chalk.full, 0.9);
  }
}

function TargetCircle({
  cx,
  cy,
  r,
  state,
  anim,
  reducedMotion,
  bg,
}: {
  cx: number;
  cy: number;
  r: number;
  state: TargetState;
  anim: TruthAnim;
  reducedMotion: boolean;
  bg: string;
}) {
  // Selection pop — 1 → 1.06, 120ms, snap (DESIGN §5.4).
  const sel = useSharedValue(1);
  useEffect(() => {
    cancelAnimation(sel);
    if (state === "selected") {
      if (reducedMotion) {
        sel.value = 1.06;
      } else {
        sel.value = 1;
        sel.value = withTiming(1.06, { duration: 120, easing: EASE_OUT });
      }
    } else {
      sel.value = 1;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state === "selected", reducedMotion]);
  const selTransform = useDerivedValue(() => [{ scale: sel.value }]);

  // Punch-out disc under every state — the circles share rows with bodies
  // and route lines, and stroke-only circles let the chalk underneath merge
  // with the gap letter (unreadable; see the display-row note in reps/gaps.ts).
  const punch = <Circle cx={cx} cy={cy} r={r - 0.5} color={withAlpha(bg, 0.92)} />;

  switch (state) {
    case "hit":
      return (
        <ScaleGroup cx={cx} cy={cy} v={anim.pulse}>
          {punch}
          <Circle cx={cx} cy={cy} r={r} style="stroke" strokeWidth={2} color={BOARD.correct} />
        </ScaleGroup>
      );
    case "answer":
      return (
        <Group>
          {punch}
          <Circle
            cx={cx}
            cy={cy}
            r={r}
            style="stroke"
            strokeWidth={1.5}
            color={withAlpha(chalk.full, 0.4)}
          />
          <Group opacity={anim.late}>
            <Circle
              cx={cx}
              cy={cy}
              r={r + 3}
              style="stroke"
              strokeWidth={2}
              color={chalk.yellow}
            />
          </Group>
        </Group>
      );
    case "wrong":
      return (
        <ShakeGroup v={anim.shake}>
          {punch}
          <Circle cx={cx} cy={cy} r={r} style="stroke" strokeWidth={2} color={BOARD.wrong} />
        </ShakeGroup>
      );
    case "dim":
      return (
        <Group>
          {punch}
          <Circle
            cx={cx}
            cy={cy}
            r={r}
            style="stroke"
            strokeWidth={1.5}
            color={withAlpha(chalk.full, 0.3)}
          />
        </Group>
      );
    case "selected":
      return (
        <Group origin={vec(cx, cy)} transform={selTransform}>
          {punch}
          <Circle cx={cx} cy={cy} r={r} style="stroke" strokeWidth={2} color={BOARD.accent} />
        </Group>
      );
    default:
      return (
        <Group>
          {punch}
          <Circle
            cx={cx}
            cy={cy}
            r={r}
            style="stroke"
            strokeWidth={1.5}
            color={withAlpha(chalk.full, 0.9)}
          />
        </Group>
      );
  }
}

// ————————————————————————————————————————————— live actors

function LiveBody({
  track,
  m,
  playTime,
  w,
  h,
  r,
  dim,
  ring,
}: {
  track: Track;
  m: WorkletMap;
  playTime: SharedValue<number>;
  w: number;
  h: number;
  r: number;
  dim: boolean;
  ring: boolean;
}) {
  const transform = useDerivedValue(() => {
    const p = evaluatePose(track, playTime.value);
    const px = m.mir ? (p.x - m.xMin) * m.ppy : (m.xMax - p.x) * m.ppy;
    const py = (p.y - m.yMin) * m.ppy;
    const rot = (((m.mir ? -p.facingDeg : p.facingDeg) % 360) * Math.PI) / 180;
    return [
      { translateX: px },
      { translateY: py },
      { rotate: rot },
      { scaleY: 1 - 0.45 * p.flatten },
    ];
  }, [track, playTime, m]);
  const path = useMemo(() => bodyPathAtOrigin(w, h, r), [w, h, r]);
  const ringPath = useMemo(
    () => (ring ? bodyPathAtOrigin(w + 6, h + 6, r + 3) : null),
    [ring, w, h, r]
  );
  return (
    <Group transform={transform} opacity={dim ? 0.25 : 1}>
      <Path path={path} style="stroke" strokeWidth={STROKE.offense} color={chalk.body} />
      {ringPath && (
        <Path path={ringPath} style="stroke" strokeWidth={2} color={BOARD.accent} />
      )}
    </Group>
  );
}

function LiveBall({
  track,
  m,
  playTime,
  ppy,
}: {
  track: Track;
  m: WorkletMap;
  playTime: SharedValue<number>;
  ppy: number;
}) {
  const transform = useDerivedValue(() => {
    const p = evaluatePose(track, playTime.value);
    const px = m.mir ? (p.x - m.xMin) * m.ppy : (m.xMax - p.x) * m.ppy;
    const py = (p.y - m.yMin) * m.ppy;
    return [{ translateX: px }, { translateY: py }];
  }, [track, playTime, m]);
  return (
    <Group transform={transform}>
      <Oval
        x={-BALL.rx * ppy}
        y={-BALL.ry * ppy}
        width={BALL.rx * 2 * ppy}
        height={BALL.ry * 2 * ppy}
        style="stroke"
        strokeWidth={1.5}
        color={chalk.body}
      />
    </Group>
  );
}

function LiveGhost({
  track,
  m,
  playTime,
  w,
  h,
  r,
  ppy,
}: {
  track: Track;
  m: WorkletMap;
  playTime: SharedValue<number>;
  w: number;
  h: number;
  r: number;
  ppy: number;
}) {
  const transform = useDerivedValue(() => {
    const p = evaluatePose(track, playTime.value);
    const px = m.mir ? (p.x - m.xMin) * m.ppy : (m.xMax - p.x) * m.ppy;
    const py = (p.y - m.yMin) * m.ppy;
    const rot = (((m.mir ? -p.facingDeg : p.facingDeg) % 360) * Math.PI) / 180;
    return [{ translateX: px }, { translateY: py }, { rotate: rot }];
  }, [track, playTime, m]);
  const path = useMemo(() => bodyPathAtOrigin(w, h, r), [w, h, r]);
  const last = track.segs.length ? track.segs[track.segs.length - 1] : null;
  const end = last ? { x: last.x1, y: last.y1 } : { x: track.x0, y: track.y0 };
  const endPx = {
    x: m.mir ? (end.x - m.xMin) * m.ppy : (m.xMax - end.x) * m.ppy,
    y: (end.y - m.yMin) * m.ppy,
  };
  return (
    <Group>
      {/* Target ring at the attack point. */}
      <Circle
        cx={endPx.x}
        cy={endPx.y}
        r={1.5 * ppy}
        style="stroke"
        strokeWidth={1.5}
        color={withAlpha(BOARD.accent, 0.6)}
      >
        <DashPathEffect intervals={[5, 5]} />
      </Circle>
      <Group transform={transform} opacity={0.6}>
        <Path path={path} style="stroke" strokeWidth={STROKE.defender} color={BOARD.accent}>
          <DashPathEffect intervals={[4, 4]} />
        </Path>
      </Group>
    </Group>
  );
}

function LiveTrail({
  track,
  m,
  playTime,
}: {
  track: Track;
  m: WorkletMap;
  playTime: SharedValue<number>;
}) {
  const { path, fracs, times, t0, endMs } = useMemo(() => {
    const pts = [{ x: track.x0, y: track.y0 }].concat(
      track.segs.map((s) => ({ x: s.x1, y: s.y1 }))
    );
    const px = pts.map((p) => ({
      x: m.mir ? (p.x - m.xMin) * m.ppy : (m.xMax - p.x) * m.ppy,
      y: (p.y - m.yMin) * m.ppy,
    }));
    const skPath = Skia.Path.Make();
    skPath.moveTo(px[0].x, px[0].y);
    for (let i = 1; i < px.length; i++) skPath.lineTo(px[i].x, px[i].y);
    const lens: number[] = [0];
    let total = 0;
    for (let i = 1; i < px.length; i++) {
      total += Math.hypot(px[i].x - px[i - 1].x, px[i].y - px[i - 1].y);
      lens.push(total);
    }
    const fr = lens.map((l) => (total > 0 ? l / total : 0));
    const ts = [track.segs[0]?.t0 ?? 0].concat(track.segs.map((s) => s.t1));
    return {
      path: skPath as SkPath,
      fracs: fr,
      times: ts,
      t0: track.segs[0]?.t0 ?? 0,
      endMs: track.endMs,
    };
  }, [track, m]);

  const end = useDerivedValue(() => {
    const t = playTime.value;
    if (t <= times[0]) return 0;
    for (let i = 1; i < times.length; i++) {
      if (t < times[i]) {
        const u = (t - times[i - 1]) / Math.max(1, times[i] - times[i - 1]);
        return fracs[i - 1] + u * (fracs[i] - fracs[i - 1]);
      }
    }
    return 1;
  }, [playTime, times, fracs]);

  const opacity = useDerivedValue(() => {
    const t = playTime.value;
    if (t <= t0) return 0;
    if (t <= endMs) return Math.min(1, (t - t0) / 150);
    // Fade 600ms after the primitive completes.
    return Math.max(0, 1 - (t - endMs) / 600);
  }, [playTime, t0, endMs]);

  return (
    <Path
      path={path}
      style="stroke"
      strokeWidth={2}
      strokeCap="round"
      strokeJoin="round"
      color={chalk.body}
      start={0}
      end={end}
      opacity={opacity}
    />
  );
}

function LiveLabel({
  track,
  m,
  playTime,
  w,
  h,
  text,
  color,
  dim,
}: {
  track: Track;
  m: WorkletMap;
  playTime: SharedValue<number>;
  w: number;
  h: number;
  text: string;
  color: string;
  dim: boolean;
}) {
  const style = useAnimatedStyle(() => {
    const p = evaluatePose(track, playTime.value);
    const px = m.mir ? (p.x - m.xMin) * m.ppy : (m.xMax - p.x) * m.ppy;
    const py = (p.y - m.yMin) * m.ppy;
    return {
      transform: [{ translateX: px - w / 2 }, { translateY: py - h / 2 }],
    };
  }, [track, playTime, m, w, h]);
  return (
    <Animated.View
      pointerEvents="none"
      style={[styles.labelBox, { width: w, height: h, opacity: dim ? 0.25 : 1 }, style]}
    >
      <Text
        maxFontSizeMultiplier={1}
        style={{
          color,
          fontSize: Math.max(9, Math.round(h * 0.52)),
          fontFamily: "Barlow_700Bold",
        }}
      >
        {text}
      </Text>
    </Animated.View>
  );
}

// ————————————————————————————————————————————— shared pieces

/** YOU — the only solid body on the board. */
function YouBody({ x, y, w, h, r }: { x: number; y: number; w: number; h: number; r: number }) {
  const path = bodyPath(x, y, w, h, "up", r);
  return (
    <Group>
      <Path path={path} color={BOARD.actionFill} />
      <Path path={path} style="stroke" strokeWidth={STROKE.youRing} color={chalk.full} />
    </Group>
  );
}

/**
 * The scrub progress bar (F5). A hairline chalk track with a chalk-white fill
 * whose width follows playTime/totalMs on the UI thread — the "how far through
 * the play" reference under your finger. Plain RN + Reanimated so it lives
 * outside the Skia board (pigments untouched); chalk colors only, no new hue.
 */
function ScrubTrack({
  width,
  playTime,
  totalMs,
}: {
  width: number;
  playTime: SharedValue<number>;
  totalMs: number;
}) {
  const fillStyle = useAnimatedStyle(() => {
    const total = totalMs > 0 ? totalMs : 1;
    const t = playTime.value;
    const frac = t <= 0 ? 0 : t >= total ? 1 : t / total;
    return { width: `${frac * 100}%` };
  }, [playTime, totalMs]);
  return (
    <View style={{ width, paddingTop: space.s8 }}>
      <View
        style={{
          height: space.s2,
          borderRadius: radius.pill,
          backgroundColor: chalk.line,
          overflow: "hidden",
        }}
      >
        <Animated.View
          style={[
            { height: space.s2, borderRadius: radius.pill, backgroundColor: chalk.full },
            fillStyle,
          ]}
        />
      </View>
    </View>
  );
}

function BoardLabel({
  x,
  y,
  w,
  h,
  text,
  color,
  size,
  family,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  text: string;
  color: string;
  size?: number;
  family?: string;
}) {
  return (
    <View
      pointerEvents="none"
      style={[styles.labelBox, { left: x - w / 2, top: y - h / 2, width: w, height: h }]}
    >
      <Text
        maxFontSizeMultiplier={1}
        numberOfLines={1}
        style={{
          color,
          fontSize: size ?? Math.max(9, Math.round(h * 0.52)),
          fontFamily: family ?? "Barlow_700Bold",
        }}
      >
        {text}
      </Text>
    </View>
  );
}

function offenseLabel(p: FormationPlayer): string {
  if (p.receiverNumber) return String(p.receiverNumber);
  switch (p.role) {
    case "C":
      return "C";
    case "LG":
    case "RG":
      return "G";
    case "LT":
    case "RT":
      return "T";
    case "TE":
      return "Y";
    case "QB":
      return "Q";
    case "RB":
      return "R";
    default:
      return "";
  }
}

function actorLabel(actorId: string): string {
  if (actorId.startsWith("#")) return actorId.slice(1, 2);
  switch (actorId) {
    case "LG":
    case "RG":
      return "G";
    case "LT":
    case "RT":
      return "T";
    case "TE":
      return "Y";
    case "QB":
      return "Q";
    case "RB":
      return "R";
    default:
      return actorId;
  }
}

function gapLetterSpots(formation: Formation): { x: number; y: number; letter: string }[] {
  const spots: { x: number; y: number; letter: string }[] = [];
  const y = -0.6;
  for (const side of [1, -1] as const) {
    const hasTE = formation.players.some(
      (p) => p.role === "TE" && Math.sign(p.x) === side
    );
    spots.push({ x: 0.8 * side, y, letter: "A" });
    spots.push({ x: 2.5 * side, y, letter: "B" });
    spots.push({ x: (hasTE ? 4.4 : 4.9) * side, y, letter: "C" });
    if (hasTE) spots.push({ x: 6.5 * side, y, letter: "D" });
  }
  return spots;
}

/**
 * A body with a facing: rounded rect with one flat edge — the front.
 * A circle has no front; this single choice does more for comprehension
 * than anything else on the board.
 */
function bodyPath(
  cx: number,
  cy: number,
  w: number,
  h: number,
  facing: "up" | "down",
  r: number
): SkPath {
  const left = cx - w / 2;
  const right = cx + w / 2;
  const top = cy - h / 2;
  const bottom = cy + h / 2;
  const p = Skia.Path.Make();
  if (facing === "up") {
    // Flat top (front), rounded bottom corners (back).
    p.moveTo(left, top);
    p.lineTo(right, top);
    p.lineTo(right, bottom - r);
    p.quadTo(right, bottom, right - r, bottom);
    p.lineTo(left + r, bottom);
    p.quadTo(left, bottom, left, bottom - r);
    p.close();
  } else {
    // Flat bottom (front), rounded top corners (back).
    p.moveTo(left, bottom);
    p.lineTo(right, bottom);
    p.lineTo(right, top + r);
    p.quadTo(right, top, right - r, top);
    p.lineTo(left + r, top);
    p.quadTo(left, top, left, top + r);
    p.close();
  }
  return p;
}

/** Live bodies are drawn at origin facing down; the transform does the rest. */
function bodyPathAtOrigin(w: number, h: number, r: number): SkPath {
  return bodyPath(0, 0, w, h, "down", Math.min(r, w / 2, h / 2));
}

const styles = StyleSheet.create({
  labelBox: {
    position: "absolute",
    left: 0,
    top: 0,
    alignItems: "center",
    justifyContent: "center",
  },
  notes: {
    paddingTop: 8,
    paddingHorizontal: 20,
    gap: 2,
  },
});
