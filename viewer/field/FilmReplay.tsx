// Replays a play tracked from film: all 22 players animated over a whole-play
// field view. Self-contained. Coordinates are ball-origin (+y = defensive
// backfield, +x = the defense's left), in yards, matching positions.json.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { Canvas, Group, Line, Path, Skia, vec } from "@shopify/react-native-skia";
import type { SkPath } from "@shopify/react-native-skia";
import { makeMapper } from "./constants";

// Player body: a rounded shape with a flat "front" (facing down), rotated to
// the direction of travel. Mirrors FieldCanvas.bodyPath so film players read
// like the board figures, not dots.
function bodyPathAtOrigin(w: number, h: number, r: number): SkPath {
  const rr = Math.min(r, w / 2, h / 2);
  const left = -w / 2, right = w / 2, top = -h / 2, bottom = h / 2;
  const p = Skia.Path.Make();
  p.moveTo(left, bottom);
  p.lineTo(right, bottom);
  p.lineTo(right, top + rr);
  p.quadTo(right, top, right - rr, top);
  p.lineTo(left + rr, top);
  p.quadTo(left, top, left, top + rr);
  p.close();
  return p;
}

type Sample = { t: number; x: number; y: number; f?: number };
type Player = { id: string; team: "O" | "D"; label?: string | null; jersey?: string | null; samples: Sample[] };
export type FilmPlay = { play_id: string; players: Player[] };

const O_COLOR = "#22d3ee"; // offense — cyan
const D_COLOR = "#e6b31e"; // defense — gold
const PAD_YD = 3;
// clip the crop so one deep/wide receiver can't shrink the whole play; the core
// action stays big and only a far outlier gets cropped at the edge.
const MAX_HALF_W = 22; // yд each side of the ball, laterally
const Y_LO = -8, Y_HI = 26; // yд behind LOS ... into the defensive backfield

function posAt(samples: Sample[], t: number): { x: number; y: number; f: number } | null {
  if (!samples.length) return null;
  const clampFace = (s: Sample) => (s.f ?? 0);
  if (t <= samples[0].t) return { ...samples[0], f: clampFace(samples[0]) };
  const lastS = samples[samples.length - 1];
  if (t >= lastS.t) return { ...lastS, f: clampFace(lastS) };
  for (let i = 1; i < samples.length; i++) {
    if (t <= samples[i].t) {
      const a = samples[i - 1], b = samples[i];
      const k = (t - a.t) / Math.max(1, b.t - a.t);
      return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, f: b.f ?? a.f ?? 0 };
    }
  }
  return { ...lastS, f: clampFace(lastS) };
}

export default function FilmReplay({ play, width = 360 }: { play: FilmPlay; width?: number }) {
  const [ms, setMs] = useState(0);
  const [playing, setPlaying] = useState(true);
  const raf = useRef<number | null>(null);
  const last = useRef<number>(0);

  // whole-play extent → crop (so deep routes + full width fit, not the box crop)
  const { crop, totalMs } = useMemo(() => {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity, tMax = 0;
    for (const p of play.players)
      for (const s of p.samples) {
        xMin = Math.min(xMin, s.x); xMax = Math.max(xMax, s.x);
        yMin = Math.min(yMin, s.y); yMax = Math.max(yMax, s.y);
        tMax = Math.max(tMax, s.t);
      }
    return {
      crop: {
        xMin: Math.max(xMin - PAD_YD, -MAX_HALF_W),
        xMax: Math.min(xMax + PAD_YD, MAX_HALF_W),
        yMin: Math.max(yMin - PAD_YD, Y_LO),
        yMax: Math.min(yMax + PAD_YD, Y_HI),
      },
      totalMs: tMax,
    };
  }, [play]);

  const map = useMemo(() => makeMapper(crop, width), [crop, width]);
  const ppy = map.pxPerYd;
  const bodyW = Math.max(10, ppy * 1.15);
  const bodyH = Math.max(14, ppy * 1.7);
  const body = useMemo(() => bodyPathAtOrigin(bodyW, bodyH, bodyW * 0.45), [bodyW, bodyH]);

  useEffect(() => {
    if (!playing) { if (raf.current) cancelAnimationFrame(raf.current); return; }
    last.current = 0;
    const tick = (now: number) => {
      if (!last.current) last.current = now;
      const dt = now - last.current;
      last.current = now;
      setMs((m) => (m + dt > totalMs ? 0 : m + dt)); // loop
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [playing, totalMs]);

  // yard lines every 5 yд of depth within the crop
  const yLines: number[] = [];
  for (let y = Math.ceil(crop.yMin / 5) * 5; y <= crop.yMax; y += 5) yLines.push(y);

  const live = play.players
    .map((p) => ({ p, pos: posAt(p.samples, ms) }))
    .filter((e) => e.pos) as { p: Player; pos: { x: number; y: number; f: number } }[];

  return (
    <View>
      <View style={{ width, height: map.height, backgroundColor: "#123", borderRadius: 8, overflow: "hidden" }}>
        <Canvas style={{ width, height: map.height }}>
          {yLines.map((y) => (
            <Line key={y} p1={vec(0, map.sy(y))} p2={vec(width, map.sy(y))}
              color={y === 0 ? "#ffffff" : "rgba(255,255,255,0.18)"} strokeWidth={y === 0 ? 2 : 1} />
          ))}
          <Line p1={vec(map.sx(8.9), 0)} p2={vec(map.sx(8.9), map.height)} color="rgba(255,255,255,0.10)" strokeWidth={1} />
          <Line p1={vec(map.sx(-8.9), 0)} p2={vec(map.sx(-8.9), map.height)} color="rgba(255,255,255,0.10)" strokeWidth={1} />
          {live.map(({ p, pos }) => {
            // facing 0 = downfield = down-screen (bodyPath's rest). Negate on
            // the mirrored x axis so bodies turn the correct way on screen.
            const rot = ((pos.f ?? 0) * Math.PI) / 180;
            return (
              <Group key={p.id}
                transform={[{ translateX: map.sx(pos.x) }, { translateY: map.sy(pos.y) }, { rotate: -rot }]}>
                <Path path={body} color={p.team === "O" ? O_COLOR : D_COLOR} />
              </Group>
            );
          })}
        </Canvas>
        {/* labels as RN text overlays (avoids Skia font loading) */}
        {live.map(({ p, pos }) =>
          p.label ? (
            <Text key={p.id} style={[styles.lbl, { left: map.sx(pos.x) - 14, top: map.sy(pos.y) + bodyH / 2 + 1 }]}>
              {p.label}
            </Text>
          ) : null
        )}
      </View>
      <View style={styles.controls}>
        <Pressable style={styles.btn} onPress={() => setPlaying((v) => !v)}>
          <Text style={styles.btnTxt}>{playing ? "❚❚" : "▶"}</Text>
        </Pressable>
        <Pressable style={styles.btn} onPress={() => { setMs(0); }}>
          <Text style={styles.btnTxt}>⟲</Text>
        </Pressable>
        <Text style={styles.time}>{(ms / 1000).toFixed(1)}s / {(totalMs / 1000).toFixed(1)}s</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  lbl: {
    position: "absolute", width: 28, textAlign: "center",
    color: "#fff", fontSize: 9, fontWeight: "700",
    textShadowColor: "#000", textShadowRadius: 2,
  },
  controls: { flexDirection: "row", alignItems: "center", gap: 10, marginTop: 10 },
  btn: { backgroundColor: "#182238", borderRadius: 8, paddingVertical: 6, paddingHorizontal: 14 },
  btnTxt: { color: "#fff", fontSize: 16 },
  time: { color: "#9aa4b2", fontFamily: "monospace", fontSize: 12 },
});
