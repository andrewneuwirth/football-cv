// Lazy Skia gate + fallbacks. The Skia canvas is code-split so web can load
// CanvasKit first. While it loads, the board skeleton draws its lines in
// chalk with bodies absent (DESIGN §6.11 — no spinners). If it can't load
// (a bus with no signal), the card renders as text — text always works.
import React, { Suspense } from "react";
import { View } from "react-native";
import { AppText, positionTagLabel } from "../components/ui";
import { useTheme, chalk } from "../theme/colors";
import type { FieldProps } from "./FieldCanvas";
import { CROP, HASH_X, boardHeight, makeMapper } from "./constants";
import { useSkiaReady } from "./useSkiaReady";

const FieldCanvas = React.lazy(() => import("./FieldCanvas"));

export function FieldRenderer(props: FieldProps) {
  const { ready, failed } = useSkiaReady();

  if (failed) return <TextFallback {...props} />;
  if (!ready) return <BoardSkeleton width={props.width} cropProp={props.crop} />;
  return (
    <Suspense fallback={<BoardSkeleton width={props.width} cropProp={props.crop} />}>
      <FieldCanvas {...props} />
    </Suspense>
  );
}

/** Chalk lines, bodies absent — plain views, works before Skia exists. */
function BoardSkeleton({
  width,
  cropProp,
}: {
  width: number;
  cropProp?: FieldProps["crop"];
}) {
  const { theme } = useTheme();
  const crop = cropProp ?? CROP;
  const height = boardHeight(crop, width);
  const map = makeMapper(crop, width);
  const yardLines: number[] = [];
  for (let y = Math.ceil(crop.yMin / 5) * 5; y <= crop.yMax; y += 5) {
    if (y !== 0) yardLines.push(y);
  }
  const hashInCrop = HASH_X >= crop.xMin && HASH_X <= crop.xMax;
  return (
    <View
      style={{
        width,
        height,
        backgroundColor: theme.dark ? chalk.boardDark : chalk.boardLight,
      }}
    >
      {yardLines.map((y) => (
        <View
          key={`yl${y}`}
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: map.sy(y),
            height: 1,
            backgroundColor: chalk.line,
          }}
        />
      ))}
      <View
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: map.sy(0) - 1,
          height: 2,
          backgroundColor: chalk.los,
        }}
      />
      {hashInCrop &&
        [HASH_X, -HASH_X].map((hx) => (
          <View
            key={`hx${hx}`}
            style={{
              position: "absolute",
              left: map.sx(hx) - 3,
              top: 0,
              bottom: 0,
              width: 1,
              backgroundColor: chalk.line,
              opacity: 0.5,
            }}
          />
        ))}
    </View>
  );
}

function TextFallback({ call, formation, highlightSlot }: FieldProps) {
  const { theme } = useTheme();
  return (
    <View style={{ gap: 4, paddingVertical: 12, paddingHorizontal: 20 }}>
      <AppText role="micro" color={theme.textSecondary}>
        Board offline — running it from the card.
      </AppText>
      {call ? (
        call.assignments.map((a) => (
          <AppText
            key={a.slot}
            role={a.slot === highlightSlot ? "bodyStrong" : "sub"}
            color={a.slot === highlightSlot ? theme.accent : theme.textSecondary}
          >
            {positionTagLabel(a.label)} — {a.alignment.raw}
          </AppText>
        ))
      ) : (
        <AppText role="sub" color={theme.textSecondary}>
          {formation.name}
        </AppText>
      )}
    </View>
  );
}
