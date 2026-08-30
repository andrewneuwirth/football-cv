// Minimal neutral text primitives for the viewer. No branding — just enough
// for the field renderer's fallback + annotation text.
import React from "react";
import { Text } from "react-native";

type Role = "micro" | "sub" | "body" | "bodyStrong";

const SIZE: Record<Role, number> = {
  micro: 11,
  sub: 13,
  body: 15,
  bodyStrong: 15,
};

export function AppText({
  role = "body",
  color,
  children,
}: {
  role?: Role;
  color?: string;
  children?: React.ReactNode;
}) {
  return (
    <Text
      maxFontSizeMultiplier={1}
      style={{
        color,
        fontSize: SIZE[role],
        fontWeight: role === "bodyStrong" ? "700" : "400",
      }}
    >
      {children}
    </Text>
  );
}

/** Display label for a generic position token (DL/LB/CB/S, OL/QB/RB/WR). */
export function positionTagLabel(id: string): string {
  return id;
}
