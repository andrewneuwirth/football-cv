// viewer/theme/colors.ts — neutral defaults (no team branding)
export const palette = {
  primary: "#2563EB",   // generic blue accent
  primaryLight: "#60A5FA",
  white: "#FFFFFF",
  ink: "#0B1220",
  paper: "#F7F8FA",
  slate: "#64748B",
};

// Board pigments — the chalk grammar the renderer draws with. Neutral,
// non-themed values so the field reads the same day and night.
export const chalk = {
  boardDark: "#14231D",
  boardLight: "#1E322A",
  line: "rgba(242,245,250,0.20)",
  los: "rgba(242,245,250,0.55)",
  body: "rgba(242,245,250,0.60)",
  full: "#F2F5FA",
  yellow: "#F5C84C",
} as const;

// Spacing + radius scales referenced by the renderer's plain-view chrome.
export const space = { s2: 2, s8: 8 } as const;
export const radius = { pill: 999 } as const;

// A minimal theme shape + a light/dark default. Neutral accents pulled from
// the palette above — no team colors.
export type Theme = {
  dark: boolean;
  accent: string;
  actionFill: string;
  correct: string;
  wrong: string;
  textSecondary: string;
};

export const lightTheme: Theme = {
  dark: false,
  accent: palette.primary,
  actionFill: palette.primary,
  correct: "#0E7A4E",
  wrong: "#C2402A",
  textSecondary: palette.slate,
};

export const darkTheme: Theme = {
  dark: true,
  accent: palette.primaryLight,
  actionFill: palette.primary,
  correct: "#3ECF8E",
  wrong: "#FF7A70",
  textSecondary: "#AEB6C8",
};

/** Neutral theme hook — no provider required; returns the dark board theme. */
export function useTheme(): { theme: Theme } {
  return { theme: darkTheme };
}
