import { MD3DarkTheme, MD3LightTheme } from "react-native-paper";

// AlphaMind "Quant Terminal" design system, unified with the web frontend.
// Light: bright canvas for daytime desks. Dark: low-glare terminal.
// Brand indigo with a cyan accent; color-coded positive/negative deltas.

export const lightTheme = {
  ...MD3LightTheme,
  roundness: 3,
  colors: {
    ...MD3LightTheme.colors,
    primary: "#4f46e5",
    primaryLight: "#e7e9fd",
    primaryDark: "#4338ca",
    onPrimary: "#ffffff",
    secondary: "#56657f",
    onSecondary: "#ffffff",
    tertiary: "#0891b2",
    accent: "#0891b2",
    error: "#e11d48",
    onError: "#ffffff",
    success: "#047857",
    positive: "#047857",
    negative: "#e11d48",
    warning: "#b45309",
    background: "#f5f7fb",
    surface: "#ffffff",
    surfaceVariant: "#eef1f7",
    elevation: {
      ...MD3LightTheme.colors.elevation,
      level1: "#ffffff",
      level2: "#f8fafc",
    },
    onBackground: "#0e1729",
    onSurface: "#0e1729",
    onSurfaceVariant: "#56657f",
    onSurfaceFaint: "#8593ad",
    outline: "#cdd6e4",
    outlineVariant: "#e1e7f0",
  },
};

export const darkTheme = {
  ...MD3DarkTheme,
  roundness: 3,
  colors: {
    ...MD3DarkTheme.colors,
    primary: "#6366f1",
    primaryLight: "#1e2547",
    primaryDark: "#818cf8",
    onPrimary: "#ffffff",
    secondary: "#94a2bb",
    onSecondary: "#0e1729",
    tertiary: "#22d3ee",
    accent: "#22d3ee",
    error: "#fb7185",
    onError: "#0e1729",
    success: "#34d399",
    positive: "#34d399",
    negative: "#fb7185",
    warning: "#fbbf24",
    background: "#080d18",
    surface: "#101a2e",
    surfaceVariant: "#16223a",
    elevation: {
      ...MD3DarkTheme.colors.elevation,
      level1: "#101a2e",
      level2: "#16223a",
    },
    onBackground: "#e9eef8",
    onSurface: "#e9eef8",
    onSurfaceVariant: "#94a2bb",
    onSurfaceFaint: "#61708b",
    outline: "#2c3e5e",
    outlineVariant: "#21304a",
  },
};

export const getTheme = (isDark) => (isDark ? darkTheme : lightTheme);
