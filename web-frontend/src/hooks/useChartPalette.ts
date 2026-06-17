import { useTheme } from "../contexts/ThemeContext";

export interface ChartPalette {
  brand: string;
  accent: string;
  pos: string;
  neg: string;
  grid: string;
  axis: string;
  tooltipBg: string;
  tooltipBorder: string;
  series: string[];
}

const LIGHT: ChartPalette = {
  brand: "#4f46e5",
  accent: "#0891b2",
  pos: "#047857",
  neg: "#e11d48",
  grid: "#e1e7f0",
  axis: "#8593ad",
  tooltipBg: "#ffffff",
  tooltipBorder: "#e1e7f0",
  series: ["#4f46e5", "#0891b2", "#7c3aed", "#d97706", "#e11d48", "#047857"],
};

const DARK: ChartPalette = {
  brand: "#818cf8",
  accent: "#22d3ee",
  pos: "#34d399",
  neg: "#fb7185",
  grid: "#21304a",
  axis: "#61708b",
  tooltipBg: "#16223a",
  tooltipBorder: "#21304a",
  series: ["#818cf8", "#22d3ee", "#a78bfa", "#fbbf24", "#fb7185", "#34d399"],
};

export function useChartPalette(): ChartPalette {
  const { theme } = useTheme();
  return theme === "dark" ? DARK : LIGHT;
}

export function chartTooltipStyle(p: ChartPalette) {
  return {
    backgroundColor: p.tooltipBg,
    border: `1px solid ${p.tooltipBorder}`,
    borderRadius: 10,
    fontSize: 12,
    color: p.axis,
  } as const;
}
