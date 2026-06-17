import type React from "react";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCurrency } from "../utils/format";
import {
  useRiskMetrics,
  useStressScenarios,
  useCorrelationMatrix,
  useRiskRadar,
} from "../hooks/usePortfolioPerformance";
import { useChartPalette, chartTooltipStyle } from "../hooks/useChartPalette";

const CELL_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6"];

const heatColor = (v: number) => {
  if (v >= 0.8) return "bg-neg-soft text-neg";
  if (v >= 0.6) return "bg-warn-soft text-warn";
  if (v >= 0.4) return "bg-surface-2 text-ink-muted";
  return "bg-pos-soft text-pos";
};

export const RiskManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    "overview" | "stress" | "correlation"
  >("overview");

  const { data: riskMetrics, isLoading: rmLoading } = useRiskMetrics();
  const palette = useChartPalette();
  const { data: stressScenarios, isLoading: stressLoading } =
    useStressScenarios();
  const { data: correlationMatrix, isLoading: corrLoading } =
    useCorrelationMatrix();
  const { data: riskRadar, isLoading: radarLoading } = useRiskRadar();

  const isLoading = rmLoading || stressLoading || corrLoading || radarLoading;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand" />
      </div>
    );
  }

  const metrics = riskMetrics!;
  const scenarios = (stressScenarios ?? []) as {
    name: string;
    pnl: number;
    duration: string;
    recovery: string;
    portfolioImpact: number;
  }[];
  const corrData = (correlationMatrix ?? []) as Record<string, unknown>[];
  const radarData = (riskRadar ?? []) as { metric: string; value: number }[];
  const assets =
    corrData.length > 0
      ? Object.keys(corrData[0]).filter((k) => k !== "asset")
      : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-ink">Risk Management</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Portfolio risk analytics and stress testing
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-line">
        <nav className="flex space-x-6">
          {(["overview", "stress", "correlation"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setActiveTab(t)}
              className={`py-3 text-sm font-medium border-b-2 capitalize transition-colors ${activeTab === t ? "border-brand text-brand" : "border-transparent text-ink-muted hover:text-ink-muted"}`}
            >
              {t === "stress"
                ? "Stress Tests"
                : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
      </div>

      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Key Risk Metrics */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              {
                label: "1-Day VaR (95%)",
                value: `${metrics.var.toFixed(2)}%`,
                desc: "of NAV",
              },
              {
                label: "CVaR (95%)",
                value: `${metrics.cvar.toFixed(2)}%`,
                desc: "of NAV",
              },
              {
                label: "Sharpe Ratio",
                value: metrics.sharpeRatio.toFixed(2),
                desc: "annualised",
              },
              {
                label: "Sortino Ratio",
                value: metrics.sortinoRatio.toFixed(2),
                desc: "annualised",
              },
              {
                label: "Max Drawdown",
                value: `${(metrics.maxDrawdown * 100).toFixed(1)}%`,
                desc: "historical",
              },
              {
                label: "Portfolio Beta",
                value: metrics.beta.toFixed(2),
                desc: "vs S&P 500",
              },
              {
                label: "Correlation",
                value: metrics.correlation.toFixed(2),
                desc: "avg cross-asset",
              },
              {
                label: "Annualised Vol",
                value: `${(metrics.volatility * 100).toFixed(1)}%`,
                desc: "realised",
              },
            ].map((m) => (
              <div key={m.label} className="am-card p-4">
                <p className="text-xs text-ink-muted">{m.label}</p>
                <p className="text-2xl font-bold text-ink mt-1">{m.value}</p>
                <p className="text-xs text-ink-faint mt-0.5">{m.desc}</p>
              </div>
            ))}
          </div>

          {/* Risk Radar */}
          {radarData.length > 0 && (
            <div className="am-card p-6">
              <h2 className="text-base font-semibold text-ink mb-4">
                Risk Profile Radar
              </h2>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart
                  cx="50%"
                  cy="50%"
                  outerRadius="80%"
                  data={radarData}
                >
                  <PolarGrid stroke={palette.grid} />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fontSize: 12, fill: palette.axis }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={{ fontSize: 10, fill: palette.axis }}
                    stroke={palette.grid}
                  />
                  <Radar
                    name="Risk Score"
                    dataKey="value"
                    stroke={palette.neg}
                    fill={palette.neg}
                    fillOpacity={0.3}
                  />
                  <Tooltip
                    formatter={(v) => [`${v} / 100`, "Risk Score"]}
                    contentStyle={chartTooltipStyle(palette)}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {activeTab === "stress" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {scenarios.map((s) => (
              <div key={s.name} className="am-card p-5">
                <h3 className="text-sm font-semibold text-ink">{s.name}</h3>
                <p
                  className={`text-2xl font-bold mt-2 ${s.pnl < 0 ? "text-neg" : "text-pos"}`}
                >
                  {formatCurrency(s.pnl)}
                </p>
                <p className="text-xs text-ink-faint mt-1">
                  Portfolio impact:{" "}
                  <span className="font-medium text-neg">
                    {s.portfolioImpact.toFixed(1)}%
                  </span>
                </p>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-ink-muted">
                  <span>Duration: {s.duration}</span>
                  <span>Recovery: {s.recovery}</span>
                </div>
              </div>
            ))}
          </div>

          {scenarios.length > 0 && (
            <div className="am-card p-6">
              <h2 className="text-base font-semibold text-ink mb-4">
                Scenario P&L Impact
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={scenarios} layout="vertical">
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    stroke={palette.grid}
                  />
                  <XAxis
                    type="number"
                    tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                    tick={{ fontSize: 11, fill: palette.axis }}
                  />
                  <YAxis
                    dataKey="name"
                    type="category"
                    width={140}
                    tick={{ fontSize: 11, fill: palette.axis }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle(palette)}
                    formatter={(v) => [formatCurrency(Number(v)), "P&L"]}
                  />
                  <Bar dataKey="pnl" radius={[0, 4, 4, 0]}>
                    {scenarios.map((_, i) => (
                      <Cell
                        key={i}
                        fill={CELL_COLORS[Math.min(i, CELL_COLORS.length - 1)]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {activeTab === "correlation" && corrData.length > 0 && (
        <div className="am-card p-6">
          <h2 className="text-base font-semibold text-ink mb-4">
            Asset Correlation Matrix
          </h2>
          <div className="overflow-x-auto">
            <table className="text-sm w-full">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-left text-xs text-ink-muted uppercase">
                    Asset
                  </th>
                  {assets.map((a) => (
                    <th
                      key={a}
                      className="px-3 py-2 text-left text-xs text-ink-muted uppercase"
                    >
                      {a}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corrData.map((row) => (
                  <tr key={String(row.asset)}>
                    <td className="px-3 py-2 font-semibold text-ink">
                      {String(row.asset)}
                    </td>
                    {assets.map((a) => {
                      const v = Number(row[a]);
                      return (
                        <td
                          key={a}
                          className={`px-3 py-2 text-center rounded ${heatColor(v)}`}
                        >
                          {v.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
