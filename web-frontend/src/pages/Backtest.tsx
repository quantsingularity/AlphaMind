import type React from "react";
import { useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCurrency, formatPercentage } from "../utils/format";
import { useStrategies } from "../hooks/useStrategies";
import { useChartPalette, chartTooltipStyle } from "../hooks/useChartPalette";
import { useRunBacktest } from "../hooks/useBacktest";
import type { BacktestResult } from "../types";

interface BacktestConfig {
  strategyId: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  benchmark: string;
  transactionCost: number;
  slippage: number;
}

const DEFAULT_CONFIG: BacktestConfig = {
  strategyId: "",
  startDate: "2019-01-01",
  endDate: "2023-12-31",
  initialCapital: 100_000,
  benchmark: "SPY",
  transactionCost: 0.0005,
  slippage: 0.0002,
};

export const Backtest: React.FC = () => {
  const [config, setConfig] = useState<BacktestConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<
    (BacktestResult & { equityCurve?: unknown[] }) | null
  >(null);

  const { data: strategies, isLoading: strategiesLoading } = useStrategies();
  const palette = useChartPalette();
  const runBacktest = useRunBacktest();

  const handleRun = async () => {
    if (!config.strategyId) return;
    const res = await runBacktest.mutateAsync({
      strategyId: config.strategyId,
      startDate: config.startDate,
      endDate: config.endDate,
      initialCapital: config.initialCapital,
    });
    setResult(res);
  };

  const equityCurve = result?.equityCurve ?? [];
  const monthlyReturns =
    equityCurve.length > 0
      ? equityCurve
          .filter((_, i) => i % 21 === 0)
          .map((pt, i) => ({
            month: `M${i + 1}`,
            return:
              i === 0
                ? 0
                : parseFloat(
                    (
                      (pt.equity /
                        equityCurve[Math.max(0, i * 21 - 21)].equity -
                        1) *
                      100
                    ).toFixed(2),
                  ),
          }))
      : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-ink">Backtesting</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Run walk-forward out-of-sample backtests on your strategies
        </p>
      </div>

      {/* Config Panel */}
      <div className="am-card p-6">
        <h2 className="text-base font-semibold text-ink mb-4">
          Backtest Configuration
        </h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1">
              Strategy
            </label>
            <select
              value={config.strategyId}
              onChange={(e) =>
                setConfig({ ...config, strategyId: e.target.value })
              }
              className="w-full border border-line rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            >
              <option value="">Select a strategy…</option>
              {(strategies ?? []).map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1">
              Start Date
            </label>
            <input
              type="date"
              value={config.startDate}
              onChange={(e) =>
                setConfig({ ...config, startDate: e.target.value })
              }
              className="w-full border border-line rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1">
              End Date
            </label>
            <input
              type="date"
              value={config.endDate}
              onChange={(e) =>
                setConfig({ ...config, endDate: e.target.value })
              }
              className="w-full border border-line rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1">
              Initial Capital ($)
            </label>
            <input
              type="number"
              value={config.initialCapital}
              onChange={(e) =>
                setConfig({ ...config, initialCapital: Number(e.target.value) })
              }
              className="w-full border border-line rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1">
              Transaction Cost (bps)
            </label>
            <input
              type="number"
              step="0.0001"
              value={config.transactionCost}
              onChange={(e) =>
                setConfig({
                  ...config,
                  transactionCost: Number(e.target.value),
                })
              }
              className="w-full border border-line rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1">
              Slippage (bps)
            </label>
            <input
              type="number"
              step="0.0001"
              value={config.slippage}
              onChange={(e) =>
                setConfig({ ...config, slippage: Number(e.target.value) })
              }
              className="w-full border border-line rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
        </div>
        <div className="mt-4 flex items-center space-x-3">
          <button
            onClick={handleRun}
            disabled={
              !config.strategyId || runBacktest.isPending || strategiesLoading
            }
            className="px-6 py-2 bg-brand text-white text-sm font-medium rounded-md hover:bg-brand-strong disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {runBacktest.isPending && (
              <span className="animate-spin h-4 w-4 border-2 border-surface border-t-transparent rounded-full" />
            )}
            <span>{runBacktest.isPending ? "Running…" : "Run Backtest"}</span>
          </button>
          {result && (
            <span className="text-xs text-pos font-medium">
              ✓ Backtest complete
            </span>
          )}
        </div>
        {runBacktest.isError && (
          <p className="mt-2 text-xs text-neg">
            Backtest failed. Check that the backend is running and a strategy is
            selected.
          </p>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Summary metrics */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-6">
            {[
              {
                label: "Total Return",
                value: formatPercentage(result.totalReturn),
              },
              {
                label: "Ann. Return",
                value: formatPercentage(result.annualisedReturn ?? 0),
              },
              {
                label: "Sharpe",
                value: (result.sharpeRatio ?? 0).toFixed(2),
              },
              {
                label: "Sortino",
                value: (result.sortinoRatio ?? 0).toFixed(2),
              },
              {
                label: "Max DD",
                value: formatPercentage(Math.abs(result.maxDrawdown ?? 0)),
              },
              {
                label: "Win Rate",
                value: formatPercentage(result.winRate ?? 0),
              },
              {
                label: "Final Capital",
                value: formatCurrency(result.finalCapital),
              },
              {
                label: "Profit Factor",
                value: (result.profitFactor ?? 0).toFixed(2),
              },
            ].map(({ label, value }) => (
              <div key={label} className="am-card p-4">
                <p className="text-xs text-ink-muted">{label}</p>
                <p className="text-lg font-bold text-ink mt-1">{value}</p>
              </div>
            ))}
          </div>

          {/* Equity curve */}
          {equityCurve.length > 0 && (
            <div className="am-card p-6">
              <h2 className="text-base font-semibold text-ink mb-4">
                Equity Curve vs Benchmark
              </h2>
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={equityCurve.filter((_, i) => i % 2 === 0)}>
                  <CartesianGrid strokeDasharray="3 3" stroke={palette.grid} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10, fill: palette.axis }}
                  />
                  <YAxis
                    yAxisId="left"
                    tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                    tick={{ fontSize: 11, fill: palette.axis }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fontSize: 11, fill: palette.axis }}
                    domain={["auto", 0]}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle(palette)}
                    formatter={(v, n) =>
                      n === "drawdown"
                        ? [`${Number(v).toFixed(2)}%`, "Drawdown"]
                        : [
                            formatCurrency(Number(v)),
                            n === "equity" ? "Strategy" : "Benchmark",
                          ]
                    }
                  />
                  <Legend />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="benchmark"
                    name="Benchmark"
                    stroke={palette.axis}
                    fill={palette.grid}
                    fillOpacity={0.14}
                    dot={false}
                  />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="equity"
                    name="Strategy"
                    stroke={palette.brand}
                    fill={palette.brand}
                    fillOpacity={0.2}
                    dot={false}
                  />
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="drawdown"
                    name="Drawdown %"
                    stroke={palette.neg}
                    fill={palette.neg}
                    fillOpacity={0.18}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Monthly returns */}
          {monthlyReturns.length > 0 && (
            <div className="am-card p-6">
              <h2 className="text-base font-semibold text-ink mb-4">
                Monthly Returns (%)
              </h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={monthlyReturns}>
                  <CartesianGrid strokeDasharray="3 3" stroke={palette.grid} />
                  <XAxis
                    dataKey="month"
                    tick={{ fontSize: 11, fill: palette.axis }}
                  />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fontSize: 11, fill: palette.axis }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle(palette)}
                    formatter={(v) => [`${Number(v).toFixed(2)}%`, "Return"]}
                  />
                  <ReferenceLine y={0} stroke={palette.axis} strokeWidth={1} />
                  <Bar dataKey="return" radius={[3, 3, 0, 0]}>
                    {monthlyReturns.map((m, i) => (
                      <Cell
                        key={i}
                        fill={m.return >= 0 ? palette.pos : palette.neg}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
