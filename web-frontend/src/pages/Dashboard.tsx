import type React from "react";
import { useCallback } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAuth } from "../contexts/AuthContext";
import {
  usePortfolio,
  usePositions,
  useClosePosition,
} from "../hooks/usePortfolio";
import { usePortfolioPerformance } from "../hooks/usePortfolioPerformance";
import { useChartPalette, chartTooltipStyle } from "../hooks/useChartPalette";
import {
  Badge,
  EmptyState,
  ErrorState,
  Spinner,
  StatCard,
} from "../components/ui";
import {
  formatCurrency,
  formatPercentage,
  getColorForValue,
} from "../utils/format";

const ICONS = {
  value:
    "M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v-.375c0-.621.504-1.125 1.125-1.125H20.25M2.25 6v9m18-10.5v.75c0 .414.336.75.75.75h.75m-1.5-1.5h.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-.375m1.5-1.5H21a.75.75 0 00-.75.75v.75m0 0H3.75m0 0h-.375a1.125 1.125 0 01-1.125-1.125V15m1.5 1.5v-.75A.75.75 0 003 15h-.75M15 10.5a3 3 0 11-6 0 3 3 0 016 0z",
  pnl: "M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941",
  total:
    "M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z",
  cash: "M2.25 8.25h19.5M2.25 9h19.5m-16.5 5.25h6m-6 2.25h3m-3.75 3h15a2.25 2.25 0 002.25-2.25V6.75A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25v10.5A2.25 2.25 0 004.5 19.5z",
};

const Icon: React.FC<{ d: string }> = ({ d }) => (
  <svg
    className="h-4 w-4"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.7}
    stroke="currentColor"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d={d} />
  </svg>
);

export const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const palette = useChartPalette();
  const {
    data: portfolio,
    isLoading: portfolioLoading,
    error: portfolioError,
    refetch,
  } = usePortfolio();
  const { data: positions, isLoading: positionsLoading } = usePositions();
  const { data: performance, isLoading: perfLoading } =
    usePortfolioPerformance("1M");
  const closePosition = useClosePosition();

  const handleClose = useCallback(
    (id: string) => {
      if (confirm("Close this position?")) closePosition.mutate(id);
    },
    [closePosition],
  );

  if (portfolioLoading || positionsLoading || perfLoading) {
    return <Spinner label="Loading portfolio" />;
  }

  if (portfolioError || !portfolio) {
    return (
      <ErrorState
        message="Unable to load portfolio data. Check that the backend is running, or continue in demo mode."
        onRetry={() => refetch()}
      />
    );
  }

  const openPositions = positions ?? [];
  const equityCurve =
    (performance as { equityCurve?: { timestamp: string; value: number }[] })
      ?.equityCurve ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm text-ink-muted">
            Welcome back{user?.name ? `, ${user.name.split(" ")[0]}` : ""}
          </p>
          <h1 className="font-display text-3xl font-bold text-ink">
            Trading dashboard
          </h1>
        </div>
        <Badge tone="pos">
          <span className="h-1.5 w-1.5 rounded-full bg-pos" /> Markets open
        </Badge>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Value"
          value={formatCurrency(portfolio.totalValue)}
          icon={<Icon d={ICONS.value} />}
        />
        <StatCard
          label="Daily P&L"
          value={formatCurrency(portfolio.dailyPnL)}
          trend={portfolio.dailyPnL >= 0 ? "up" : "down"}
          hint={
            portfolio.totalValue
              ? formatPercentage(portfolio.dailyPnL / portfolio.totalValue)
              : undefined
          }
          icon={<Icon d={ICONS.pnl} />}
        />
        <StatCard
          label="Total P&L"
          value={formatCurrency(portfolio.totalPnL)}
          trend={portfolio.totalPnL >= 0 ? "up" : "down"}
          icon={<Icon d={ICONS.total} />}
        />
        <StatCard
          label="Cash"
          value={formatCurrency(portfolio.cash)}
          icon={<Icon d={ICONS.cash} />}
        />
      </div>

      {/* Equity curve */}
      <div className="am-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ink">Equity curve</h2>
            <p className="text-sm text-ink-muted">Trailing 30 days</p>
          </div>
          <Badge tone="brand">30D</Badge>
        </div>
        {equityCurve.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={equityCurve}>
              <defs>
                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="0%"
                    stopColor={palette.brand}
                    stopOpacity={0.35}
                  />
                  <stop
                    offset="100%"
                    stopColor={palette.brand}
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={palette.grid}
                vertical={false}
              />
              <XAxis
                dataKey="timestamp"
                tick={{ fontSize: 11, fill: palette.axis }}
                stroke={palette.grid}
              />
              <YAxis
                tick={{ fontSize: 11, fill: palette.axis }}
                stroke={palette.grid}
                tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip
                contentStyle={chartTooltipStyle(palette)}
                formatter={(v: number) => [`$${v.toLocaleString()}`, "Equity"]}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke={palette.brand}
                strokeWidth={2}
                fill="url(#eqGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <EmptyState
            title="No performance data yet"
            hint="Equity history appears once trades settle."
          />
        )}
      </div>

      {/* Positions */}
      <div className="am-card overflow-hidden">
        <div className="flex items-center justify-between border-b border-line px-6 py-4">
          <h2 className="text-lg font-semibold text-ink">
            Open positions{" "}
            <span className="font-mono text-sm text-ink-muted">
              ({openPositions.length})
            </span>
          </h2>
        </div>
        {openPositions.length === 0 ? (
          <EmptyState
            title="No open positions"
            hint="Activate a strategy to start trading."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-line">
              <thead className="bg-surface-2">
                <tr>
                  {[
                    "Ticker",
                    "Qty",
                    "Entry",
                    "Current",
                    "Unrealised P&L",
                    "",
                  ].map((h) => (
                    <th
                      key={h}
                      className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-ink-muted last:text-right"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {openPositions.map((pos) => (
                  <tr
                    key={pos.id}
                    className="transition-colors hover:bg-surface-2"
                  >
                    <td className="px-6 py-4 text-sm font-semibold text-ink">
                      {pos.ticker}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink-muted">
                      {pos.quantity}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink-muted">
                      {formatCurrency(pos.entryPrice)}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink-muted">
                      {formatCurrency(pos.currentPrice)}
                    </td>
                    <td
                      className={`px-6 py-4 font-mono text-sm font-medium ${getColorForValue(pos.unrealizedPnL)}`}
                    >
                      {formatCurrency(pos.unrealizedPnL)}
                    </td>
                    <td className="px-6 py-4 text-right text-sm">
                      <button
                        onClick={() => handleClose(pos.id)}
                        disabled={closePosition.isPending}
                        className="font-medium text-neg hover:underline disabled:opacity-40"
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
