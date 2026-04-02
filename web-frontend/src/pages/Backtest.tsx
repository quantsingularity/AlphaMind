import type React from "react";
import { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatCurrency, formatPercentage } from "../utils/format";

interface BacktestConfig {
  strategyId: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
}

interface BacktestResultData {
  equityCurve: { date: string; equity: number; benchmark: number }[];
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  finalCapital: number;
}

const mockStrategies = [
  { id: "1", name: "TFT Alpha Strategy" },
  { id: "2", name: "RL Portfolio Optimizer" },
  { id: "3", name: "Hybrid ML Strategy" },
  { id: "4", name: "Sentiment-Enhanced" },
];

function generateMockResult(config: BacktestConfig): BacktestResultData {
  const start = new Date(config.startDate);
  const end = new Date(config.endDate);
  const days = Math.max(
    1,
    Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)),
  );

  const equityCurve = [];
  let equity = config.initialCapital;
  let benchmark = config.initialCapital;

  for (let i = 0; i <= days; i++) {
    equity += (Math.random() - 0.44) * (config.initialCapital * 0.005);
    benchmark += (Math.random() - 0.48) * (config.initialCapital * 0.003);
    const date = new Date(start);
    date.setDate(start.getDate() + i);
    equityCurve.push({
      date: date.toLocaleDateString(),
      equity: Math.round(equity),
      benchmark: Math.round(benchmark),
    });
  }

  const finalCapital =
    equityCurve[equityCurve.length - 1]?.equity ?? config.initialCapital;
  const totalReturn =
    (finalCapital - config.initialCapital) / config.initialCapital;

  return {
    equityCurve,
    totalReturn,
    sharpeRatio: 1.5 + Math.random() * 1.2,
    maxDrawdown: 0.05 + Math.random() * 0.15,
    winRate: 0.5 + Math.random() * 0.2,
    profitFactor: 1.5 + Math.random() * 2.5,
    totalTrades: Math.floor(days * 0.8 + Math.random() * 50),
    finalCapital,
  };
}

export const Backtest: React.FC = () => {
  const [config, setConfig] = useState<BacktestConfig>({
    strategyId: "1",
    startDate: "2024-01-01",
    endDate: "2024-12-31",
    initialCapital: 100000,
  });
  const [result, setResult] = useState<BacktestResultData | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRunBacktest = async () => {
    if (!config.strategyId || !config.startDate || !config.endDate) {
      setError("Please fill in all required fields.");
      return;
    }
    if (new Date(config.startDate) >= new Date(config.endDate)) {
      setError("Start date must be before end date.");
      return;
    }
    if (config.initialCapital <= 0) {
      setError("Initial capital must be a positive number.");
      return;
    }

    setError(null);
    setIsRunning(true);
    setResult(null);

    await new Promise((res) => setTimeout(res, 1500));

    try {
      const mockResult = generateMockResult(config);
      setResult(mockResult);
    } catch {
      setError("Failed to run backtest. Please try again.");
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Backtesting</h1>
        <p className="mt-1 text-sm text-gray-500">
          Run historical simulations on your trading strategies
        </p>
      </div>

      {/* Configuration Panel */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">
          Backtest Configuration
        </h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <label
              htmlFor="strategy"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Strategy
            </label>
            <select
              id="strategy"
              value={config.strategyId}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, strategyId: e.target.value }))
              }
              className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              {mockStrategies.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label
              htmlFor="startDate"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Start Date
            </label>
            <input
              id="startDate"
              type="date"
              value={config.startDate}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, startDate: e.target.value }))
              }
              className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <div>
            <label
              htmlFor="endDate"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              End Date
            </label>
            <input
              id="endDate"
              type="date"
              value={config.endDate}
              onChange={(e) =>
                setConfig((prev) => ({ ...prev, endDate: e.target.value }))
              }
              className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <div>
            <label
              htmlFor="capital"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Initial Capital ($)
            </label>
            <input
              id="capital"
              type="number"
              min="1000"
              step="1000"
              value={config.initialCapital}
              onChange={(e) =>
                setConfig((prev) => ({
                  ...prev,
                  initialCapital: Number(e.target.value),
                }))
              }
              className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        <div className="mt-4 flex space-x-3">
          <button
            onClick={handleRunBacktest}
            disabled={isRunning}
            className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {isRunning ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Running...
              </>
            ) : (
              <>
                <svg
                  className="-ml-1 mr-2 h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Run Backtest
              </>
            )}
          </button>
          {result && (
            <button
              onClick={handleReset}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Summary Metrics */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            {[
              {
                label: "Total Return",
                value: formatPercentage(result.totalReturn),
                color:
                  result.totalReturn >= 0 ? "text-green-600" : "text-red-600",
              },
              {
                label: "Final Capital",
                value: formatCurrency(result.finalCapital),
                color: "text-gray-900",
              },
              {
                label: "Sharpe Ratio",
                value: result.sharpeRatio.toFixed(2),
                color: "text-gray-900",
              },
              {
                label: "Max Drawdown",
                value: formatPercentage(result.maxDrawdown),
                color: "text-red-600",
              },
              {
                label: "Win Rate",
                value: formatPercentage(result.winRate),
                color: "text-gray-900",
              },
              {
                label: "Total Trades",
                value: result.totalTrades.toString(),
                color: "text-gray-900",
              },
            ].map((metric) => (
              <div
                key={metric.label}
                className="bg-white shadow rounded-lg p-4"
              >
                <p className="text-xs font-medium text-gray-500">
                  {metric.label}
                </p>
                <p className={`text-lg font-bold ${metric.color}`}>
                  {metric.value}
                </p>
              </div>
            ))}
          </div>

          {/* Equity Curve */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              Equity Curve vs Benchmark
            </h2>
            <ResponsiveContainer width="100%" height={350}>
              <AreaChart data={result.equityCurve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  interval={Math.floor(result.equityCurve.length / 8)}
                />
                <YAxis
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  formatter={(value) =>
                    formatCurrency(
                      typeof value === "number" ? value : Number(value),
                    )
                  }
                />
                <Area
                  type="monotone"
                  dataKey="benchmark"
                  stroke="#9ca3af"
                  fill="#f3f4f6"
                  fillOpacity={0.5}
                  name="Benchmark"
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#2563eb"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                  name="Strategy"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
};
