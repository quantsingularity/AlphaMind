import type React from "react";
import { useState } from "react";
import { useStrategies } from "../hooks/useStrategies";
import { formatPercentage } from "../utils/format";
import type { Strategy } from "../types";

const defaultStrategies: Strategy[] = [
  {
    id: "1",
    name: "TFT Alpha Strategy",
    type: "TFT",
    status: "active",
    description:
      "Temporal Fusion Transformer based multi-horizon forecasting strategy",
    performance: {
      sharpeRatio: 2.1,
      maxDrawdown: 0.12,
      profitFactor: 3.4,
      winRate: 0.62,
      totalReturn: 0.45,
      volatility: 0.15,
      alpha: 0.08,
      beta: 0.9,
    },
    parameters: {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: "2",
    name: "RL Portfolio Optimizer",
    type: "RL",
    status: "active",
    description: "Reinforcement learning based portfolio optimization",
    performance: {
      sharpeRatio: 1.8,
      maxDrawdown: 0.15,
      profitFactor: 2.9,
      winRate: 0.58,
      totalReturn: 0.38,
      volatility: 0.18,
      alpha: 0.06,
      beta: 1.1,
    },
    parameters: {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: "3",
    name: "Hybrid ML Strategy",
    type: "HYBRID",
    status: "inactive",
    description: "Combined approach using multiple ML models",
    performance: {
      sharpeRatio: 2.4,
      maxDrawdown: 0.09,
      profitFactor: 4.1,
      winRate: 0.65,
      totalReturn: 0.52,
      volatility: 0.14,
      alpha: 0.1,
      beta: 0.85,
    },
    parameters: {},
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];

export const Strategies: React.FC = () => {
  const { data: fetchedStrategies, isLoading } = useStrategies();
  const [localStrategies, setLocalStrategies] = useState<Strategy[]>(
    fetchedStrategies ?? defaultStrategies,
  );
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(
    null,
  );

  const strategies = fetchedStrategies ?? localStrategies;

  const toggleStatus = (id: string) => {
    setLocalStrategies((prev) =>
      prev.map((s) =>
        s.id === id
          ? { ...s, status: s.status === "active" ? "inactive" : "active" }
          : s,
      ),
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Strategies</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage and monitor your trading strategies
          </p>
        </div>
        <button className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
          <svg
            className="-ml-1 mr-2 h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          New Strategy
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {strategies.map((strategy) => (
          <div
            key={strategy.id}
            className="bg-white shadow rounded-lg overflow-hidden"
          >
            <div className="p-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900 truncate pr-2">
                  {strategy.name}
                </h3>
                <span
                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium flex-shrink-0 ${
                    strategy.status === "active"
                      ? "bg-green-100 text-green-800"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  {strategy.status}
                </span>
              </div>
              <p className="mt-2 text-sm text-gray-500">
                {strategy.description}
              </p>

              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-500">Sharpe Ratio</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {strategy.performance.sharpeRatio.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Win Rate</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {formatPercentage(strategy.performance.winRate)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Max Drawdown</p>
                  <p className="text-lg font-semibold text-red-600">
                    -{formatPercentage(strategy.performance.maxDrawdown)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Total Return</p>
                  <p className="text-lg font-semibold text-green-600">
                    {formatPercentage(strategy.performance.totalReturn)}
                  </p>
                </div>
              </div>

              <div className="mt-6 flex space-x-3">
                <button
                  onClick={() => setSelectedStrategy(strategy)}
                  className="flex-1 text-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  View Details
                </button>
                <button
                  onClick={() => toggleStatus(strategy.id)}
                  className={`flex-1 px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    strategy.status === "active"
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-green-600 hover:bg-green-700"
                  }`}
                >
                  {strategy.status === "active" ? "Deactivate" : "Activate"}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Strategy Detail Modal */}
      {selectedStrategy && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
          onClick={() => setSelectedStrategy(null)}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <h2 className="text-xl font-bold text-gray-900">
                  {selectedStrategy.name}
                </h2>
                <span
                  className={`inline-flex items-center mt-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    selectedStrategy.status === "active"
                      ? "bg-green-100 text-green-800"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  {selectedStrategy.status}
                </span>
              </div>
              <button
                onClick={() => setSelectedStrategy(null)}
                className="text-gray-400 hover:text-gray-600 focus:outline-none"
                aria-label="Close"
              >
                <svg
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            <p className="text-sm text-gray-600 mb-6">
              {selectedStrategy.description}
            </p>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: "Type", value: selectedStrategy.type },
                {
                  label: "Sharpe Ratio",
                  value: selectedStrategy.performance.sharpeRatio.toFixed(2),
                },
                {
                  label: "Win Rate",
                  value: formatPercentage(selectedStrategy.performance.winRate),
                },
                {
                  label: "Max Drawdown",
                  value: `-${formatPercentage(selectedStrategy.performance.maxDrawdown)}`,
                },
                {
                  label: "Total Return",
                  value: formatPercentage(
                    selectedStrategy.performance.totalReturn,
                  ),
                },
                {
                  label: "Profit Factor",
                  value: selectedStrategy.performance.profitFactor.toFixed(2),
                },
                {
                  label: "Volatility",
                  value: formatPercentage(
                    selectedStrategy.performance.volatility,
                  ),
                },
                {
                  label: "Alpha",
                  value: formatPercentage(selectedStrategy.performance.alpha),
                },
                {
                  label: "Beta",
                  value: selectedStrategy.performance.beta.toFixed(2),
                },
              ].map(({ label, value }) => (
                <div key={label} className="bg-gray-50 rounded-md p-3">
                  <p className="text-xs text-gray-500">{label}</p>
                  <p className="text-sm font-semibold text-gray-900">{value}</p>
                </div>
              ))}
            </div>
            <div className="mt-6 flex justify-end">
              <button
                onClick={() => setSelectedStrategy(null)}
                className="px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
