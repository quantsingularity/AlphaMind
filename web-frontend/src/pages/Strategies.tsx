import React from "react";
import { useStrategies } from "../hooks/useStrategies";
import { Link } from "react-router-dom";
import { formatPercentage } from "../utils/format";

export const Strategies: React.FC = () => {
  const { data: strategies, isLoading } = useStrategies();

  const mockStrategies = strategies || [
    {
      id: "1",
      name: "TFT Alpha Strategy",
      type: "TFT" as const,
      status: "active" as const,
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
      type: "RL" as const,
      status: "active" as const,
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
      type: "HYBRID" as const,
      status: "inactive" as const,
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
        <button className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700">
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
        {mockStrategies.map((strategy) => (
          <div
            key={strategy.id}
            className="bg-white shadow rounded-lg overflow-hidden"
          >
            <div className="p-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">
                  {strategy.name}
                </h3>
                <span
                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
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
                    {strategy.performance.sharpeRatio}
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
                    {formatPercentage(strategy.performance.maxDrawdown)}
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
                <Link
                  to={`/strategies/${strategy.id}`}
                  className="flex-1 text-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                >
                  View Details
                </Link>
                <button className="flex-1 px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700">
                  {strategy.status === "active" ? "Deactivate" : "Activate"}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
