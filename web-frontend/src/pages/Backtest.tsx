import React from "react";

export const Backtest: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Backtesting</h1>
      <div className="bg-white shadow rounded-lg p-6">
        <p className="text-gray-500">
          Run historical backtests on your trading strategies.
        </p>
        <button className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
          New Backtest
        </button>
      </div>
    </div>
  );
};
