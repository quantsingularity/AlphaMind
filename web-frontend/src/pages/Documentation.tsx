import React from "react";

export const Documentation: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Documentation</h1>
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Getting Started</h2>
        <p className="text-gray-600 mb-4">
          Welcome to AlphaMind documentation. This platform provides
          comprehensive tools for quantitative trading.
        </p>
        <h3 className="text-lg font-semibold mt-6 mb-2">Key Features</h3>
        <ul className="list-disc list-inside space-y-2 text-gray-600">
          <li>Alternative data integration</li>
          <li>Machine learning models</li>
          <li>Risk management tools</li>
          <li>Backtesting framework</li>
        </ul>
      </div>
    </div>
  );
};
