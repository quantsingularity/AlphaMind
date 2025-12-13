import React from "react";
import { usePositions } from "../hooks/usePortfolio";
import { formatCurrency, getColorForValue } from "../utils/format";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"];

export const Portfolio: React.FC = () => {
  const { data: positions } = usePositions();

  const mockPositions = positions || [
    {
      id: "1",
      ticker: "AAPL",
      quantity: 100,
      entryPrice: 150.0,
      currentPrice: 155.5,
      unrealizedPnL: 550.0,
    },
    {
      id: "2",
      ticker: "MSFT",
      quantity: 50,
      entryPrice: 300.0,
      currentPrice: 310.0,
      unrealizedPnL: 500.0,
    },
    {
      id: "3",
      ticker: "GOOGL",
      quantity: 25,
      entryPrice: 2800.0,
      currentPrice: 2750.0,
      unrealizedPnL: -1250.0,
    },
    {
      id: "4",
      ticker: "TSLA",
      quantity: 30,
      entryPrice: 700.0,
      currentPrice: 720.0,
      unrealizedPnL: 600.0,
    },
  ];

  const allocationData = mockPositions.map((p) => ({
    name: p.ticker,
    value: p.quantity * p.currentPrice,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Portfolio</h1>
        <p className="mt-1 text-sm text-gray-500">
          Overview of your current positions and allocation
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">
            Asset Allocation
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={allocationData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(props: any) =>
                  `${props.name || ""} ${props.percent ? (props.percent * 100).toFixed(0) : 0}%`
                }
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {allocationData.map((_, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => formatCurrency(value)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">
            Portfolio Summary
          </h2>
          <dl className="space-y-4">
            <div className="flex justify-between">
              <dt className="text-sm font-medium text-gray-500">Total Value</dt>
              <dd className="text-sm font-semibold text-gray-900">
                {formatCurrency(125430.5)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm font-medium text-gray-500">
                Cash Balance
              </dt>
              <dd className="text-sm font-semibold text-gray-900">
                {formatCurrency(25430.5)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm font-medium text-gray-500">Total P&L</dt>
              <dd className={`text-sm font-semibold ${getColorForValue(2340)}`}>
                {formatCurrency(2340.25)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm font-medium text-gray-500">
                Number of Positions
              </dt>
              <dd className="text-sm font-semibold text-gray-900">
                {mockPositions.length}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Positions</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Ticker
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Quantity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Entry Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Current Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Market Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Unrealized P&L
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {mockPositions.map((position) => (
                <tr key={position.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {position.ticker}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {position.quantity}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatCurrency(position.entryPrice)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatCurrency(position.currentPrice)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatCurrency(position.quantity * position.currentPrice)}
                  </td>
                  <td
                    className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getColorForValue(position.unrealizedPnL)}`}
                  >
                    {formatCurrency(position.unrealizedPnL)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
