import React from "react";
import { Link } from "react-router-dom";

const features = [
  {
    name: "Alternative Data Integration",
    description:
      "Process satellite imagery, SEC filings, sentiment analysis, and social media data for comprehensive market insights.",
    icon: "🛰️",
    items: [
      "Satellite imagery processing",
      "SEC 8K real-time monitoring",
      "Earnings call NLP sentiment analysis",
      "Social media sentiment tracking",
    ],
  },
  {
    name: "Quantitative Research",
    description:
      "Advanced ML models and quantitative techniques for alpha generation and strategy development.",
    icon: "📊",
    items: [
      "Machine learning factor models",
      "Regime switching detection",
      "Exotic derivatives pricing",
      "Advanced portfolio optimization",
    ],
  },
  {
    name: "Execution Infrastructure",
    description:
      "High-performance execution engine with smart order routing and adaptive algorithms.",
    icon: "⚡",
    items: [
      "Microsecond latency arbitrage",
      "Hawkes process liquidity forecasting",
      "Smart order routing",
      "Adaptive execution algorithms",
    ],
  },
  {
    name: "Risk Management",
    description:
      "Comprehensive risk assessment and mitigation framework with real-time monitoring.",
    icon: "🛡️",
    items: [
      "Bayesian VaR with regime adjustments",
      "Counterparty credit risk (CVA/DVA)",
      "Extreme scenario stress testing",
      "Real-time risk monitoring",
    ],
  },
];

const performanceMetrics = [
  {
    strategy: "TFT Alpha",
    sharpeRatio: 2.1,
    maxDD: "12%",
    profitFactor: 3.4,
    winRate: "62%",
  },
  {
    strategy: "RL Portfolio",
    sharpeRatio: 1.8,
    maxDD: "15%",
    profitFactor: 2.9,
    winRate: "58%",
  },
  {
    strategy: "Hybrid Approach",
    sharpeRatio: 2.4,
    maxDD: "9%",
    profitFactor: 4.1,
    winRate: "65%",
  },
  {
    strategy: "Sentiment-Enhanced",
    sharpeRatio: 2.2,
    maxDD: "11%",
    profitFactor: 3.7,
    winRate: "63%",
  },
];

export const Home: React.FC = () => {
  return (
    <div className="bg-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 bg-white sm:pb-16 md:pb-20 lg:w-full lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <div className="sm:text-center lg:text-left">
                <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span className="block">Next-Gen</span>
                  <span className="block text-blue-600">
                    Quantitative AI Trading System
                  </span>
                </h1>
                <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                  An institutional-grade platform combining alternative data,
                  advanced machine learning, and high-frequency execution
                  strategies for superior market performance.
                </p>
                <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                  <div className="rounded-md shadow">
                    <Link
                      to="/dashboard"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 md:py-4 md:text-lg md:px-10"
                    >
                      Get Started
                    </Link>
                  </div>
                  <div className="mt-3 sm:mt-0 sm:ml-3">
                    <a
                      href="https://github.com/quantsingularity/AlphaMind"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 md:py-4 md:text-lg md:px-10"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      View on GitHub
                    </a>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              Key Features
            </h2>
            <p className="mt-4 text-xl text-gray-500">
              Comprehensive tools for institutional-grade quantitative trading
            </p>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-4">
              {features.map((feature) => (
                <div
                  key={feature.name}
                  className="bg-white rounded-lg shadow-lg p-6"
                >
                  <div className="text-4xl mb-4">{feature.icon}</div>
                  <h3 className="text-lg font-medium text-gray-900">
                    {feature.name}
                  </h3>
                  <p className="mt-2 text-sm text-gray-500">
                    {feature.description}
                  </p>
                  <ul className="mt-4 space-y-2">
                    {feature.items.map((item) => (
                      <li
                        key={item}
                        className="text-sm text-gray-600 flex items-start"
                      >
                        <span className="text-blue-500 mr-2">✓</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* AI/ML Core Section */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-2 lg:gap-8 items-center">
            <div>
              <h2 className="text-3xl font-extrabold text-gray-900">
                AI/ML Core
              </h2>
              <p className="mt-4 text-lg text-gray-500">
                AlphaMind leverages cutting-edge artificial intelligence and
                machine learning techniques to gain a competitive edge in
                financial markets.
              </p>
              <ul className="mt-8 space-y-4">
                {[
                  "Temporal Fusion Transformers for multi-horizon forecasting",
                  "Reinforcement Learning for portfolio optimization",
                  "Generative Models for market simulation",
                  "Attention Mechanisms for time series analysis",
                  "Sentiment Analysis for alternative data processing",
                ].map((item) => (
                  <li key={item} className="flex items-start">
                    <span className="flex-shrink-0 h-6 w-6 text-blue-500">
                      ✓
                    </span>
                    <span className="ml-3 text-gray-700">{item}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-8">
                <Link
                  to="/documentation"
                  className="inline-flex items-center px-6 py-3 border border-blue-600 text-base font-medium rounded-md text-blue-600 bg-white hover:bg-blue-50"
                >
                  Learn More
                </Link>
              </div>
            </div>
            <div className="mt-10 lg:mt-0">
              <img
                src="/src/assets/images/ai-ml-diagram.svg"
                alt="AI/ML Architecture"
                className="w-full h-auto"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics Section */}
      <div className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              Performance Metrics
            </h2>
            <p className="mt-4 text-xl text-gray-500">
              Historical backtesting results across different strategies
            </p>
          </div>

          <div className="mt-10">
            <div className="flex flex-col">
              <div className="-my-2 overflow-x-auto sm:-mx-6 lg:-mx-8">
                <div className="py-2 align-middle inline-block min-w-full sm:px-6 lg:px-8">
                  <div className="shadow overflow-hidden border-b border-gray-200 sm:rounded-lg">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Strategy
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Sharpe Ratio
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Max DD
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Profit Factor
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Win Rate
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {performanceMetrics.map((metric) => (
                          <tr key={metric.strategy}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {metric.strategy}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {metric.sharpeRatio}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {metric.maxDD}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {metric.profitFactor}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {metric.winRate}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Getting Started Section */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              Getting Started
            </h2>
            <p className="mt-4 text-xl text-gray-500">
              Quick setup to get AlphaMind running
            </p>
          </div>

          <div className="mt-10">
            <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
              <pre className="text-gray-100 text-sm">
                <code>
                  {`# Clone repository
git clone https://github.com/quantsingularity/AlphaMind.git
cd AlphaMind

# Install dependencies
pip install -r requirements.txt

# Start alternative data ingestion
python -m alternative_data.main \\
  --satellite-api-key $SAT_KEY \\
  --sec-monitor-tickers AAPL,TSLA,NVDA`}
                </code>
              </pre>
            </div>
            <div className="mt-6 text-center">
              <Link
                to="/documentation"
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                View Full Documentation
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
