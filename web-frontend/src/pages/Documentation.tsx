import type React from "react";
import { useState } from "react";

const sections = [
  {
    id: "getting-started",
    title: "Getting Started",
    content: [
      {
        heading: "Prerequisites",
        body: "Python 3.10+, Node.js 18+, Docker (optional), and API keys for alternative data sources.",
      },
      {
        heading: "Installation",
        body: `Clone the repository and install dependencies:\n\ngit clone https://github.com/quantsingularity/AlphaMind.git\ncd AlphaMind\npip install -r requirements.txt\nnpm install --prefix web-frontend`,
        isCode: true,
      },
      {
        heading: "Configuration",
        body: "Copy .env.example to .env and fill in your API keys and database credentials before starting the application.",
      },
    ],
  },
  {
    id: "strategies",
    title: "Trading Strategies",
    content: [
      {
        heading: "TFT Alpha Strategy",
        body: "Uses Temporal Fusion Transformers for multi-horizon price forecasting. Configurable lookback windows, attention heads, and forecast horizons.",
      },
      {
        heading: "RL Portfolio Optimizer",
        body: "Proximal Policy Optimization (PPO) agent that learns dynamic portfolio weights. Reward function balances Sharpe ratio against drawdown penalties.",
      },
      {
        heading: "Hybrid ML Strategy",
        body: "Ensemble approach combining TFT signals with RL position sizing and sentiment scores from alternative data. Achieves the highest risk-adjusted returns in backtests.",
      },
    ],
  },
  {
    id: "alternative-data",
    title: "Alternative Data",
    content: [
      {
        heading: "Satellite Imagery",
        body: "Processes parking lot occupancy, shipping traffic, and agricultural data from satellite providers. Used to generate lead indicators for retail and commodity sectors.",
      },
      {
        heading: "SEC Filings",
        body: "Real-time monitoring of 8-K, 10-K, and 13-F filings. NLP pipeline extracts sentiment, forward-looking statements, and unusual disclosures.",
      },
      {
        heading: "Social Sentiment",
        body: "Aggregates Reddit, Twitter/X, and StockTwits mentions. BERT-based classifier produces per-ticker sentiment scores updated every 5 minutes.",
      },
    ],
  },
  {
    id: "risk",
    title: "Risk Management",
    content: [
      {
        heading: "Value at Risk (VaR)",
        body: "Bayesian VaR with regime adjustments. Switches between normal, trending, and crisis regimes using Hidden Markov Models.",
      },
      {
        heading: "Stress Testing",
        body: "Pre-built scenarios include 2008 Financial Crisis, COVID-19 crash, and custom user-defined shocks. Reports P&L impact and factor exposures.",
      },
      {
        heading: "Position Limits",
        body: "Configurable per-ticker, per-sector, and total gross/net exposure limits. Breaches are flagged in real-time and orders are blocked automatically.",
      },
    ],
  },
  {
    id: "api",
    title: "API Reference",
    content: [
      {
        heading: "Authentication",
        body: "All endpoints require a Bearer token in the Authorization header. Tokens are issued via POST /api/auth/token.",
        isCode: false,
      },
      {
        heading: "Strategies Endpoints",
        body: `GET    /api/strategies          # List all strategies\nPOST   /api/strategies          # Create strategy\nGET    /api/strategies/:id      # Get strategy\nPUT    /api/strategies/:id      # Update strategy\nDELETE /api/strategies/:id      # Delete strategy\nPOST   /api/strategies/:id/activate`,
        isCode: true,
      },
      {
        heading: "Portfolio Endpoints",
        body: `GET /api/portfolio        # Portfolio overview\nGET /api/positions        # All open positions\nPOST /api/positions/:id/close`,
        isCode: true,
      },
    ],
  },
];

export const Documentation: React.FC = () => {
  const [activeSection, setActiveSection] = useState(sections[0].id);

  const current = sections.find((s) => s.id === activeSection) ?? sections[0];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Documentation</h1>
        <p className="mt-1 text-sm text-gray-500">
          Complete reference for the AlphaMind platform
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar */}
        <nav className="lg:w-56 flex-shrink-0">
          <div className="bg-white shadow rounded-lg p-4">
            <ul className="space-y-1">
              {sections.map((section) => (
                <li key={section.id}>
                  <button
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                      activeSection === section.id
                        ? "bg-blue-50 text-blue-700"
                        : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                    }`}
                  >
                    {section.title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="bg-white shadow rounded-lg p-6 space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">
              {current.title}
            </h2>
            {current.content.map((item) => (
              <div key={item.heading}>
                <h3 className="text-base font-semibold text-gray-900 mb-2">
                  {item.heading}
                </h3>
                {item.isCode ? (
                  <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 text-sm overflow-x-auto whitespace-pre">
                    <code>{item.body}</code>
                  </pre>
                ) : (
                  <p className="text-sm text-gray-600 leading-relaxed">
                    {item.body}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
