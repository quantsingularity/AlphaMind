import type React from "react";
import { Link } from "react-router-dom";

const teamMembers = [
  {
    role: "Quantitative Research Lead",
    focus: "ML model architecture, alpha generation",
    icon: "🔬",
  },
  {
    role: "Execution Engineer",
    focus: "Low-latency systems, smart order routing",
    icon: "⚡",
  },
  {
    role: "Risk Analyst",
    focus: "Portfolio risk, stress testing, VaR modeling",
    icon: "🛡️",
  },
  {
    role: "Data Engineer",
    focus: "Alternative data pipelines, real-time feeds",
    icon: "📡",
  },
];

const stack = [
  {
    category: "AI/ML",
    items: ["PyTorch", "TensorFlow", "Scikit-learn", "Hugging Face"],
  },
  {
    category: "Backend",
    items: ["Python 3.10+", "Flask", "Celery", "Redis"],
  },
  {
    category: "Frontend",
    items: ["React 19", "TypeScript", "Tailwind CSS", "Recharts"],
  },
  {
    category: "Infrastructure",
    items: ["Docker", "PostgreSQL", "Apache Kafka", "Prometheus"],
  },
];

export const About: React.FC = () => {
  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">About AlphaMind</h1>
        <p className="mt-1 text-sm text-gray-500">
          Institutional-grade quantitative trading powered by AI
        </p>
      </div>

      {/* Mission */}
      <div className="bg-white shadow rounded-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Our Mission</h2>
        <p className="text-gray-600 leading-relaxed">
          AlphaMind is an institutional-grade quantitative trading system that
          combines cutting-edge alternative data sources, advanced machine
          learning algorithms, and high-frequency execution strategies to
          achieve superior risk-adjusted returns in financial markets.
        </p>
        <p className="mt-4 text-gray-600 leading-relaxed">
          By leveraging Temporal Fusion Transformers, reinforcement learning,
          and real-time alternative data processing, AlphaMind continuously
          adapts to market regimes and discovers alpha where traditional methods
          fall short.
        </p>
        <div className="mt-6">
          <Link
            to="/documentation"
            className="inline-flex items-center px-4 py-2 border border-blue-600 text-sm font-medium rounded-md text-blue-600 hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Read the Docs →
          </Link>
        </div>
      </div>

      {/* Team */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Core Team</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {teamMembers.map((member) => (
            <div
              key={member.role}
              className="bg-white shadow rounded-lg p-6 text-center"
            >
              <div className="text-4xl mb-3">{member.icon}</div>
              <h3 className="font-semibold text-gray-900 text-sm">
                {member.role}
              </h3>
              <p className="mt-1 text-xs text-gray-500">{member.focus}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-white shadow rounded-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Technology Stack
        </h2>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {stack.map((group) => (
            <div key={group.category}>
              <h3 className="text-sm font-semibold text-blue-600 uppercase tracking-wider mb-3">
                {group.category}
              </h3>
              <ul className="space-y-2">
                {group.items.map((item) => (
                  <li
                    key={item}
                    className="flex items-center text-sm text-gray-700"
                  >
                    <span className="text-green-500 mr-2">✓</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Open Source */}
      <div className="bg-blue-50 border border-blue-100 rounded-lg p-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Open Source</h2>
          <p className="mt-1 text-gray-600 text-sm">
            AlphaMind is open source. Contributions, issues and feature requests
            are welcome.
          </p>
        </div>
        <a
          href="https://github.com/quantsingularity/AlphaMind"
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 inline-flex items-center px-5 py-2.5 border border-transparent text-sm font-medium rounded-md text-white bg-gray-900 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          <svg className="h-5 w-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
            <path
              fillRule="evenodd"
              d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              clipRule="evenodd"
            />
          </svg>
          View on GitHub
        </a>
      </div>
    </div>
  );
};
