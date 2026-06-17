import type React from "react";
import { Link } from "react-router-dom";
import { SectionHeading } from "../components/ui";

const teamMembers = [
  {
    role: "Quantitative Research Lead",
    focus: "ML model architecture and alpha generation",
  },
  {
    role: "Execution Engineer",
    focus: "Low-latency systems and smart order routing",
  },
  {
    role: "Risk Analyst",
    focus: "Portfolio risk, stress testing, and VaR modeling",
  },
  {
    role: "Data Engineer",
    focus: "Alternative data pipelines and real-time feeds",
  },
];

const stack = [
  {
    category: "AI / ML",
    items: ["PyTorch", "TensorFlow", "Scikit-learn", "Hugging Face"],
  },
  { category: "Backend", items: ["Python 3.10+", "Flask", "Celery", "Redis"] },
  {
    category: "Frontend",
    items: ["React 19", "TypeScript", "Tailwind CSS", "Recharts"],
  },
  {
    category: "Infrastructure",
    items: ["Docker", "PostgreSQL", "Apache Kafka", "Prometheus"],
  },
];

const CheckIcon = () => (
  <svg
    className="mr-2 h-4 w-4 text-pos"
    viewBox="0 0 20 20"
    fill="currentColor"
    aria-hidden="true"
  >
    <path
      fillRule="evenodd"
      d="M16.704 5.29a1 1 0 010 1.42l-7.5 7.5a1 1 0 01-1.42 0l-3.5-3.5a1 1 0 011.42-1.42l2.79 2.79 6.79-6.79a1 1 0 011.42 0z"
      clipRule="evenodd"
    />
  </svg>
);

export const About: React.FC = () => {
  return (
    <div className="mx-auto max-w-6xl space-y-12 px-6 py-12">
      <SectionHeading
        eyebrow="About"
        title="Institutional quant trading, powered by AI"
        subtitle="AlphaMind blends alternative data, machine learning, and disciplined execution to pursue durable, risk-adjusted returns."
      />

      <section className="am-card p-8">
        <h2 className="font-display text-2xl font-semibold text-ink">
          Our mission
        </h2>
        <p className="mt-4 leading-relaxed text-ink-muted">
          AlphaMind is an institutional-grade quantitative trading system that
          combines cutting-edge alternative data sources, advanced machine
          learning algorithms, and high-frequency execution strategies to
          achieve superior risk-adjusted returns in financial markets.
        </p>
        <p className="mt-4 leading-relaxed text-ink-muted">
          By leveraging Temporal Fusion Transformers, reinforcement learning,
          and real-time alternative data processing, AlphaMind continuously
          adapts to market regimes and discovers alpha where traditional methods
          fall short.
        </p>
        <div className="mt-6">
          <Link to="/documentation" className="am-btn am-btn-ghost">
            Read the documentation
          </Link>
        </div>
      </section>

      <section>
        <h2 className="mb-6 font-display text-2xl font-semibold text-ink">
          Core team
        </h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {teamMembers.map((member) => (
            <div key={member.role} className="am-card p-6">
              <h3 className="text-sm font-semibold text-ink">{member.role}</h3>
              <p className="mt-2 text-sm text-ink-muted">{member.focus}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="am-card p-8">
        <h2 className="mb-6 font-display text-2xl font-semibold text-ink">
          Technology stack
        </h2>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {stack.map((group) => (
            <div key={group.category}>
              <h3 className="am-eyebrow mb-3 text-brand">{group.category}</h3>
              <ul className="space-y-2">
                {group.items.map((item) => (
                  <li
                    key={item}
                    className="flex items-center text-sm text-ink-muted"
                  >
                    <CheckIcon />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="am-card flex flex-col items-center justify-between gap-4 p-8 sm:flex-row">
        <div>
          <h2 className="font-display text-xl font-semibold text-ink">
            Open source
          </h2>
          <p className="mt-1 text-sm text-ink-muted">
            AlphaMind is open source. Contributions, issues, and feature
            requests are welcome.
          </p>
        </div>
        <a
          href="https://github.com/quantsingularity/AlphaMind"
          target="_blank"
          rel="noopener noreferrer"
          className="am-btn am-btn-primary shrink-0"
        >
          <svg
            className="mr-2 h-5 w-5"
            fill="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              clipRule="evenodd"
            />
          </svg>
          View on GitHub
        </a>
      </section>
    </div>
  );
};

export default About;
