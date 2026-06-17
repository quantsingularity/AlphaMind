import type React from "react";
import { Link } from "react-router-dom";
import { Badge, SectionHeading } from "../components/ui";

const features = [
  {
    name: "Alternative Data",
    description:
      "Fuse satellite imagery, SEC filings, and social sentiment into a single tradable signal.",
    points: [
      "Satellite occupancy & shipping",
      "Real-time 8-K monitoring",
      "Earnings-call NLP sentiment",
      "Cross-source signal fusion",
    ],
    icon: "M2.25 12.76c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 011.037-.443 48.282 48.282 0 005.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z",
  },
  {
    name: "Quant Research",
    description:
      "Machine-learning factor models and regime detection for systematic alpha.",
    points: [
      "ML factor models",
      "Regime-switching detection",
      "Exotic derivative pricing",
      "Portfolio optimization",
    ],
    icon: "M3.75 3v11.25A2.25 2.25 0 006 16.5h12M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5m.75-9l3-3 2.148 2.148A12.061 12.061 0 0116.5 7.605",
  },
  {
    name: "Execution Engine",
    description:
      "High-performance routing with adaptive algorithms and microsecond order management.",
    points: [
      "Microsecond arbitrage",
      "Hawkes liquidity forecasting",
      "Smart order routing",
      "Adaptive TWAP / VWAP",
    ],
    icon: "M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z",
  },
  {
    name: "Risk Management",
    description:
      "Probabilistic risk with regime-aware VaR, stress testing, and live monitoring.",
    points: [
      "Bayesian VaR",
      "Counterparty CVA / DVA",
      "Extreme scenario stress",
      "Real-time risk monitoring",
    ],
    icon: "M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z",
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

const tickers = [
  { sym: "AAPL", chg: "+1.24%", up: true },
  { sym: "NVDA", chg: "+2.81%", up: true },
  { sym: "TSLA", chg: "-0.92%", up: false },
  { sym: "MSFT", chg: "+0.47%", up: true },
  { sym: "SPY", chg: "+0.33%", up: true },
];

export const Home: React.FC = () => {
  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-line">
        <div className="am-grid-bg absolute inset-0 opacity-[0.55] [mask-image:radial-gradient(ellipse_at_top,black,transparent_72%)]" />
        <div
          className="pointer-events-none absolute -top-40 left-1/2 h-[34rem] w-[34rem] -translate-x-1/2 rounded-full blur-3xl"
          style={{
            background:
              "radial-gradient(circle, color-mix(in oklab, var(--brand) 28%, transparent), transparent 70%)",
          }}
        />
        <div className="relative mx-auto max-w-7xl px-4 py-20 sm:px-6 sm:py-28 lg:px-8">
          <div className="mx-auto max-w-3xl text-center">
            <span className="am-rise inline-flex">
              <Badge tone="brand">Institutional-grade · Quant AI</Badge>
            </span>
            <h1 className="am-rise mt-6 font-display text-4xl font-bold leading-[1.05] tracking-tight text-ink sm:text-6xl">
              Where alternative data
              <br />
              becomes{" "}
              <span className="bg-gradient-to-r from-brand to-accent bg-clip-text text-transparent">
                durable alpha
              </span>
            </h1>
            <p className="am-rise mx-auto mt-6 max-w-xl text-lg text-ink-muted">
              AlphaMind pairs machine learning, alternative data, and
              high-frequency execution into one quantitative trading system,
              built for desks that measure edge in basis points.
            </p>
            <div className="am-rise mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Link
                to="/signup"
                className="am-btn am-btn-primary px-7 py-3 text-base"
              >
                Get started free
              </Link>
              <Link
                to="/signin"
                className="am-btn am-btn-ghost px-7 py-3 text-base"
              >
                Sign in
              </Link>
            </div>
          </div>

          {/* Live ticker strip */}
          <div className="am-rise mx-auto mt-14 max-w-3xl">
            <div className="am-card flex flex-wrap items-center justify-center gap-x-6 gap-y-2 px-5 py-3">
              <span className="flex items-center gap-2 text-xs font-medium text-ink-muted">
                <span className="h-2 w-2 rounded-full bg-pos am-pulse" /> Live
              </span>
              {tickers.map((t) => (
                <span
                  key={t.sym}
                  className="flex items-center gap-1.5 font-mono text-sm"
                >
                  <span className="font-semibold text-ink">{t.sym}</span>
                  <span className={t.up ? "text-pos" : "text-neg"}>
                    {t.chg}
                  </span>
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <SectionHeading
          center
          eyebrow="Capabilities"
          title="Four engines, one edge"
          subtitle="Every layer of the stack is built for systematic, risk-aware trading."
        />
        <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f) => (
            <div
              key={f.name}
              className="am-card p-6 transition-transform hover:-translate-y-1"
            >
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-brand-soft text-brand">
                <svg
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.6}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d={f.icon}
                  />
                </svg>
              </span>
              <h3 className="mt-4 text-lg font-semibold text-ink">{f.name}</h3>
              <p className="mt-2 text-sm text-ink-muted">{f.description}</p>
              <ul className="mt-4 space-y-2">
                {f.points.map((p) => (
                  <li
                    key={p}
                    className="flex items-start gap-2 text-sm text-ink-muted"
                  >
                    <svg
                      className="mt-0.5 h-4 w-4 flex-shrink-0 text-brand"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={2.5}
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M4.5 12.75l6 6 9-13.5"
                      />
                    </svg>
                    {p}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* Performance */}
      <section className="border-y border-line bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <SectionHeading
            center
            eyebrow="Track record"
            title="Backtested performance"
            subtitle="Historical risk-adjusted returns across the core strategy families."
          />
          <div className="mt-10 overflow-hidden rounded-2xl border border-line">
            <table className="min-w-full divide-y divide-line">
              <thead className="bg-surface-2">
                <tr>
                  {[
                    "Strategy",
                    "Sharpe",
                    "Max DD",
                    "Profit Factor",
                    "Win Rate",
                  ].map((h) => (
                    <th
                      key={h}
                      className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-ink-muted"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line bg-surface">
                {performanceMetrics.map((m) => (
                  <tr
                    key={m.strategy}
                    className="transition-colors hover:bg-surface-2"
                  >
                    <td className="px-6 py-4 text-sm font-medium text-ink">
                      {m.strategy}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-pos">
                      {m.sharpeRatio}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink-muted">
                      {m.maxDD}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink-muted">
                      {m.profitFactor}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink-muted">
                      {m.winRate}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* AI/ML core */}
      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div>
            <p className="am-eyebrow mb-3">AI / ML core</p>
            <h2 className="text-3xl font-bold text-ink sm:text-4xl">
              Models that learn the regime
            </h2>
            <p className="mt-4 text-base text-ink-muted">
              AlphaMind continuously adapts to shifting market conditions,
              discovering structure where traditional factor models go stale.
            </p>
            <ul className="mt-8 space-y-4">
              {[
                "Temporal Fusion Transformers for multi-horizon forecasting",
                "Reinforcement learning for portfolio optimization",
                "Generative models for market simulation",
                "Attention mechanisms over time-series structure",
                "Sentiment analysis across alternative data",
              ].map((item) => (
                <li key={item} className="flex items-start gap-3">
                  <span className="mt-1 grid h-5 w-5 flex-shrink-0 place-items-center rounded-full bg-brand-soft text-brand">
                    <svg
                      className="h-3 w-3"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={3}
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M4.5 12.75l6 6 9-13.5"
                      />
                    </svg>
                  </span>
                  <span className="text-ink">{item}</span>
                </li>
              ))}
            </ul>
            <Link to="/documentation" className="am-btn am-btn-ghost mt-8">
              Read the docs
            </Link>
          </div>
          <div className="am-card relative overflow-hidden p-1">
            <div className="am-grid-bg absolute inset-0 opacity-30" />
            <pre className="relative overflow-x-auto rounded-xl bg-[#080d18] p-6 font-mono text-[13px] leading-relaxed text-[#aeb9d0]">
              <code>{`# AlphaMind signal pipeline
from alphamind import Pipeline, TFT, RLAgent

pipe = Pipeline(
    data=["satellite", "sec_8k", "sentiment"],
    model=TFT(horizon=20, heads=8),
    sizing=RLAgent(reward="sharpe"),
)

signals = pipe.fit(universe="SP500").predict()
# -> Sharpe 2.4 - MaxDD 9% - PF 4.1`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-7xl px-4 pb-24 sm:px-6 lg:px-8">
        <div className="relative overflow-hidden rounded-3xl border border-line bg-[#080d18] px-8 py-16 text-center">
          <div className="am-grid-bg absolute inset-0 opacity-40" />
          <div
            className="pointer-events-none absolute left-1/2 top-0 h-72 w-72 -translate-x-1/2 rounded-full blur-3xl"
            style={{
              background:
                "radial-gradient(circle, rgba(99,102,241,0.4), transparent 70%)",
            }}
          />
          <div className="relative">
            <h2 className="font-display text-3xl font-bold text-white sm:text-4xl">
              Put the terminal to work
            </h2>
            <p className="mx-auto mt-4 max-w-lg text-[#aeb9d0]">
              Spin up the dashboard, wire your strategies, and start backtesting
              in minutes.
            </p>
            <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Link
                to="/signup"
                className="am-btn am-btn-primary px-7 py-3 text-base"
              >
                Create your account
              </Link>
              <a
                href="https://github.com/quantsingularity/AlphaMind"
                target="_blank"
                rel="noopener noreferrer"
                className="am-btn px-7 py-3 text-base text-white"
                style={{ border: "1px solid rgba(255,255,255,0.2)" }}
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
