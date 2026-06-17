import type React from "react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { BrandLockup } from "./Brand";
import { ThemeToggle } from "./ThemeToggle";

const highlights = [
  "Temporal Fusion Transformers for multi-horizon forecasting",
  "Reinforcement learning portfolio optimization",
  "Real-time alternative data and sentiment fusion",
  "Bayesian VaR with regime-aware stress testing",
];

export const AuthScreen: React.FC<{
  title: string;
  subtitle: string;
  children: ReactNode;
}> = ({ title, subtitle, children }) => (
  <div className="flex min-h-screen bg-canvas">
    {/* Brand panel */}
    <div className="relative hidden w-1/2 overflow-hidden bg-[#080d18] lg:block">
      <div className="am-grid-bg absolute inset-0 opacity-40" />
      <div
        className="absolute -left-24 top-1/3 h-96 w-96 rounded-full blur-3xl"
        style={{
          background:
            "radial-gradient(circle, rgba(99,102,241,0.35), transparent 70%)",
        }}
      />
      <div
        className="absolute -right-16 bottom-0 h-80 w-80 rounded-full blur-3xl"
        style={{
          background:
            "radial-gradient(circle, rgba(34,211,238,0.25), transparent 70%)",
        }}
      />
      <div className="relative flex h-full flex-col justify-between p-12">
        <Link to="/" className="flex items-center">
          <BrandLockup />
        </Link>
        <div>
          <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#22d3ee]">
            Quant Terminal
          </p>
          <h2 className="mt-4 max-w-md font-display text-4xl font-bold leading-tight text-white">
            Trade the signal, not the noise.
          </h2>
          <ul className="mt-8 space-y-3">
            {highlights.map((h) => (
              <li
                key={h}
                className="flex items-start gap-3 text-sm text-[#aeb9d0]"
              >
                <svg
                  className="mt-0.5 h-5 w-5 flex-shrink-0 text-[#22d3ee]"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={2}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4.5 12.75l6 6 9-13.5"
                  />
                </svg>
                {h}
              </li>
            ))}
          </ul>
        </div>
        <p className="font-mono text-xs text-[#5b6a87]">
          AlphaMind &copy; {new Date().getFullYear()}
        </p>
      </div>
    </div>

    {/* Form panel */}
    <div className="flex w-full flex-col lg:w-1/2">
      <div className="flex items-center justify-between p-6">
        <Link to="/" className="flex items-center lg:hidden">
          <BrandLockup size={26} />
        </Link>
        <span className="hidden lg:block" />
        <ThemeToggle />
      </div>
      <div className="flex flex-1 items-center justify-center px-6 pb-12">
        <div className="w-full max-w-sm">
          <h1 className="font-display text-3xl font-bold text-ink">{title}</h1>
          <p className="mt-2 text-sm text-ink-muted">{subtitle}</p>
          <div className="mt-8">{children}</div>
        </div>
      </div>
    </div>
  </div>
);
