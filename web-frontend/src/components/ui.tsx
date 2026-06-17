import type React from "react";
import type { ReactNode } from "react";

export const Spinner: React.FC<{ label?: string; full?: boolean }> = ({
  label = "Loading",
  full = true,
}) => (
  <div
    className={`flex flex-col items-center justify-center gap-3 ${full ? "h-64" : "py-10"}`}
    role="status"
  >
    <span className="h-9 w-9 animate-spin rounded-full border-2 border-line border-t-brand" />
    <span className="text-sm text-ink-muted">{label}</span>
  </div>
);

type Trend = "up" | "down" | "flat";

export const StatCard: React.FC<{
  label: string;
  value: ReactNode;
  hint?: string;
  trend?: Trend;
  icon?: ReactNode;
}> = ({ label, value, hint, trend = "flat", icon }) => {
  const trendColor =
    trend === "up"
      ? "text-pos"
      : trend === "down"
        ? "text-neg"
        : "text-ink-muted";
  return (
    <div className="am-card p-5">
      <div className="flex items-start justify-between">
        <p className="text-xs font-medium uppercase tracking-wider text-ink-muted">
          {label}
        </p>
        {icon && (
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-soft text-brand">
            {icon}
          </span>
        )}
      </div>
      <p className="mt-3 font-mono text-2xl font-semibold tracking-tight text-ink">
        {value}
      </p>
      {hint && (
        <p className={`mt-1 text-sm font-medium ${trendColor}`}>{hint}</p>
      )}
    </div>
  );
};

export const Badge: React.FC<{
  children: ReactNode;
  tone?: "brand" | "pos" | "neg" | "warn" | "neutral";
}> = ({ children, tone = "neutral" }) => {
  const tones: Record<string, string> = {
    brand: "bg-brand-soft text-brand",
    pos: "bg-pos-soft text-pos",
    neg: "bg-neg-soft text-neg",
    warn: "bg-warn-soft text-warn",
    neutral: "bg-surface-2 text-ink-muted",
  };
  return <span className={`am-badge ${tones[tone]}`}>{children}</span>;
};

export const SectionHeading: React.FC<{
  eyebrow?: string;
  title: string;
  subtitle?: string;
  center?: boolean;
}> = ({ eyebrow, title, subtitle, center }) => (
  <div className={center ? "text-center" : ""}>
    {eyebrow && <p className="am-eyebrow mb-3">{eyebrow}</p>}
    <h2 className="text-3xl font-bold text-ink sm:text-4xl">{title}</h2>
    {subtitle && (
      <p
        className={`mt-3 text-base text-ink-muted ${center ? "mx-auto max-w-2xl" : "max-w-2xl"}`}
      >
        {subtitle}
      </p>
    )}
  </div>
);

export const ErrorState: React.FC<{
  message: string;
  onRetry?: () => void;
}> = ({ message, onRetry }) => (
  <div className="am-card border-neg/40 bg-neg-soft p-5">
    <div className="flex items-start gap-3">
      <svg
        className="mt-0.5 h-5 w-5 flex-shrink-0 text-neg"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth={1.8}
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
        />
      </svg>
      <div className="flex-1">
        <p className="text-sm text-ink">{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="mt-3 text-sm font-semibold text-brand hover:underline"
          >
            Try again
          </button>
        )}
      </div>
    </div>
  </div>
);

export const EmptyState: React.FC<{ title: string; hint?: string }> = ({
  title,
  hint,
}) => (
  <div className="py-12 text-center">
    <p className="text-sm font-medium text-ink">{title}</p>
    {hint && <p className="mt-1 text-sm text-ink-muted">{hint}</p>}
  </div>
);
