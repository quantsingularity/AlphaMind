import type React from "react";

export const LogoMark: React.FC<{ size?: number; className?: string }> = ({
  size = 32,
  className = "",
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 64 64"
    fill="none"
    className={className}
    aria-hidden="true"
  >
    <defs>
      <linearGradient id="am-logo-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="var(--brand)" />
        <stop offset="1" stopColor="var(--accent)" />
      </linearGradient>
    </defs>
    <rect width="64" height="64" rx="15" fill="url(#am-logo-grad)" />
    <text
      x="32"
      y="45"
      fontFamily="Space Grotesk, sans-serif"
      fontSize="38"
      fontWeight="700"
      fill="#fff"
      textAnchor="middle"
    >
      α
    </text>
  </svg>
);

export const Wordmark: React.FC<{ className?: string }> = ({
  className = "",
}) => (
  <span
    className={`font-display text-xl font-bold tracking-tight ${className}`}
  >
    <span className="text-ink">Alpha</span>
    <span className="bg-gradient-to-r from-brand to-accent bg-clip-text text-transparent">
      Mind
    </span>
  </span>
);

export const BrandLockup: React.FC<{ size?: number }> = ({ size = 30 }) => (
  <span className="inline-flex items-center gap-2.5">
    <LogoMark size={size} />
    <Wordmark />
  </span>
);
