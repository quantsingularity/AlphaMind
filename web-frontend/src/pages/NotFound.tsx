import type React from "react";
import { Link } from "react-router-dom";

export const NotFound: React.FC = () => {
  return (
    <div className="am-grid-bg grid min-h-screen place-items-center bg-canvas px-6 text-center">
      <div className="am-rise">
        <p className="font-display text-7xl font-bold text-brand">404</p>
        <h1 className="mt-4 font-display text-3xl font-semibold text-ink">
          Page not found
        </h1>
        <p className="mt-2 text-base text-ink-muted">
          The page you are looking for does not exist or has moved.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <Link to="/" className="am-btn am-btn-primary">
            Back to home
          </Link>
          <Link to="/dashboard" className="am-btn am-btn-ghost">
            Open dashboard
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
