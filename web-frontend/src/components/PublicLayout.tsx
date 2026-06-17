import type React from "react";
import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { BrandLockup } from "./Brand";
import { ThemeToggle } from "./ThemeToggle";

const publicNav = [
  { name: "Platform", to: "/" },
  { name: "Documentation", to: "/documentation" },
  { name: "Research", to: "/research" },
  { name: "About", to: "/about" },
];

export const PublicLayout: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const [open, setOpen] = useState(false);

  return (
    <div className="flex min-h-screen flex-col bg-canvas">
      <header className="sticky top-0 z-40 border-b border-line bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center">
            <BrandLockup />
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {publicNav.map((item) => (
              <NavLink
                key={item.name}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    isActive ? "text-ink" : "text-ink-muted hover:text-ink"
                  }`
                }
              >
                {item.name}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            {isAuthenticated ? (
              <Link
                to="/dashboard"
                className="am-btn am-btn-primary hidden sm:inline-flex"
              >
                Open dashboard
              </Link>
            ) : (
              <>
                <Link
                  to="/signin"
                  className="hidden rounded-lg px-3 py-2 text-sm font-semibold text-ink-muted transition-colors hover:text-ink sm:inline-flex"
                >
                  Sign in
                </Link>
                <Link to="/signup" className="am-btn am-btn-primary">
                  Get started
                </Link>
              </>
            )}
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-line-strong text-ink-muted md:hidden"
              aria-label="Toggle menu"
              aria-expanded={open}
            >
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.8}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d={
                    open
                      ? "M6 18L18 6M6 6l12 12"
                      : "M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
                  }
                />
              </svg>
            </button>
          </div>
        </div>

        {open && (
          <div className="border-t border-line bg-surface md:hidden">
            <nav className="space-y-1 px-4 py-3">
              {publicNav.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.to}
                  end={item.to === "/"}
                  onClick={() => setOpen(false)}
                  className={({ isActive }) =>
                    `block rounded-lg px-3 py-2 text-sm font-medium ${
                      isActive ? "bg-brand-soft text-brand" : "text-ink-muted"
                    }`
                  }
                >
                  {item.name}
                </NavLink>
              ))}
              {!isAuthenticated && (
                <Link
                  to="/signin"
                  onClick={() => setOpen(false)}
                  className="block rounded-lg px-3 py-2 text-sm font-semibold text-ink-muted"
                >
                  Sign in
                </Link>
              )}
            </nav>
          </div>
        )}
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-line bg-surface">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div>
              <BrandLockup size={26} />
              <p className="mt-3 max-w-sm text-sm text-ink-muted">
                Institutional-grade quantitative trading powered by alternative
                data, machine learning, and high-frequency execution.
              </p>
            </div>
            <div className="flex items-center gap-6">
              <Link
                to="/documentation"
                className="text-sm text-ink-muted hover:text-ink"
              >
                Docs
              </Link>
              <Link
                to="/about"
                className="text-sm text-ink-muted hover:text-ink"
              >
                About
              </Link>
              <a
                href="https://github.com/quantsingularity/AlphaMind"
                target="_blank"
                rel="noopener noreferrer"
                className="text-ink-muted hover:text-ink"
                aria-label="GitHub"
              >
                <svg
                  className="h-5 w-5"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    fillRule="evenodd"
                    clipRule="evenodd"
                    d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  />
                </svg>
              </a>
            </div>
          </div>
          <div className="mt-8 border-t border-line pt-6 text-sm text-ink-faint">
            &copy; {new Date().getFullYear()} AlphaMind. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
};
