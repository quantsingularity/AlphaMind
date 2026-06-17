import type React from "react";
import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AuthScreen } from "../components/AuthScreen";
import { useAuth } from "../contexts/AuthContext";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export const SignIn: React.FC = () => {
  const { login, loginDemo } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string })?.from ?? "/dashboard";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!email.trim()) return setError("Enter your email address.");
    if (!EMAIL_RE.test(email.trim()))
      return setError("Enter a valid email address.");
    if (!password) return setError("Enter your password.");

    setSubmitting(true);
    try {
      await login({ email: email.trim(), password });
      navigate(from, { replace: true });
    } catch (err) {
      setError(
        (err as { message?: string })?.message ??
          "We couldn't sign you in. Check your details and try again.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleDemo = () => {
    loginDemo();
    navigate("/dashboard", { replace: true });
  };

  return (
    <AuthScreen title="Welcome back" subtitle="Sign in to your trading desk.">
      <form onSubmit={handleSubmit} className="space-y-4" noValidate>
        <div>
          <label
            htmlFor="email"
            className="mb-1.5 block text-sm font-medium text-ink"
          >
            Email
          </label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="am-input"
            placeholder="you@firm.com"
          />
        </div>

        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <label
              htmlFor="password"
              className="block text-sm font-medium text-ink"
            >
              Password
            </label>
            <button
              type="button"
              onClick={() => setShowPw((s) => !s)}
              className="text-xs font-medium text-brand hover:underline"
            >
              {showPw ? "Hide" : "Show"}
            </button>
          </div>
          <input
            id="password"
            type={showPw ? "text" : "password"}
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="am-input"
            placeholder="••••••••"
          />
        </div>

        {error && (
          <p
            className="rounded-lg bg-neg-soft px-3 py-2 text-sm text-neg"
            role="alert"
          >
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="am-btn am-btn-primary w-full"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <div className="my-6 flex items-center gap-3">
        <span className="h-px flex-1 bg-line" />
        <span className="text-xs text-ink-faint">or</span>
        <span className="h-px flex-1 bg-line" />
      </div>

      <button onClick={handleDemo} className="am-btn am-btn-ghost w-full">
        Continue with a demo account
      </button>

      <p className="mt-8 text-center text-sm text-ink-muted">
        New to AlphaMind?{" "}
        <Link to="/signup" className="font-semibold text-brand hover:underline">
          Create an account
        </Link>
      </p>
    </AuthScreen>
  );
};
