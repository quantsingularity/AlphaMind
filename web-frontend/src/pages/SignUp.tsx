import type React from "react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AuthScreen } from "../components/AuthScreen";
import { useAuth } from "../contexts/AuthContext";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export const SignUp: React.FC = () => {
  const { register, loginDemo } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!name.trim()) return setError("Enter your full name.");
    if (!EMAIL_RE.test(email.trim()))
      return setError("Enter a valid email address.");
    if (password.length < 8)
      return setError("Use a password with at least 8 characters.");
    if (password !== confirm) return setError("Passwords do not match.");

    setSubmitting(true);
    try {
      await register({ name: name.trim(), email: email.trim(), password });
      navigate("/dashboard", { replace: true });
    } catch (err) {
      setError(
        (err as { message?: string })?.message ??
          "We couldn't create your account. Try again.",
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
    <AuthScreen
      title="Create your account"
      subtitle="Start building and backtesting strategies in minutes."
    >
      <form onSubmit={handleSubmit} className="space-y-4" noValidate>
        <div>
          <label
            htmlFor="name"
            className="mb-1.5 block text-sm font-medium text-ink"
          >
            Full name
          </label>
          <input
            id="name"
            type="text"
            autoComplete="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="am-input"
            placeholder="Ada Lovelace"
          />
        </div>

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
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="am-input"
            placeholder="At least 8 characters"
          />
        </div>

        <div>
          <label
            htmlFor="confirm"
            className="mb-1.5 block text-sm font-medium text-ink"
          >
            Confirm password
          </label>
          <input
            id="confirm"
            type={showPw ? "text" : "password"}
            autoComplete="new-password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            className="am-input"
            placeholder="Re-enter your password"
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
          {submitting ? "Creating account…" : "Create account"}
        </button>
      </form>

      <div className="my-6 flex items-center gap-3">
        <span className="h-px flex-1 bg-line" />
        <span className="text-xs text-ink-faint">or</span>
        <span className="h-px flex-1 bg-line" />
      </div>

      <button onClick={handleDemo} className="am-btn am-btn-ghost w-full">
        Explore with a demo account
      </button>

      <p className="mt-8 text-center text-sm text-ink-muted">
        Already have an account?{" "}
        <Link to="/signin" className="font-semibold text-brand hover:underline">
          Sign in
        </Link>
      </p>
    </AuthScreen>
  );
};
