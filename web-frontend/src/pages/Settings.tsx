import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { useTheme } from "../contexts/ThemeContext";
import { Badge, SectionHeading } from "../components/ui";

type ThemeOption = "light" | "dark";

const THEME_OPTIONS: { value: ThemeOption; label: string; hint: string }[] = [
  { value: "light", label: "Light", hint: "Bright canvas for daytime desks" },
  {
    value: "dark",
    label: "Dark",
    hint: "Low-glare terminal for long sessions",
  },
];

export function Settings() {
  const { user, logout } = useAuth();
  const { theme, setTheme } = useTheme();
  const navigate = useNavigate();

  const [displayName, setDisplayName] = useState(user?.name ?? "");
  const [email] = useState(user?.email ?? "");
  const [savedAt, setSavedAt] = useState<string | null>(null);

  const [notifyFills, setNotifyFills] = useState(true);
  const [notifyRisk, setNotifyRisk] = useState(true);
  const [notifyDigest, setNotifyDigest] = useState(false);
  const [compactNumbers, setCompactNumbers] = useState(false);

  const initials = (user?.name ?? "Trader")
    .split(" ")
    .map((part) => part[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();

  const handleSaveProfile = () => {
    setSavedAt(
      new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    );
  };

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  return (
    <div className="space-y-8">
      <SectionHeading
        eyebrow="Workspace"
        title="Settings"
        subtitle="Manage your profile, appearance, and notification preferences."
      />

      <section className="am-card p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-display text-lg font-semibold text-ink">
              Profile
            </h2>
            <p className="mt-1 text-sm text-ink-muted">
              This information is shown across your AlphaMind workspace.
            </p>
          </div>
          {savedAt && <Badge tone="pos">Saved {savedAt}</Badge>}
        </div>

        <div className="mt-6 flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-soft font-display text-xl font-semibold text-brand">
            {initials}
          </div>
          <div>
            <p className="font-medium text-ink">
              {user?.name ?? "AlphaMind Trader"}
            </p>
            <p className="text-sm text-ink-muted">
              {email || "demo@alphamind.io"}
            </p>
          </div>
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium text-ink-muted">
              Display name
            </span>
            <input
              className="am-input"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              placeholder="Your name"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-sm font-medium text-ink-muted">
              Email
            </span>
            <input
              className="am-input"
              value={email}
              disabled
              placeholder="you@example.com"
            />
          </label>
        </div>

        <div className="mt-6">
          <button
            type="button"
            className="am-btn am-btn-primary"
            onClick={handleSaveProfile}
          >
            Save changes
          </button>
        </div>
      </section>

      <section className="am-card p-6">
        <h2 className="font-display text-lg font-semibold text-ink">
          Appearance
        </h2>
        <p className="mt-1 text-sm text-ink-muted">
          Choose how the terminal renders for you.
        </p>

        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {THEME_OPTIONS.map((option) => {
            const active = theme === option.value;
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => setTheme(option.value)}
                className={`rounded-xl border p-4 text-left transition ${
                  active
                    ? "border-brand bg-brand-soft"
                    : "border-line bg-surface-2 hover:border-line-strong"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-ink">{option.label}</span>
                  {active && <Badge tone="brand">Active</Badge>}
                </div>
                <p className="mt-1 text-sm text-ink-muted">{option.hint}</p>
              </button>
            );
          })}
        </div>
      </section>

      <section className="am-card p-6">
        <h2 className="font-display text-lg font-semibold text-ink">
          Notifications
        </h2>
        <p className="mt-1 text-sm text-ink-muted">
          Control what AlphaMind alerts you about.
        </p>

        <div className="mt-5 divide-y divide-line">
          <ToggleRow
            label="Order fills"
            hint="Notify when strategy orders execute"
            checked={notifyFills}
            onChange={setNotifyFills}
          />
          <ToggleRow
            label="Risk thresholds"
            hint="Alert when exposure or drawdown limits are breached"
            checked={notifyRisk}
            onChange={setNotifyRisk}
          />
          <ToggleRow
            label="Daily digest"
            hint="A morning summary of performance and signals"
            checked={notifyDigest}
            onChange={setNotifyDigest}
          />
        </div>
      </section>

      <section className="am-card p-6">
        <h2 className="font-display text-lg font-semibold text-ink">Display</h2>
        <p className="mt-1 text-sm text-ink-muted">
          Fine-tune how figures are presented.
        </p>

        <div className="mt-5 divide-y divide-line">
          <ToggleRow
            label="Compact numbers"
            hint="Abbreviate large values (1.2M instead of 1,200,000)"
            checked={compactNumbers}
            onChange={setCompactNumbers}
          />
        </div>
      </section>

      <section className="am-card border-neg/40 p-6">
        <h2 className="font-display text-lg font-semibold text-ink">Session</h2>
        <p className="mt-1 text-sm text-ink-muted">
          Sign out of this device. Your strategies and data remain safe.
        </p>
        <div className="mt-5">
          <button
            type="button"
            onClick={handleLogout}
            className="am-btn border border-neg/50 text-neg hover:bg-neg-soft"
          >
            Sign out
          </button>
        </div>
      </section>
    </div>
  );
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-4">
      <div>
        <p className="font-medium text-ink">{label}</p>
        <p className="text-sm text-ink-muted">{hint}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 shrink-0 rounded-full transition ${
          checked ? "bg-brand" : "bg-surface-3"
        }`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all ${
            checked ? "left-[22px]" : "left-0.5"
          }`}
        />
      </button>
    </div>
  );
}

export default Settings;
