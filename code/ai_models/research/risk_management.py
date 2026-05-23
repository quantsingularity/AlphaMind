"""Risk management analytics: VaR, CVaR, drawdown analysis, and stress testing."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import genextreme, norm
from scipy.stats import t as student_t

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]
pd.set_option("display.float_format", "{:.5f}".format)

SEED = 42
rng = np.random.default_rng(SEED)
print("Risk Management environment ready.")

N = 1260
N_PORT = 5  # 5 strategies / portfolios
dates = pd.bdate_range("2020-01-02", periods=N)

# Simulate realistic fat-tailed portfolio returns (Student-t with nu=5)
NU = 5
port_mu = np.array([0.0004, 0.0003, 0.0002, 0.0001, 0.00015])
port_sd = np.array([0.010, 0.012, 0.008, 0.015, 0.009])

raw = student_t.rvs(df=NU, size=(N, N_PORT), random_state=SEED)
raw = raw / np.sqrt(NU / (NU - 2))  # normalise to unit variance

port_rets = pd.DataFrame(
    raw * port_sd + port_mu,
    index=dates,
    columns=[f"Port_{i+1}" for i in range(N_PORT)],
)

# Equal-weight combined portfolio
port_rets["Combined"] = port_rets.mean(axis=1)

cum = (1 + port_rets).cumprod()

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
for col, color in zip(port_rets.columns, COLORS):
    axes[0].plot(cum.index, cum[col], linewidth=1.4, label=col, color=color)
axes[0].set_title("Cumulative Portfolio Returns", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Cumulative Return")
axes[0].legend(ncol=3, fontsize=9)

axes[1].plot(
    port_rets.index,
    port_rets["Combined"] * 100,
    color=COLORS[1],
    linewidth=0.8,
    alpha=0.8,
)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.7)
axes[1].set_ylabel("Daily Return (%)")
axes[1].set_title("Combined Portfolio Daily Returns", fontweight="bold")
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()


def var_historical(rets, alpha=0.01):
    """Historical simulation VaR."""
    return -np.quantile(rets, alpha)


def var_gaussian(rets, alpha=0.01):
    """Parametric Gaussian VaR."""
    mu, sigma = rets.mean(), rets.std()
    return -(mu + sigma * norm.ppf(alpha))


def var_student_t(rets, alpha=0.01):
    """Parametric Student-t VaR (fitted nu)."""
    nu, loc, scale = student_t.fit(rets, floc=rets.mean())
    return -(loc + scale * student_t.ppf(alpha, df=nu))


def var_ewma(rets, alpha=0.01, lam=0.94):
    """EWMA volatility scaled VaR."""
    var_t = rets.var()
    for r in rets:
        var_t = lam * var_t + (1 - lam) * r**2
    sigma_ewma = np.sqrt(var_t)
    return -(rets.mean() + sigma_ewma * norm.ppf(alpha))


ALPHAS = [0.05, 0.01, 0.005]
methods = {
    "Historical": var_historical,
    "Gaussian": var_gaussian,
    "Student-t": var_student_t,
    "EWMA": var_ewma,
}

target = port_rets["Combined"].values
var_tbl = {}
for alpha in ALPHAS:
    row = {m: fn(target, alpha) for m, fn in methods.items()}
    var_tbl[f"{(1-alpha):.0%} VaR"] = row

var_df = pd.DataFrame(var_tbl).T

print("1-Day VaR Estimates - Combined Portfolio:")
print(var_df.to_string())


def cvar_historical(rets, alpha=0.01):
    """Historical CVaR (Expected Shortfall)."""
    var = np.quantile(rets, alpha)
    return -rets[rets <= var].mean()


def cvar_gaussian(rets, alpha=0.01):
    """Parametric Gaussian CVaR."""
    mu, sigma = rets.mean(), rets.std()
    return -(mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha)


def cvar_student_t(rets, alpha=0.01):
    """Parametric Student-t CVaR."""
    nu, loc, scale = student_t.fit(rets, floc=rets.mean())
    q = student_t.ppf(alpha, df=nu)
    es = -(loc + scale * (student_t.pdf(q, df=nu) / alpha) * (nu + q**2) / (nu - 1))
    return es


cvar_tbl = {}
for alpha in ALPHAS:
    row = {
        "Historical CVaR": cvar_historical(target, alpha),
        "Gaussian CVaR": cvar_gaussian(target, alpha),
        "Student-t CVaR": cvar_student_t(target, alpha),
        "VaR (Hist)": var_historical(target, alpha),
    }
    cvar_tbl[f"{(1-alpha):.0%}"] = row

cvar_df = pd.DataFrame(cvar_tbl).T
print("CVaR vs VaR Comparison - Combined Portfolio:")
print(cvar_df.to_string())

# VaR vs CVaR visualisation
r = target
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(r * 100, bins=80, density=True, color=COLORS[0], alpha=0.6, edgecolor="white")

var99 = var_historical(r, 0.01)
cvar99 = cvar_historical(r, 0.01)
ax.axvline(-var99 * 100, color=COLORS[1], linewidth=2, label=f"99% VaR = {var99:.3%}")
ax.axvline(
    -cvar99 * 100,
    color=COLORS[2],
    linewidth=2,
    linestyle="--",
    label=f"99% CVaR = {cvar99:.3%}",
)

ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Density")
ax.set_title(
    "Return Distribution with 99% VaR and CVaR", fontsize=12, fontweight="bold"
)
ax.legend()
plt.tight_layout()
plt.show()


def drawdown_series(rets):
    cum = (1 + pd.Series(rets)).cumprod()
    return cum / cum.cummax() - 1


def drawdown_stats(rets):
    dd = drawdown_series(rets)
    mdd = dd.min()
    # Duration of max drawdown
    end_idx = dd.idxmin()
    peak_idx = dd[:end_idx][(dd[:end_idx] == 0)].index
    peak_idx = peak_idx[-1] if len(peak_idx) > 0 else dd.index[0]
    duration = (end_idx - peak_idx).days
    # Recovery: first return to 0 after trough
    after = dd[dd.index >= end_idx]
    recov = after[after == 0]
    recov_days = (recov.index[0] - end_idx).days if len(recov) > 0 else None
    return mdd, duration, recov_days


fig, axes = plt.subplots(N_PORT + 1, 1, figsize=(16, 3.5 * (N_PORT + 1)), sharex=True)

for ax, (col, color) in zip(axes, zip(port_rets.columns, COLORS)):
    dd = drawdown_series(port_rets[col].values)
    dd.index = port_rets.index
    ax.fill_between(dd.index, dd * 100, 0, alpha=0.6, color=color)
    mdd, dur, rec = drawdown_stats(port_rets[col])
    ax.set_title(
        f"{col} - MDD={mdd:.2%}  Duration={dur}d  Recovery={rec}d",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.suptitle(
    "Drawdown Analysis - All Portfolios", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.show()


# Fit Generalised Extreme Value (GEV) distribution to minimum returns block
def block_minima(rets, block_size=21):
    n_blocks = len(rets) // block_size
    return np.array(
        [rets[i * block_size : (i + 1) * block_size].min() for i in range(n_blocks)]
    )


minima = block_minima(port_rets["Combined"].values)
c, loc, scale = genextreme.fit(-minima)  # fit to negated minima (maxima)

print(
    f"GEV fit parameters:  xi (shape)={c:.4f}  mu (loc)={loc:.4f}  sigma (scale)={scale:.4f}"
)
if c > 0:
    print("Heavy-tailed (Frechet class) - consistent with equity returns.")
elif c < 0:
    print("Bounded upper tail (Weibull class).")
else:
    print("Gumbel class (thin tail).")

# Return levels
return_periods = [2, 5, 10, 25, 50]
quantiles = [1 - 1 / rp for rp in return_periods]
return_levels = genextreme.ppf(quantiles, c, loc=loc, scale=scale)

rl_df = pd.DataFrame(
    {
        "Return Period (years)": return_periods,
        "1-Month Tail Loss (%)": return_levels * 100,
    }
).set_index("Return Period (years)")
print("\nGEV Return Levels:")
print(rl_df.to_string())

# Historical and hypothetical stress scenarios
scenarios = {
    "2008 GFC (peak drawdown)": -0.45,
    "2020 COVID Crash (Feb-Mar)": -0.34,
    "1987 Black Monday": -0.22,
    "2000 Dot-Com Peak-to-Trough": -0.49,
    "Rate Shock (+200bps in 1 month)": -0.08,
    "Liquidity Crisis (-50% volume)": -0.12,
    "Currency Crisis (-15% FX)": -0.09,
}

initial_nav = 1_000_000
stress_rows = []
for scenario, shock in scenarios.items():
    loss_abs = initial_nav * abs(shock)
    loss_pct = shock
    days_to_recover = (
        abs(shock) / port_rets["Combined"].mean()
        if port_rets["Combined"].mean() > 0
        else np.inf
    )
    stress_rows.append(
        {
            "Scenario": scenario,
            "Portfolio Loss (%)": loss_pct,
            "Loss ($)": -loss_abs,
            "Est. Recovery (days)": min(int(days_to_recover), 9999),
        }
    )

stress_df = pd.DataFrame(stress_rows).set_index("Scenario")

fig, ax = plt.subplots(figsize=(12, 6))
stress_df["Portfolio Loss (%)"].mul(100).sort_values().plot.barh(
    ax=ax, color=COLORS[1], edgecolor="white", alpha=0.8
)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title(
    "Stress Test Scenarios - Estimated Portfolio Impact", fontsize=12, fontweight="bold"
)
ax.set_xlabel("Portfolio Loss (%)")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()

print(stress_df.to_string())

# Compare correlations in normal vs tail regimes
combined = port_rets["Combined"]
threshold = combined.quantile(0.05)

normal_mask = combined >= threshold
stress_mask = combined < threshold

corr_normal = port_rets.drop(columns="Combined")[normal_mask].corr()
corr_stress = port_rets.drop(columns="Combined")[stress_mask].corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
kw = dict(
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.4,
    annot_kws={"size": 9},
)
sns.heatmap(corr_normal, ax=axes[0], **kw, cbar=False)
axes[0].set_title(
    f"Normal Regime Correlations\n(n={normal_mask.sum()} days)",
    fontsize=11,
    fontweight="bold",
)

sns.heatmap(corr_stress, ax=axes[1], **kw)
axes[1].set_title(
    f"Stress Regime Correlations (bottom 5%)\n(n={stress_mask.sum()} days)",
    fontsize=11,
    fontweight="bold",
)

plt.suptitle(
    "Correlation Breakdown Under Stress - Diversification Reduction",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

diff = corr_stress - corr_normal
print("Average correlation increase in stress vs normal regime:")
print(f"  {diff.values[np.triu_indices_from(diff.values, k=1)].mean():.3f}")


def kelly_fraction(mu, sigma, risk_free=0.0):
    """Full Kelly fraction for a single asset."""
    excess = mu - risk_free
    return excess / (sigma**2) if sigma > 0 else 0.0


def half_kelly(mu, sigma, risk_free=0.0):
    return 0.5 * kelly_fraction(mu, sigma, risk_free)


kelly_rows = []
for col in port_rets.columns:
    r = port_rets[col]
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    fk = kelly_fraction(mu, sd)
    hk = half_kelly(mu, sd)
    kelly_rows.append(
        {
            "Portfolio": col,
            "Ann. Return": mu,
            "Ann. Vol": sd,
            "Sharpe": mu / sd if sd > 0 else 0,
            "Full Kelly": fk,
            "Half Kelly": hk,
            "Quarter Kelly": fk * 0.25,
        }
    )

kelly_df = pd.DataFrame(kelly_rows).set_index("Portfolio")

fig, ax = plt.subplots(figsize=(12, 5))
kelly_df[["Full Kelly", "Half Kelly", "Quarter Kelly"]].plot.bar(
    ax=ax, color=COLORS[:3], edgecolor="white", alpha=0.8, width=0.6
)
ax.axhline(
    1.0, color="red", linestyle="--", linewidth=0.8, label="100% (fully levered)"
)
ax.set_title(
    "Kelly Criterion Position Sizing by Strategy", fontsize=12, fontweight="bold"
)
ax.set_ylabel("Capital Fraction")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

print(kelly_df.to_string())

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# Cumulative returns
ax1 = fig.add_subplot(gs[0, :2])
for col, color in zip(port_rets.columns, COLORS):
    cum_p = (1 + port_rets[col]).cumprod()
    ax1.plot(cum_p.index, cum_p, linewidth=1.4, label=col, color=color)
ax1.set_title("Cumulative Returns", fontweight="bold")
ax1.legend(fontsize=8, ncol=3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Return distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(
    port_rets["Combined"] * 100,
    bins=60,
    density=True,
    color=COLORS[0],
    alpha=0.7,
    edgecolor="white",
)
x = np.linspace(
    port_rets["Combined"].min() * 100, port_rets["Combined"].max() * 100, 200
)
ax2.plot(
    x,
    norm.pdf(x, port_rets["Combined"].mean() * 100, port_rets["Combined"].std() * 100),
    color="red",
    linewidth=1.5,
    label="Normal fit",
)
ax2.axvline(
    -var99 * 100, color="orange", linewidth=1.5, linestyle="--", label=f"99% VaR"
)
ax2.set_title("Return Distribution", fontweight="bold")
ax2.legend(fontsize=8)

# Drawdown combined
ax3 = fig.add_subplot(gs[1, :2])
dd_comb = drawdown_series(port_rets["Combined"].values)
dd_comb.index = port_rets.index
ax3.fill_between(dd_comb.index, dd_comb * 100, 0, alpha=0.6, color=COLORS[1])
ax3.set_title("Combined Portfolio Drawdown", fontweight="bold")
ax3.set_ylabel("Drawdown (%)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Kelly sizing
ax4 = fig.add_subplot(gs[1, 2])
kelly_df["Half Kelly"].plot.bar(
    ax=ax4, color=COLORS[: len(kelly_df)], edgecolor="white"
)
ax4.set_title("Half Kelly Sizing", fontweight="bold")
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax4.tick_params(axis="x", rotation=30)

# VaR comparison bar
ax5 = fig.add_subplot(gs[2, :2])
var_methods = {
    "Hist.": var_historical(target, 0.01),
    "Gaussian": var_gaussian(target, 0.01),
    "Student-t": var_student_t(target, 0.01),
    "CVaR Hist.": cvar_historical(target, 0.01),
}
pd.Series(var_methods).mul(100).plot.bar(
    ax=ax5, color=COLORS[:4], edgecolor="white", alpha=0.85
)
ax5.set_title("99% VaR / CVaR Estimates - Combined Portfolio", fontweight="bold")
ax5.set_ylabel("Loss (%)")
ax5.tick_params(axis="x", rotation=15)

# Stress test
ax6 = fig.add_subplot(gs[2, 2])
stress_df["Portfolio Loss (%)"].mul(100).sort_values().plot.barh(
    ax=ax6, color=COLORS[1], edgecolor="white", alpha=0.8
)
ax6.set_title("Stress Scenarios", fontweight="bold")
ax6.tick_params(axis="y", labelsize=7)
ax6.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.suptitle(
    "AlphaMind - Risk Management Dashboard", fontsize=15, fontweight="bold", y=1.01
)
plt.show()
