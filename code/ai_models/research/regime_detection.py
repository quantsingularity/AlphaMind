"""Market regime detection using HMM and volatility-based classification."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]
REGIME_COLORS = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}
REGIME_LABELS = {0: "Bull", 1: "Neutral", 2: "Bear"}

SEED = 42
rng = np.random.default_rng(SEED)
pd.set_option("display.float_format", "{:.4f}".format)
print("Environment ready.")

N = 1512  # 6 years of daily data
dates = pd.bdate_range("2019-01-02", periods=N)

# Simulate a regime-switching return series
regime_params = {
    "bull": dict(mu=0.0008, sigma=0.008),
    "neutral": dict(mu=0.0001, sigma=0.012),
    "bear": dict(mu=-0.0010, sigma=0.022),
}

# Transition matrix (row = from, col = to)
P = np.array(
    [
        [0.985, 0.010, 0.005],  # from bull
        [0.030, 0.940, 0.030],  # from neutral
        [0.010, 0.040, 0.950],  # from bear
    ]
)


def simulate_regimes(n, P, rng):
    regime = np.zeros(n, dtype=int)
    regime[0] = 0
    for t in range(1, n):
        regime[t] = rng.choice(3, p=P[regime[t - 1]])
    return regime


true_regime = simulate_regimes(N, P, rng)
param_list = [regime_params["bull"], regime_params["neutral"], regime_params["bear"]]

returns = np.array(
    [rng.normal(param_list[r]["mu"], param_list[r]["sigma"]) for r in true_regime]
)

prices = 100 * np.cumprod(1 + returns)

mkt = pd.DataFrame(
    {
        "return": returns,
        "price": prices,
        "true_regime": true_regime,
    },
    index=dates,
)

print(f"Simulated {N} days. Regime distribution:")
for r, label in REGIME_LABELS.items():
    print(f"  {label}: {(true_regime == r).mean():.1%}")

mkt["vol_21d"] = mkt["return"].rolling(21).std() * np.sqrt(252)
mkt["vol_63d"] = mkt["return"].rolling(63).std() * np.sqrt(252)
mkt["mom_21d"] = mkt["price"].pct_change(21)
mkt["drawdown"] = mkt["price"] / mkt["price"].cummax() - 1

# Simple vol-based regime label (tertile)
vol_33 = mkt["vol_21d"].quantile(0.33)
vol_67 = mkt["vol_21d"].quantile(0.67)
mkt["vol_regime"] = 0  # low vol
mkt.loc[mkt["vol_21d"] > vol_33, "vol_regime"] = 1
mkt.loc[mkt["vol_21d"] > vol_67, "vol_regime"] = 2

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

axes[0].plot(mkt.index, mkt["price"], color=COLORS[0], linewidth=1.5)
axes[0].set_ylabel("Price")
axes[0].set_title("Simulated Market Price", fontweight="bold")

axes[1].plot(mkt.index, mkt["return"] * 100, color=COLORS[1], linewidth=0.8, alpha=0.7)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.7)
axes[1].set_ylabel("Daily Return (%)")
axes[1].set_title("Daily Returns", fontweight="bold")

axes[2].plot(
    mkt.index, mkt["vol_21d"] * 100, color=COLORS[2], linewidth=1.5, label="21d"
)
axes[2].plot(
    mkt.index,
    mkt["vol_63d"] * 100,
    color=COLORS[3],
    linewidth=1.5,
    linestyle="--",
    label="63d",
)
axes[2].axhline(vol_33 * 100, color="green", linestyle=":", linewidth=0.8)
axes[2].axhline(vol_67 * 100, color="orange", linestyle=":", linewidth=0.8)
axes[2].set_ylabel("Realised Vol (%)")
axes[2].set_title("Rolling Volatility with Regime Thresholds", fontweight="bold")
axes[2].legend()

axes[3].fill_between(mkt.index, mkt["drawdown"] * 100, 0, alpha=0.5, color=COLORS[1])
axes[3].set_ylabel("Drawdown (%)")
axes[3].set_title("Drawdown from Peak", fontweight="bold")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.tight_layout()
plt.show()

try:
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not installed. Using Gaussian mixture EM fallback.")

if HMM_AVAILABLE:
    X_hmm = mkt["return"].dropna().values.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=3, covariance_type="full", n_iter=200, random_state=SEED
    )
    hmm.fit(X_hmm)
    hmm_states = hmm.predict(X_hmm)

    # Re-label: sort by mean return (0=lowest, 2=highest)
    order = np.argsort(hmm.means_.ravel())
    remap = {old: new for new, old in enumerate(order)}
    hmm_states = np.array([remap[s] for s in hmm_states])
    mkt["hmm_regime"] = hmm_states

    print("HMM estimated means and stds:")
    for i, (mi, ci) in enumerate(
        zip(hmm.means_.ravel()[order], np.sqrt(hmm.covars_.ravel()[order]))
    ):
        label = REGIME_LABELS.get(i, str(i))
        print(f"  {label}: mean={mi:.5f}, std={ci:.5f}")
else:
    # Gaussian mixture fallback
    from sklearn.mixture import GaussianMixture

    X_hmm = mkt["return"].dropna().values.reshape(-1, 1)
    gm = GaussianMixture(
        n_components=3, covariance_type="full", n_init=10, random_state=SEED
    )
    gm.fit(X_hmm)
    states = gm.predict(X_hmm)
    order = np.argsort(gm.means_.ravel())
    remap = {old: new for new, old in enumerate(order)}
    mkt["hmm_regime"] = np.array([remap[s] for s in states])

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

for r, (label, color) in enumerate(zip(REGIME_LABELS.values(), REGIME_COLORS.values())):
    mask = mkt["true_regime"] == r
    axes[0].fill_between(
        mkt.index,
        0,
        1,
        where=mask,
        alpha=0.4,
        color=color,
        transform=axes[0].get_xaxis_transform(),
    )
axes[0].plot(
    mkt.index,
    mkt["price"] / mkt["price"].iloc[0],
    color="black",
    linewidth=1.2,
    label="Normalised Price",
)
axes[0].set_title("True Regime Labels", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Norm. Price")
patches = [
    mpatches.Patch(color=c, label=l)
    for l, c in zip(REGIME_LABELS.values(), REGIME_COLORS.values())
]
axes[0].legend(handles=patches, loc="upper left", fontsize=9)

if "hmm_regime" in mkt.columns:
    for r, color in REGIME_COLORS.items():
        mask = mkt["hmm_regime"] == r
        axes[1].fill_between(
            mkt.index,
            0,
            1,
            where=mask,
            alpha=0.4,
            color=color,
            transform=axes[1].get_xaxis_transform(),
        )
    axes[1].plot(
        mkt.index, mkt["price"] / mkt["price"].iloc[0], color="black", linewidth=1.2
    )
    axes[1].set_title("HMM-Detected Regimes", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Norm. Price")
    axes[1].legend(handles=patches, loc="upper left", fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()


# CUSUM-based change point detection (no ruptures dependency)
def cusum_changepoints(series, threshold=4.0, min_segment=21):
    """
    Simple CUSUM change-point detection.
    Returns list of detected change-point indices.
    """
    z = (series - series.mean()) / series.std()
    cp = []
    t = 0
    while t < len(z):
        cusum_pos = 0.0
        cusum_neg = 0.0
        for i in range(t, len(z)):
            cusum_pos = max(0, cusum_pos + z.iloc[i] - 0.5)
            cusum_neg = max(0, cusum_neg - z.iloc[i] - 0.5)
            if cusum_pos > threshold or cusum_neg > threshold:
                if i - t >= min_segment:
                    cp.append(i)
                    t = i + 1
                    break
        else:
            break
    return cp


vol_series = mkt["vol_21d"].dropna()
cp_indices = cusum_changepoints(vol_series, threshold=5.0, min_segment=21)
cp_dates = vol_series.index[cp_indices]

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(
    vol_series.index, vol_series * 100, color=COLORS[0], linewidth=1.5, label="21d Vol"
)
for cd in cp_dates:
    ax.axvline(cd, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

ax.set_ylabel("Annualised Volatility (%)")
ax.set_title(
    f"CUSUM Change-Point Detection on Volatility ({len(cp_dates)} points detected)",
    fontsize=12,
    fontweight="bold",
)
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()

print(f"Detected {len(cp_dates)} change points:")
for d in cp_dates:
    print(f"  {d.date()}")

regime_col = "hmm_regime" if "hmm_regime" in mkt.columns else "vol_regime"
mkt_clean = mkt.dropna()

regime_stats = mkt_clean.groupby(regime_col)["return"].agg(
    Observations="count",
    Mean=lambda x: x.mean() * 252,
    Volatility=lambda x: x.std() * np.sqrt(252),
    Skewness="skew",
    Kurtosis=lambda x: x.kurt(),
    Min=lambda x: x.min(),
    Max=lambda x: x.max(),
)
regime_stats.index = [REGIME_LABELS.get(i, str(i)) for i in regime_stats.index]
regime_stats["Sharpe"] = regime_stats["Mean"] / regime_stats["Volatility"]
regime_stats["Hit Rate"] = (
    mkt_clean.groupby(regime_col)["return"].apply(lambda x: (x > 0).mean()).values
)

print("Regime-Conditional Return Statistics (Annualised where applicable):")
print(regime_stats.to_string())

from_states = mkt_clean[regime_col].values[:-1]
to_states = mkt_clean[regime_col].values[1:]

n_regimes = 3
trans = np.zeros((n_regimes, n_regimes))
for f, t in zip(from_states, to_states):
    trans[f, t] += 1
trans_prob = trans / trans.sum(axis=1, keepdims=True)

labels = [REGIME_LABELS[i] for i in range(n_regimes)]
trans_df = pd.DataFrame(trans_prob, index=labels, columns=labels)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    trans_df,
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Transition Probability"},
)
ax.set_xlabel("To Regime")
ax.set_ylabel("From Regime")
ax.set_title(
    "Empirical Regime Transition Probability Matrix", fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.show()

# Expected duration per regime
durations = {labels[i]: 1 / (1 - trans_prob[i, i]) for i in range(n_regimes)}
print("Expected regime durations (days):")
for label, dur in durations.items():
    print(f"  {label}: {dur:.1f} days")

# Simulate a momentum signal and measure its IC in each regime
sig = mkt["return"].shift(5).rolling(21).mean()
ret5 = mkt["return"].rolling(5).sum().shift(-5)
ic_df = pd.DataFrame(
    {"signal": sig, "fwd_5d": ret5, "regime": mkt[regime_col]}
).dropna()

regime_ic = {}
for r in range(n_regimes):
    sub = ic_df[ic_df["regime"] == r]
    if len(sub) > 10:
        ic, pval = stats.spearmanr(sub["signal"], sub["fwd_5d"])
        regime_ic[REGIME_LABELS[r]] = {"IC": ic, "p-value": pval, "N": len(sub)}

ic_cond = pd.DataFrame(regime_ic).T
print("Momentum Signal IC by Regime:")
print(ic_cond.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
colors_r = [REGIME_COLORS[r] for r in range(n_regimes)]
ic_cond["IC"].plot.bar(ax=ax, color=colors_r, edgecolor="white", width=0.5)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title(
    "Momentum Signal IC Conditional on Market Regime", fontsize=12, fontweight="bold"
)
ax.set_ylabel("Spearman IC")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

# Trade the signal only in favourable regimes (Bull or Neutral)
# In Bear regime, hold cash or go flat

ic_df["signal_scaled"] = ic_df["signal"] / ic_df["signal"].rolling(63).std()

strategy_rets = []
baseline_rets = []
for idx, row in ic_df.iterrows():
    r = row["regime"]
    sig_val = row["signal_scaled"] if r in [0, 1] else 0.0
    pos = np.sign(sig_val) * 0.5  # 50% max position
    fwd = row["fwd_5d"]
    strategy_rets.append(pos * fwd / 5)  # daily approx
    baseline_rets.append(fwd / 5)

strat = pd.Series(strategy_rets, index=ic_df.index)
base = pd.Series(baseline_rets, index=ic_df.index)

cum_strat = (1 + strat).cumprod()
cum_base = (1 + base).cumprod()

fig, axes = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

axes[0].plot(
    cum_strat.index, cum_strat, color=COLORS[0], linewidth=2, label="Regime-Adaptive"
)
axes[0].plot(
    cum_base.index,
    cum_base,
    color=COLORS[1],
    linewidth=1.5,
    linestyle="--",
    label="Always-On Baseline",
)
axes[0].set_title(
    "Regime-Adaptive vs Always-On Strategy", fontsize=12, fontweight="bold"
)
axes[0].set_ylabel("Cumulative Return")
axes[0].legend()

dd = cum_strat / cum_strat.cummax() - 1
axes[1].fill_between(dd.index, dd, 0, alpha=0.5, color=COLORS[1])
axes[1].set_ylabel("Drawdown")
axes[1].set_title("Strategy Drawdown", fontweight="bold")
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()


def perf_stats(rets, label):
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sr = ann_ret / ann_vol if ann_vol > 0 else 0
    mdd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
    print(
        f"  {label:25s}  Ret={ann_ret:.2%}  Vol={ann_vol:.2%}  SR={sr:.2f}  MDD={mdd:.2%}"
    )


print("Performance Summary:")
perf_stats(strat, "Regime-Adaptive")
perf_stats(base, "Always-On Baseline")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# Price with regime shading
ax1 = fig.add_subplot(gs[0, :2])
regime_col_plot = "hmm_regime" if "hmm_regime" in mkt.columns else "vol_regime"
for r, color in REGIME_COLORS.items():
    mask = mkt[regime_col_plot] == r
    ax1.fill_between(
        mkt.index,
        0,
        1,
        where=mask,
        alpha=0.3,
        color=color,
        transform=ax1.get_xaxis_transform(),
    )
ax1.plot(mkt.index, mkt["price"] / mkt["price"].iloc[0], color="black", linewidth=1.2)
ax1.set_title("Price with Detected Regimes", fontweight="bold")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Transition matrix
ax2 = fig.add_subplot(gs[0, 2])
sns.heatmap(
    trans_df,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.3,
    ax=ax2,
    cbar=False,
    annot_kws={"size": 9},
)
ax2.set_title("Transition Probabilities", fontweight="bold", fontsize=10)

# Regime vol
ax3 = fig.add_subplot(gs[1, 0])
regime_stats["Volatility"].plot.bar(
    ax=ax3, color=list(REGIME_COLORS.values()), edgecolor="white"
)
ax3.set_title("Volatility by Regime", fontweight="bold", fontsize=10)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax3.tick_params(axis="x", rotation=0)

# Regime mean return
ax4 = fig.add_subplot(gs[1, 1])
regime_stats["Mean"].plot.bar(
    ax=ax4, color=list(REGIME_COLORS.values()), edgecolor="white"
)
ax4.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax4.set_title("Mean Annual Return by Regime", fontweight="bold", fontsize=10)
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax4.tick_params(axis="x", rotation=0)

# Regime Sharpe
ax5 = fig.add_subplot(gs[1, 2])
regime_stats["Sharpe"].plot.bar(
    ax=ax5, color=list(REGIME_COLORS.values()), edgecolor="white"
)
ax5.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax5.set_title("Sharpe by Regime", fontweight="bold", fontsize=10)
ax5.tick_params(axis="x", rotation=0)

# Cumulative returns
ax6 = fig.add_subplot(gs[2, :])
ax6.plot(
    cum_strat.index, cum_strat, color=COLORS[0], linewidth=2, label="Regime-Adaptive"
)
ax6.plot(
    cum_base.index,
    cum_base,
    color=COLORS[1],
    linewidth=1.5,
    linestyle="--",
    label="Baseline",
)
ax6.set_title("Strategy Cumulative Performance", fontweight="bold", fontsize=11)
ax6.set_ylabel("Cumulative Return")
ax6.legend()
ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.suptitle(
    "AlphaMind - Market Regime Detection Dashboard",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)
plt.show()
