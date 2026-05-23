"""Multi-factor model analysis: construction, attribution, and factor zoo comparison."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = [
    "#2563eb",
    "#ef4444",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#06b6d4",
    "#ec4899",
    "#6b7280",
]
sns.set_palette(COLORS)
pd.set_option("display.float_format", "{:.4f}".format)

SEED = 42
N_ASSETS = 300
N_DAYS = 756  # 3 years
rng = np.random.default_rng(SEED)

FACTORS = [
    "momentum_12_1",
    "size",
    "value_bp",
    "quality_roe",
    "low_volatility",
    "profitability",
    "investment",
    "liquidity",
]
print(f"Factor universe: {FACTORS}")
print(f"Assets: {N_ASSETS}  |  Days: {N_DAYS}")

dates = pd.bdate_range("2021-01-04", periods=N_DAYS)
assets = [f"STOCK_{i:04d}" for i in range(N_ASSETS)]

# Factor correlation structure (realistic inter-factor correlations)
corr_seed = np.array(
    [
        [1.00, -0.25, 0.05, -0.10, -0.30, 0.05, -0.05, 0.45],
        [-0.25, 1.00, -0.30, -0.10, 0.15, -0.20, -0.15, -0.50],
        [0.05, -0.30, 1.00, 0.30, 0.20, 0.35, 0.25, -0.10],
        [-0.10, -0.10, 0.30, 1.00, 0.10, 0.60, 0.30, -0.05],
        [-0.30, 0.15, 0.20, 0.10, 1.00, 0.10, -0.05, -0.20],
        [0.05, -0.20, 0.35, 0.60, 0.10, 1.00, 0.40, -0.10],
        [-0.05, -0.15, 0.25, 0.30, -0.05, 0.40, 1.00, -0.05],
        [0.45, -0.50, -0.10, -0.05, -0.20, -0.10, -0.05, 1.00],
    ]
)

L = np.linalg.cholesky(corr_seed + 1e-6 * np.eye(len(FACTORS)))
nf = len(FACTORS)

# Build panel: persistent AR(1) factor exposures per asset
panels = []
for d, date in enumerate(dates):
    z = (L @ rng.standard_normal((nf, N_ASSETS))).T  # (N_ASSETS, nf)
    row_data = {"date": date}
    for fi, fname in enumerate(FACTORS):
        xs = z[:, fi]
        xs = (xs - xs.mean()) / (xs.std() + 1e-8)
        row_data[fname] = xs
    # Simulate forward 1m return with small factor premia
    premia = np.array([0.003, -0.002, 0.004, 0.003, 0.003, 0.003, 0.002, -0.001])
    fwd_ret = z @ premia + 0.02 * rng.standard_normal(N_ASSETS)
    row_data["fwd_1m"] = fwd_ret
    row_data["asset"] = assets
    panels.append(pd.DataFrame(row_data))

panel = pd.concat(panels, ignore_index=True)
panel = panel.set_index(["date", "asset"])

print(f"Panel shape: {panel.shape}")
panel.head()

# Cross-sectional mean and std per factor per day, then summarise over time
xs_stats = []
for fname in FACTORS:
    daily = panel[fname].groupby(level="date").agg(["mean", "std", "skew"])
    xs_stats.append(
        {
            "Factor": fname,
            "Mean XS Mean": daily["mean"].mean(),
            "Mean XS Std": daily["std"].mean(),
            "Mean XS Skew": daily["skew"].mean(),
            "Autocorr(1)": panel[fname]
            .groupby(level="asset")
            .apply(lambda s: s.autocorr(1))
            .mean(),
        }
    )

desc = pd.DataFrame(xs_stats).set_index("Factor")
print(desc.to_string())


# Average cross-sectional Spearman correlation between factors
def avg_xs_corr(panel, factors, n_sample=100):
    sample_dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique())[
        [
            int(i)
            for i in np.linspace(
                0, len(panel.index.get_level_values("date").unique()) - 1, n_sample
            )
        ]
    ]
    corr_mats = []
    for d in sample_dates:
        sub = panel.loc[d, factors].dropna()
        if len(sub) > 10:
            corr_mats.append(sub.corr(method="spearman").values)
    return pd.DataFrame(np.mean(corr_mats, axis=0), index=factors, columns=factors)


factor_corr = avg_xs_corr(panel, FACTORS)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

mask = np.triu(np.ones_like(factor_corr, dtype=bool), k=1)
sns.heatmap(
    factor_corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    mask=~mask,
    ax=axes[0],
    linewidths=0.4,
    cbar_kws={"shrink": 0.8},
)
axes[0].set_title(
    "Average Cross-Sectional Factor Correlation (Spearman)",
    fontsize=12,
    fontweight="bold",
)

# VIF (variance inflation factor)
from numpy.linalg import inv

vif_vals = np.diag(inv(factor_corr.values))
pd.Series(vif_vals, index=FACTORS).sort_values(ascending=False).plot.bar(
    ax=axes[1], color=COLORS[: len(FACTORS)], edgecolor="white"
)
axes[1].axhline(5, color="red", linestyle="--", linewidth=1, label="VIF=5 threshold")
axes[1].set_title(
    "Variance Inflation Factor (Multicollinearity Diagnostic)",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_ylabel("VIF")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.show()

from numpy.linalg import lstsq


def fama_macbeth(panel, factors, ret_col="fwd_1m"):
    dates = panel.index.get_level_values("date").unique()
    betas = {f: [] for f in factors}
    t_index = []

    for d in dates:
        sub = panel.loc[d, factors + [ret_col]].dropna()
        if len(sub) < len(factors) + 2:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[factors].values])
        y = sub[ret_col].values
        coef, _, _, _ = lstsq(X, y, rcond=None)
        for i, f in enumerate(factors):
            betas[f].append(coef[i + 1])
        t_index.append(d)

    beta_df = pd.DataFrame(betas, index=pd.DatetimeIndex(t_index))
    return beta_df


fm_betas = fama_macbeth(panel, FACTORS)


# Newey-West t-statistics (lag=4)
def nw_tstat(series, lags=4):
    n = len(series)
    mu = series.mean()
    resid = series - mu
    var = (resid**2).mean()
    for lag in range(1, lags + 1):
        cov = (resid[lag:] * resid[:-lag]).mean()
        var += 2 * (1 - lag / (lags + 1)) * cov
    se = np.sqrt(var / n)
    return mu / se if se > 0 else np.nan


rows = []
for f in FACTORS:
    s = fm_betas[f].dropna()
    rows.append(
        {
            "Factor": f,
            "Mean Premium": s.mean(),
            "Std": s.std(),
            "t-stat (NW)": nw_tstat(s),
            "% Positive": (s > 0).mean(),
        }
    )

fm_tbl = pd.DataFrame(rows).set_index("Factor")
fm_tbl["Significant"] = fm_tbl["t-stat (NW)"].abs() > 1.96

print(fm_tbl.to_string())

fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
axes_flat = axes.ravel()

for ax, (fname, color) in zip(axes_flat, zip(FACTORS, COLORS)):
    cum = (1 + fm_betas[fname]).cumprod()
    ax.plot(cum.index, cum.values, color=color, linewidth=1.5)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.7)
    ax.fill_between(
        cum.index, 1, cum.values, where=cum.values >= 1, alpha=0.15, color=color
    )
    ax.fill_between(
        cum.index, 1, cum.values, where=cum.values < 1, alpha=0.15, color="#ef4444"
    )
    ann_ret = cum.iloc[-1] ** (252 / len(cum)) - 1
    ax.set_title(
        f"{fname}  (Ann. Return: {ann_ret:.1%})", fontsize=10, fontweight="bold"
    )
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.suptitle("Cumulative Factor Returns (Fama-MacBeth)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Stack daily cross-section into (n_obs, n_factors) matrix
all_xs = panel[FACTORS].dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_xs)

pca = PCA(n_components=len(FACTORS))
pca.fit(X_scaled)

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ev_ratio = pca.explained_variance_ratio_
axes[0].bar(
    range(1, len(FACTORS) + 1),
    ev_ratio,
    color=COLORS[: len(FACTORS)],
    edgecolor="white",
)
axes[0].plot(
    range(1, len(FACTORS) + 1),
    np.cumsum(ev_ratio),
    "o-",
    color="black",
    linewidth=2,
    markersize=6,
    label="Cumulative",
)
axes[0].axhline(0.8, color="red", linestyle="--", linewidth=0.8, label="80% threshold")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("PCA Scree Plot", fontsize=12, fontweight="bold")
axes[0].legend()
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Loadings heatmap (first 4 PCs)
loadings = pd.DataFrame(
    pca.components_[:4].T,
    index=FACTORS,
    columns=[f"PC{i+1}" for i in range(4)],
)
sns.heatmap(
    loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, ax=axes[1]
)
axes[1].set_title("PCA Loadings (First 4 PCs)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()

print("Explained variance by first 3 PCs: " f"{sum(ev_ratio[:3]):.1%}")

# Build a simple long-short factor portfolio and attribute its return
# Portfolio: top quintile minus bottom quintile on each factor


def quintile_ls_returns(panel, factor, ret_col="fwd_1m"):
    """Long top quintile, short bottom quintile of factor."""
    results = []
    for d in panel.index.get_level_values("date").unique():
        sub = panel.loc[d, [factor, ret_col]].dropna()
        if len(sub) < 20:
            continue
        q20 = sub[factor].quantile(0.20)
        q80 = sub[factor].quantile(0.80)
        long_ret = sub.loc[sub[factor] >= q80, ret_col].mean()
        short_ret = sub.loc[sub[factor] <= q20, ret_col].mean()
        results.append({"date": d, "ls_return": long_ret - short_ret})
    return pd.DataFrame(results).set_index("date")["ls_return"]


ls_rets = {f: quintile_ls_returns(panel, f) for f in FACTORS}
ls_df = pd.DataFrame(ls_rets).dropna()

# Attribution: contribution to equal-weight factor composite
equal_wt = ls_df.mean(axis=1)
attribution = pd.DataFrame({f: ls_df[f] / len(FACTORS) for f in FACTORS})

# Cumulative contribution chart
cum_attr = attribution.cumsum()
fig, ax = plt.subplots(figsize=(15, 6))
bottom = np.zeros(len(cum_attr))
for fname, color in zip(FACTORS, COLORS):
    ax.fill_between(
        cum_attr.index,
        bottom,
        bottom + cum_attr[fname].values,
        alpha=0.75,
        label=fname,
        color=color,
    )
    bottom = bottom + cum_attr[fname].values

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative Attribution", fontsize=11)
ax.set_title(
    "Cumulative Return Attribution by Factor (Equal-Weight Composite)",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
plt.tight_layout()
plt.show()

# Estimate factor covariance matrix from FM beta time series
# and specific risk from residual variance

factor_cov = fm_betas.cov() * 252  # annualised
factor_vol = np.sqrt(np.diag(factor_cov.values))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    factor_cov,
    annot=True,
    fmt=".5f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.4,
    ax=ax,
    cbar_kws={"label": "Annualised Covariance"},
)
ax.set_title(
    "Annualised Factor Covariance Matrix (Barra-Style Risk Model)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

# Portfolio risk decomposition for equal-weight factor portfolio
w = np.ones(len(FACTORS)) / len(FACTORS)
port_var = w @ factor_cov.values @ w
port_vol = np.sqrt(port_var)
mcr = (factor_cov.values @ w) / port_vol  # marginal contribution to risk
pct_risk = w * mcr / port_vol * 100

risk_tbl = pd.DataFrame(
    {
        "Factor Vol (Ann.)": factor_vol,
        "MCR": mcr,
        "% Risk Contribution": pct_risk,
    },
    index=FACTORS,
)

print(f"Equal-weight composite annualised vol: {port_vol:.2%}")
print(risk_tbl.to_string())


# Summarise all factors in a league table ranked by Sharpe
def factor_sharpe(ret_series):
    r = ret_series.dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan


zoo = []
for f in FACTORS:
    ret = ls_df[f]
    sr = factor_sharpe(ret)
    cum = (1 + ret).prod() - 1
    dd = (((1 + ret).cumprod() / (1 + ret).cumprod().cummax()) - 1).min()
    zoo.append(
        {"Factor": f, "Ann. Sharpe": sr, "Total Return": cum, "Max Drawdown": dd}
    )

zoo_df = (
    pd.DataFrame(zoo).set_index("Factor").sort_values("Ann. Sharpe", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
colors = [COLORS[FACTORS.index(f)] for f in zoo_df.index]
zoo_df["Ann. Sharpe"].plot.bar(ax=ax, color=colors, edgecolor="white")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(0.5, color="green", linestyle=":", linewidth=0.8, label="Sharpe=0.5")
ax.set_title(
    "Factor Zoo - Annualised Sharpe Ratio (Long-Short Quintile Portfolios)",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend()
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

print(zoo_df.to_string())
