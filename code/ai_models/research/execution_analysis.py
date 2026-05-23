"""Execution quality analysis: market impact, spread costs, and implementation shortfall."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]
pd.set_option("display.float_format", "{:.5f}".format)

SEED = 42
rng = np.random.default_rng(SEED)
print("Execution Analysis environment ready.")

# Simulate intraday order book for a single trading day
# 6.5 hours = 390 minutes
N_MIN = 390
timestamps = pd.date_range("2024-01-15 09:30", periods=N_MIN, freq="1min")

ADV = 1_000_000  # average daily volume (shares)
price_0 = 100.0
spread_bps = 5.0  # bid-ask spread in bps

# Simulate intraday volume (U-shaped)
t_norm = np.linspace(0, 1, N_MIN)
vol_profile = 1.5 * (1 - 2 * t_norm + 2 * t_norm**2) + 0.5 * rng.exponential(0.1, N_MIN)
vol_profile /= vol_profile.sum()
intraday_vol = (vol_profile * ADV).astype(int)

# Price process (GBM with microstructure noise)
sigma_intra = 0.20 / np.sqrt(252 * 390)  # per-minute vol
price = np.zeros(N_MIN)
price[0] = price_0
for t in range(1, N_MIN):
    price[t] = price[t - 1] * np.exp(rng.normal(0, sigma_intra))
price += rng.normal(0, price_0 * spread_bps / 20_000, N_MIN)  # microstructure

intraday = pd.DataFrame(
    {
        "price": price,
        "volume": intraday_vol,
        "spread_bps": spread_bps + rng.exponential(0.5, N_MIN),
    },
    index=timestamps,
)

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
axes[0].plot(intraday.index, intraday["price"], color=COLORS[0], linewidth=1.2)
axes[0].set_title("Intraday Price (Simulated)", fontweight="bold")
axes[0].set_ylabel("Price ($)")

axes[1].bar(
    intraday.index, intraday["volume"], color=COLORS[2], alpha=0.7, width=0.0006
)
axes[1].set_title("Intraday Volume Profile (U-shape)", fontweight="bold")
axes[1].set_ylabel("Volume (shares)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.show()

print(
    f"Simulated ADV: {ADV:,} shares | Avg spread: {intraday['spread_bps'].mean():.2f} bps"
)

# Almgren-Chriss linear permanent + temporary impact model
# and empirical square-root model


def impact_linear(q, adv, sigma, eta=0.1, gamma=0.1):
    """Linear Almgren-Chriss impact (bps)."""
    participation = q / adv
    perm_impact = gamma * sigma * participation * 10_000
    temp_impact = eta * sigma * participation * 10_000
    return perm_impact, temp_impact


def impact_sqrt(q, adv, sigma, alpha=0.5, beta=0.6):
    """Empirical square-root market impact model (bps)."""
    participation = q / adv
    return alpha * sigma * np.sqrt(participation) * 10_000 * beta


# Sweep over order sizes
order_fracs = np.linspace(0.001, 0.25, 100)  # 0.1% to 25% of ADV
sigma_daily = 0.20  # annual vol -> daily

perm_impacts = []
temp_impacts = []
sqrt_impacts = []

for f in order_fracs:
    q = f * ADV
    pi, ti = impact_linear(q, ADV, sigma_daily / np.sqrt(252))
    si = impact_sqrt(q, ADV, sigma_daily / np.sqrt(252))
    perm_impacts.append(pi)
    temp_impacts.append(ti)
    sqrt_impacts.append(si)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    order_fracs * 100,
    perm_impacts,
    color=COLORS[0],
    linewidth=2,
    label="Linear Permanent",
)
ax.plot(
    order_fracs * 100,
    temp_impacts,
    color=COLORS[1],
    linewidth=2,
    label="Linear Temporary",
)
ax.plot(
    order_fracs * 100,
    sqrt_impacts,
    color=COLORS[2],
    linewidth=2,
    linestyle="--",
    label="Square-Root Model",
)
ax.set_xlabel("Order Size (% of ADV)")
ax.set_ylabel("Market Impact (bps)")
ax.set_title(
    "Market Impact vs Order Size - Linear vs Square-Root Models",
    fontsize=12,
    fontweight="bold",
)
ax.legend()
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.show()

# Execute a 10% ADV order using TWAP, VWAP, and IS (optimal)
order_qty = int(0.10 * ADV)

# TWAP: equal slices every minute
twap_slice = order_qty / N_MIN
twap_executed = np.full(N_MIN, twap_slice)

# VWAP: proportional to volume profile
vwap_executed = vol_profile * order_qty

# IS (aggressive front-loading - simple proxy)
decay_factor = np.exp(-np.linspace(0, 5, N_MIN))
is_executed = decay_factor / decay_factor.sum() * order_qty


def execution_price(executed_qty, price_series):
    """Volume-weighted average execution price."""
    total = executed_qty.sum()
    return (executed_qty * price_series).sum() / total if total > 0 else np.nan


arrival_price = price[0]
twap_px = execution_price(twap_executed, price)
vwap_px = execution_price(vwap_executed, price)
is_px = execution_price(is_executed, price)
mkt_px = execution_price(vol_profile * order_qty, price)  # market VWAP

results = {
    "TWAP": {
        "Exec Price": twap_px,
        "vs Arrival (bps)": (twap_px / arrival_price - 1) * 10_000,
    },
    "VWAP": {
        "Exec Price": vwap_px,
        "vs Arrival (bps)": (vwap_px / arrival_price - 1) * 10_000,
    },
    "IS (Agressive)": {
        "Exec Price": is_px,
        "vs Arrival (bps)": (is_px / arrival_price - 1) * 10_000,
    },
    "Market VWAP": {
        "Exec Price": mkt_px,
        "vs Arrival (bps)": (mkt_px / arrival_price - 1) * 10_000,
    },
}

exec_df = pd.DataFrame(results).T
print(f"Arrival price: ${arrival_price:.4f}")
print(exec_df.to_string())

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(timestamps, price, color="gray", linewidth=1.0, label="Market Price", alpha=0.7)
ax.axhline(
    arrival_price,
    color="black",
    linewidth=1.2,
    linestyle="-",
    label=f"Arrival={arrival_price:.2f}",
)
ax.axhline(
    twap_px, color=COLORS[0], linewidth=1.5, linestyle="--", label=f"TWAP={twap_px:.4f}"
)
ax.axhline(
    vwap_px, color=COLORS[2], linewidth=1.5, linestyle="--", label=f"VWAP={vwap_px:.4f}"
)
ax.axhline(
    is_px, color=COLORS[1], linewidth=1.5, linestyle=":", label=f"IS={is_px:.4f}"
)
ax.set_title("TWAP vs VWAP vs IS Execution Benchmarks", fontsize=12, fontweight="bold")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.show()

# IS = (Execution Price - Decision Price) / Decision Price (bps)
# Decompose into: delay cost + market impact + spread cost + timing luck

decision_price = arrival_price
end_price = price[-1]


def implementation_shortfall(exec_price, decision_price, spread_bps):
    total_is = (exec_price / decision_price - 1) * 10_000
    spread_cost = spread_bps / 2  # half-spread per trade
    market_impact = total_is - spread_cost
    return {
        "Total IS (bps)": total_is,
        "Spread Cost (bps)": spread_cost,
        "Market Impact (bps)": market_impact,
    }


is_components = pd.DataFrame(
    {
        algo: implementation_shortfall(
            px, decision_price, intraday["spread_bps"].mean()
        )
        for algo, px in [("TWAP", twap_px), ("VWAP", vwap_px), ("IS Algo", is_px)]
    }
).T

fig, ax = plt.subplots(figsize=(10, 5))
is_components.plot.bar(ax=ax, color=COLORS[:3], edgecolor="white", stacked=False)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title(
    "Implementation Shortfall Decomposition by Algorithm",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylabel("Cost (bps)")
ax.tick_params(axis="x", rotation=15)
ax.legend()
plt.tight_layout()
plt.show()
print(is_components.to_string())

N_TRADES = 500
trade_sizes_bps = rng.uniform(0.5, 20, N_TRADES)  # order as % of ADV
spreads_bps = rng.exponential(5, N_TRADES) + 2
slippage_bps = (
    0.3 * trade_sizes_bps + 0.6 * spreads_bps / 2 + rng.normal(0, 1, N_TRADES)
)

trades_df = pd.DataFrame(
    {
        "size_pct_adv": trade_sizes_bps,
        "spread_bps": spreads_bps,
        "slippage_bps": slippage_bps,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    trades_df["size_pct_adv"],
    trades_df["slippage_bps"],
    alpha=0.4,
    s=20,
    color=COLORS[0],
)
m, b = np.polyfit(trades_df["size_pct_adv"], trades_df["slippage_bps"], 1)
x_line = np.linspace(0, 20, 100)
axes[0].plot(
    x_line,
    m * x_line + b,
    color=COLORS[1],
    linewidth=2,
    label=f"Linear fit: {m:.2f}x + {b:.2f}",
)
axes[0].set_xlabel("Order Size (% ADV)")
axes[0].set_ylabel("Slippage (bps)")
axes[0].set_title("Slippage vs Order Size", fontweight="bold")
axes[0].legend()

axes[1].scatter(
    trades_df["spread_bps"], trades_df["slippage_bps"], alpha=0.4, s=20, color=COLORS[2]
)
axes[1].set_xlabel("Bid-Ask Spread (bps)")
axes[1].set_ylabel("Slippage (bps)")
axes[1].set_title("Slippage vs Bid-Ask Spread", fontweight="bold")

plt.suptitle("Slippage Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("Slippage statistics:")
print(trades_df["slippage_bps"].describe().to_string())

# Participation rate: fraction of market volume our order represents
# Higher participation -> more market impact but faster execution

participation_rates = np.linspace(0.01, 0.30, 50)
time_to_complete = 1 / participation_rates  # normalised
impact_at_rate = np.array(
    [impact_sqrt(p * ADV, ADV, sigma_daily / np.sqrt(252)) for p in participation_rates]
)

# Liquidity score per asset (simulated)
liq_scores = rng.beta(2, 3, 100) * 100  # 0-100 score

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax2 = ax.twinx()
ax.plot(
    participation_rates * 100,
    impact_at_rate,
    color=COLORS[0],
    linewidth=2,
    label="Impact (bps)",
)
ax2.plot(
    participation_rates * 100,
    time_to_complete,
    color=COLORS[2],
    linewidth=2,
    linestyle="--",
    label="Completion Time (norm.)",
)
ax.set_xlabel("Participation Rate (%)")
ax.set_ylabel("Market Impact (bps)", color=COLORS[0])
ax2.set_ylabel("Completion Time (norm.)", color=COLORS[2])
ax.set_title("Impact-Time Trade-off vs Participation Rate", fontweight="bold")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

axes[1].hist(liq_scores, bins=30, color=COLORS[3], alpha=0.7, edgecolor="white")
axes[1].axvline(
    liq_scores.mean(),
    color=COLORS[1],
    linewidth=2,
    linestyle="--",
    label=f"Mean = {liq_scores.mean():.1f}",
)
axes[1].set_title("Asset Liquidity Score Distribution", fontweight="bold")
axes[1].set_xlabel("Liquidity Score (0=Illiquid, 100=Liquid)")
axes[1].legend()
plt.tight_layout()
plt.show()

# Simulate 252 daily strategy trades and compute net-of-cost P&L
N_SIM = 252
gross_alpha_bps = rng.normal(4.0, 8.0, N_SIM)  # daily gross alpha in bps

# Transaction costs depend on strategy characteristics
turnover_pct_adv = rng.uniform(1, 15, N_SIM)  # daily turnover as % ADV
spread_cost_bps = 2.5  # half-spread
impact_cost_bps = np.array(
    [
        impact_sqrt(t / 100 * ADV, ADV, sigma_daily / np.sqrt(252))
        for t in turnover_pct_adv
    ]
)
total_cost_bps = spread_cost_bps + impact_cost_bps
net_alpha_bps = gross_alpha_bps - total_cost_bps

attribution = pd.DataFrame(
    {
        "Gross Alpha (bps)": gross_alpha_bps,
        "Spread Cost (bps)": -spread_cost_bps,
        "Market Impact (bps)": -impact_cost_bps,
        "Net Alpha (bps)": net_alpha_bps,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cum_attr = attribution[
    ["Gross Alpha (bps)", "Spread Cost (bps)", "Market Impact (bps)"]
].cumsum()
cum_attr.plot(ax=axes[0], color=COLORS[:3], linewidth=1.5)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.7)
axes[0].set_title("Cumulative Return Attribution (bps)", fontweight="bold")
axes[0].set_ylabel("Cumulative bps")

(attribution["Net Alpha (bps)"] / 100).cumsum().plot(
    ax=axes[1], color=COLORS[0], linewidth=2
)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.7)
axes[1].set_title("Net-of-Cost Cumulative Alpha (%)", fontweight="bold")
axes[1].set_ylabel("Cumulative Return")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.tight_layout()
plt.show()

print("Annual cost summary:")
print(f"  Gross alpha:   {gross_alpha_bps.mean()*252:.0f} bps/yr")
print(f"  Spread costs:  {spread_cost_bps*252:.0f} bps/yr")
print(f"  Impact costs:  {impact_cost_bps.mean()*252:.0f} bps/yr")
print(f"  Net alpha:     {net_alpha_bps.mean()*252:.0f} bps/yr")

# Almgren-Chriss optimal execution: minimise E[cost] + lambda * Var[cost]
# Closed-form solution for linear impact


def ac_optimal_schedule(
    total_qty, T, adv, sigma, eta=0.1, gamma=0.05, risk_aversion=1e-5
):
    """
    Almgren-Chriss optimal liquidation trajectory.
    Returns array of shares traded in each period.
    """
    sigma_per_period = sigma * total_qty / adv  # scale sigma to position units
    kappa = np.sqrt(risk_aversion * sigma_per_period**2 / eta)
    t = np.arange(T + 1) / T
    inv_trajectory = total_qty * (np.sinh(kappa * (1 - t)) / np.sinh(kappa))
    trades = -np.diff(inv_trajectory)
    return trades


T = 78  # 2-hour execution window (2-minute intervals = 78 * 1.5min)
order_qty_ac = int(0.05 * ADV)  # 5% ADV order

schedules = {}
for risk_av in [1e-6, 1e-5, 1e-4]:
    sched = ac_optimal_schedule(
        order_qty_ac, T, ADV, sigma_daily, risk_aversion=risk_av
    )
    schedules[f"lambda={risk_av:.0e}"] = sched / order_qty_ac

schedules["TWAP"] = np.ones(T) / T
schedules["Front-Load"] = (
    np.exp(-np.linspace(0, 4, T)) / np.exp(-np.linspace(0, 4, T)).sum()
)

fig, ax = plt.subplots(figsize=(14, 5))
for (name, sched), color in zip(schedules.items(), COLORS):
    ax.plot(np.cumsum(sched) * 100, linewidth=2, label=name, color=color)
ax.set_xlabel("Period")
ax.set_ylabel("Cumulative % Executed")
ax.set_title(
    "Almgren-Chriss Optimal Execution Schedules (varying risk aversion)",
    fontsize=12,
    fontweight="bold",
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print("=" * 65)
print("  EXECUTION ANALYSIS SUMMARY")
print("=" * 65)
print()
print("Market Impact Model (Square-Root, 10% ADV order):")
imp_10pct = impact_sqrt(0.10 * ADV, ADV, sigma_daily / np.sqrt(252))
print(f"  10% ADV order -> {imp_10pct:.2f} bps market impact")
print()
print("Benchmark Execution Quality:")
for algo, px in [("TWAP", twap_px), ("VWAP", vwap_px), ("IS Algo", is_px)]:
    cost = (px / arrival_price - 1) * 10_000
    print(f"  {algo:12s}  Exec={px:.4f}  vs Arrival={cost:.2f} bps")
print()
print("Annual Cost Attribution:")
print(f"  Gross alpha:   {gross_alpha_bps.mean()*252:6.0f} bps/yr")
print(f"  Spread costs:  {spread_cost_bps*252:6.0f} bps/yr")
print(f"  Impact costs:  {impact_cost_bps.mean()*252:6.0f} bps/yr")
print(f"  Net alpha:     {net_alpha_bps.mean()*252:6.0f} bps/yr")
print(
    f"  Cost ratio:    {total_cost_bps.mean()/gross_alpha_bps.mean():.1%} of gross alpha"
)
print("=" * 65)
