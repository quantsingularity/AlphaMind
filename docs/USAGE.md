# Usage

This guide covers running the platform and what each surface does. For setup, see [INSTALLATION.md](INSTALLATION.md).

## Running everything

Open three terminals.

```bash
# 1) Backend
cd code/backend && source .venv/bin/activate && python -m main

# 2) Web dashboard
cd web-frontend && npm run dev          # http://localhost:3000

# 3) Mobile app
cd mobile-frontend && npm start         # press w / a / i
```

## First run

1. Open the web dashboard or the mobile app.
2. Create an account (Sign Up) or sign in. With the backend running, this hits `/api/auth`. If the backend is unreachable, the clients create a local demo session so you can still explore.
3. You land on the dashboard. Navigate between Strategies, Portfolio, Backtest, Risk, Market Data, Trading, Research, and Alternative Data.

## What each screen does

- Dashboard / Home: portfolio value, daily and total P&L, and a performance summary.
- Strategies: the strategy list with performance metrics; each strategy has an equity curve.
- Portfolio: total value, cash, positions, and holdings with weights.
- Backtest: pick a strategy, set a date range and initial capital, run it, and read the metrics (return, Sharpe, Sortino, drawdown, win rate, profit factor, final capital).
- Risk: value at risk, beta, volatility, max drawdown, and stress scenarios.
- Market Data: live quotes (Yahoo Finance when reachable, synthetic otherwise) and a price-history chart.
- Trading: an order ticket (market, limit, stop) and an order blotter. Market orders fill immediately; limit and stop orders stay pending and can be cancelled. All simulated in-process.
- Research: a list of research papers.
- Alternative Data: configured data sources with status, volume, and latency.
- Settings (mobile): theme, notifications, display currency, reset, and sign out.

## Live versus synthetic data

Each market-data quote carries a `source` field of `live` or `synthetic`. If you have network access, Yahoo Finance is used automatically. Set `POLYGON_API_KEY` to enable the Polygon connector. With no connectivity, a deterministic synthetic feed keeps every screen populated.

## Mobile on WSL or behind a VPN

If you run the backend in WSL and the browser on Windows, open the Expo web build at the LAN address Expo prints (for example `http://192.168.x.x:8081`) rather than `localhost:8081`. The app will then call the backend at the same host on port 8000, which avoids localhost-forwarding and VPN interception. Alternatively set `EXPO_PUBLIC_API_BASE_URL`.

## Using the research library

The modules in `code/ai_models` can be used directly. See [examples/ddpg_trading.md](examples/ddpg_trading.md) for the reinforcement-learning agent and trading environment, and [examples/api_usage.md](examples/api_usage.md) for calling the REST API.
