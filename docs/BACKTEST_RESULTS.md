# Backtests

This document explains how backtests work in AlphaMind and how to read the output. It deliberately does not publish a performance track record, because the default data is seeded and synthetic; any numbers you see are illustrative of the mechanics, not evidence of real-world returns.

## Running a backtest

Via the API:

```bash
curl -s -X POST http://localhost:8000/api/v1/backtest/ \
  -H 'Content-Type: application/json' \
  -d '{
    "strategyId": "<id from /api/v1/strategies/>",
    "startDate": "2023-01-01",
    "endDate": "2024-01-01",
    "initialCapital": 100000
  }'
```

Via the UI: open Backtest, pick a strategy, set the date range and initial capital, and run.

## Reading the result

The response is a flat object:

| Field              | Meaning                                        |
| :----------------- | :--------------------------------------------- |
| `totalReturn`      | Total return over the period.                  |
| `annualisedReturn` | Return scaled to a yearly rate.                |
| `sharpeRatio`      | Excess return per unit of total volatility.    |
| `sortinoRatio`     | Excess return per unit of downside volatility. |
| `maxDrawdown`      | Largest peak-to-trough decline (fraction).     |
| `winRate`          | Share of winning periods (fraction).           |
| `profitFactor`     | Gross profit divided by gross loss.            |
| `finalCapital`     | Ending capital from `initialCapital`.          |

## Interpreting metrics responsibly

- A backtest is a hypothesis about the past, not a promise about the future.
- The default series are synthetic and deterministic, so results are reproducible but not market-realistic.
- To run against real history, supply real price data through the market-data connectors and extend the strategy and backtest services accordingly.

## Where the logic lives

The backtesting entry point is the backtest router and its service; additional research-grade backtesting utilities live in `code/backend/market_data/backtesting.py`. The reinforcement-learning environments in `code/ai_models/environments` provide an alternative, simulation-based way to evaluate agents.
