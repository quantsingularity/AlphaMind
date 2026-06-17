# AlphaMind Documentation

This directory documents AlphaMind as it actually exists in this repository. Where a capability is scaffolding or research-only rather than wired into the live API, the docs say so.

## Start here

- [Installation](INSTALLATION.md) - set up the backend and both frontends.
- [Usage](USAGE.md) - run the stack and use each surface.
- [Configuration](CONFIGURATION.md) - environment variables and settings.

## Reference

- [Architecture](architecture.md) - how the tiers fit together.
- [API Reference](API.md) - every REST endpoint with request and response shapes.
- [CLI / Scripts](CLI.md) - the helper scripts in `scripts/`.
- [Feature Matrix](FEATURE_MATRIX.md) - what is implemented, what is scaffolding, what is planned.
- [Backtest Results](BACKTEST_RESULTS.md) - how backtests are produced and how to read them.
- [Troubleshooting](troubleshooting.md) - common problems and fixes.

## Examples

- [API usage](examples/api_usage.md) - call the API from the shell and Python.
- [DDPG trading](examples/ddpg_trading.md) - use the reinforcement-learning agent and trading environment.
- [Sentiment / alternative data](examples/sentiment_analysis.md) - work with the alternative-data surface.

## Contributing

- [Contributing guide](CONTRIBUTING.md)

## A note on data

By default the backend returns deterministic seeded and synthetic data so the platform runs with no external accounts. Market data uses Yahoo Finance automatically when reachable, with a synthetic fallback. There is no live broker; trading is simulated in-process. Keep this in mind when reading any performance figures in these docs: they illustrate the mechanics, not a real track record.
