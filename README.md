# AlphaMind

![CI/CD Status](https://img.shields.io/github/actions/workflow/status/quantsingularity/AlphaMind/cicd.yml?branch=main&label=CI%2FCD&logo=github)
[![License](https://github.com/quantsingularity/AlphaMind/blob/main/LICENSE)](https://github.com/quantsingularity/AlphaMind/blob/main/LICENSE)

## Quantitative AI Trading Platform

AlphaMind is a full-stack quantitative trading platform: a FastAPI backend that serves portfolio, strategy, risk, backtest, market-data, trading, research, and alternative-data APIs, a React web dashboard, and a React Native (Expo) mobile app, all sharing one "Quant Terminal" design language. Alongside the application is a research codebase of machine-learning and quantitative modules (forecasting transformers, reinforcement-learning agents, generative models, Bayesian risk, and execution analytics).

<div align="center">
  <img src="docs/images/alphamind_dashboard.bmp" alt="AlphaMind Dashboard" width="80%">
</div>

> Status and scope. AlphaMind is a portfolio and research-grade system, not a connected production trading desk. By default the backend serves deterministic seeded and synthetic data so the whole stack runs end to end with no external accounts. A real market-data path is wired through Yahoo Finance (and an optional Polygon connector) and is used automatically when reachable. There is no live broker integration; order placement is simulated in-process.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [What Is Actually Implemented](#what-is-actually-implemented)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation and Setup](#installation-and-setup)
- [Running the Stack](#running-the-stack)
- [API Surface](#api-surface)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

AlphaMind demonstrates an institutional-style quant workflow across a real, runnable codebase. The application tier (backend plus two clients) is fully wired and covered by tests. The research tier is a library of ML and quant modules that the application can draw on and that can be used independently for experimentation.

## Project Structure

```
AlphaMind/
├── code/
│   ├── backend/            # FastAPI service: API, auth, services, DB, infra
│   │   ├── app/            # FastAPI app, routers (v1 API), services
│   │   ├── analytics/      # Alpha research, A/B testing, model validation, viz
│   │   ├── core/           # Config and exceptions
│   │   ├── data_processing/# Caching, monitoring, parallel, pipeline, streaming
│   │   ├── db/             # SQLAlchemy models, repositories, session
│   │   ├── execution/      # Liquidity forecasting, market impact, routing, OMS
│   │   ├── infrastructure/ # Auth system (bcrypt + JWT)
│   │   ├── market_data/    # Connectors, live feed, backtesting, exchange API
│   │   ├── risk/           # Bayesian VaR, stress testing, controls, counterparty
│   │   ├── tests/          # Backend test suite
│   │   ├── alembic/        # Database migrations
│   │   ├── docker-compose.yml
│   │   └── requirements.txt
│   └── ai_models/          # Research ML library
│       ├── agents/         # DDPG, PPO, replay buffer
│       ├── environments/   # Trading and portfolio Gym environments
│       ├── forecasting/    # Transformer, attention, multi-horizon models
│       └── generative/     # GAN, generator, discriminator, regime models
├── web-frontend/           # React + TypeScript + Vite dashboard
├── mobile-frontend/        # React Native + Expo app
├── infrastructure/         # Docker, Kubernetes, Terraform, Ansible
├── scripts/                # Build, test, deploy, and dev helper scripts
├── docs/                   # Documentation (this directory)
└── README.md
```

## What Is Actually Implemented

### Application tier (wired and tested)

- FastAPI backend exposing versioned endpoints under `/api/v1` (with non-versioned `/api/*` aliases) for portfolio, strategies, risk, backtest, market data, trading, research, and alternative data, plus authentication under `/api/auth` and health checks.
- Authentication using bcrypt password hashing and HS256 JSON Web Tokens with expiry. The signing key is read from `SECRET_KEY` and the app refuses to start in production or staging if it is unset, the placeholder, or shorter than 32 characters.
- Market-data service with a live connector waterfall (Yahoo Finance, then an optional Polygon connector) and a deterministic synthetic fallback so quotes and history are always available.
- SQLAlchemy data layer. Development uses SQLite (`alphamind.db`); MySQL and PostgreSQL async drivers are included for other environments. Alembic manages migrations.
- React web dashboard: Home, Dashboard, Strategies, Portfolio, Backtest, Risk, Market Data, Trading, Research, Alternative Data, Documentation, About, Settings, and authentication screens.
- React Native (Expo) app with the same feature set across bottom-tab and stacked navigation, Redux Toolkit state, and a light/dark theme.
- Graceful degradation in both clients: if the backend is unreachable, account creation and sign-in fall back to a local demo session and data screens render empty or placeholder states instead of failing hard.

### Research tier (library modules)

- Forecasting: transformer, attention, and multi-horizon model implementations (PyTorch / TensorFlow).
- Reinforcement learning: DDPG and PPO agents with replay buffers and custom trading and portfolio Gym environments.
- Generative models: GAN components for synthetic series and a regime model.
- Risk: Bayesian Value at Risk (PyMC), stress testing, counterparty and aggregation modules.
- Execution: liquidity forecasting, market-impact, order management, and routing modules.

These modules are part of the codebase and can be imported and run; they are not all connected to the live API responses, which by default are seeded.

## Technology Stack

| Area            | Technology                                                                                           |
| :-------------- | :--------------------------------------------------------------------------------------------------- |
| Backend API     | Python 3.10+, FastAPI, Uvicorn, Pydantic v2                                                          |
| Auth            | bcrypt, PyJWT / python-jose, passlib                                                                 |
| Data layer      | SQLAlchemy 2, Alembic, SQLite (dev), MySQL / PostgreSQL (async drivers), Redis                       |
| ML / Quant      | PyTorch, TensorFlow, scikit-learn, Gymnasium, Stable-Baselines3, PyMC, QuantLib, NumPy, Pandas, Dask |
| Market data     | yfinance (Yahoo Finance), optional Polygon connector                                                 |
| Web frontend    | React 19, TypeScript, Vite 7, Tailwind CSS, React Router, TanStack Query, axios, Recharts, D3        |
| Mobile frontend | React Native, Expo, React Navigation, Redux Toolkit, React Native Paper, axios                       |
| Infrastructure  | Docker, Docker Compose, Kubernetes, Terraform, Ansible                                               |
| CI/CD           | GitHub Actions                                                                                       |
| Testing         | pytest (backend), Vitest (web), Jest (mobile)                                                        |

Not part of this project, despite being common in this space: C++ components, Apache Kafka/Spark as runtime dependencies of the app (the Compose file provisions Kafka and Redis for a fuller local stack, but the API runs without them), GraphQL, InfluxDB, and live broker connectivity.

## Architecture

AlphaMind is organized in tiers rather than a sprawl of microservices:

```
Clients
  ├── web-frontend (React/TS)            ── HTTP/JSON ──┐
  └── mobile-frontend (React Native)     ── HTTP/JSON ──┤
                                                        ▼
Backend (FastAPI)
  ├── Routers (/api/v1/*)  auth, portfolio, strategies, risk,
  │                        backtest, market-data, trading,
  │                        research, alternative-data, health
  ├── Services            portfolio, strategy, risk, market-data, trading
  ├── Auth                bcrypt + JWT (HS256), fail-closed secret
  └── Data layer          SQLAlchemy + Alembic (SQLite / MySQL / PostgreSQL)
                                                        ▼
Research library (code/ai_models, code/backend/{analytics,risk,execution})
  forecasting · reinforcement learning · generative · Bayesian risk · execution
```

The frontends are the source of presentation; the backend is the source of truth for data contracts. A dedicated contract test suite pins every response field the clients read so the tiers cannot drift apart silently.

See [docs/architecture.md](docs/architecture.md) for detail.

## Installation and Setup

Prerequisites: Python 3.10+ and Node.js 18+. Docker is optional.

```bash
git clone https://github.com/quantsingularity/AlphaMind.git
cd AlphaMind

# Backend
cd code/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web frontend
cd ../../web-frontend
npm install

# Mobile frontend
cd ../mobile-frontend
npm install
```

Full, environment-specific instructions are in [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Running the Stack

```bash
# 1) Backend (from code/backend, venv active)
python -m main                # serves http://0.0.0.0:8000, docs at /docs

# 2) Web dashboard (from web-frontend)
npm run dev                   # http://localhost:3000, proxies /api to :8000

# 3) Mobile app (from mobile-frontend)
npm start                     # press w for web, a for Android, i for iOS
```

The web dashboard proxies API calls to port 8000. The mobile app resolves the API base URL automatically: on web it uses the host the app was opened from (so opening it via a LAN IP reaches a backend on the same machine), on the Android emulator it uses `10.0.2.2:8000`, and otherwise `localhost:8000`. Override with `EXPO_PUBLIC_API_BASE_URL`.

See [docs/USAGE.md](docs/USAGE.md) and [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## API Surface

Base URL `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and `/redoc`.

| Group            | Prefix                     | Highlights                                        |
| :--------------- | :------------------------- | :------------------------------------------------ |
| Health           | `/health`, `/`             | Liveness check                                    |
| Auth             | `/api/auth`                | `register`, `login`, `refresh`, `profile`         |
| Portfolio        | `/api/v1/portfolio`        | summary, `positions`, `holdings`, `performance`   |
| Strategies       | `/api/v1/strategies`       | list, detail, `{id}/equity-curve`                 |
| Risk             | `/api/v1/risk`             | `metrics`, `stress-scenarios`, `radar`            |
| Backtest         | `/api/v1/backtest`         | run a backtest (POST)                             |
| Market data      | `/api/v1/market-data`      | `quotes`, `quote/{ticker}`, `historical/{ticker}` |
| Trading          | `/api/v1/trading`          | `orders` (GET/POST/DELETE)                        |
| Research         | `/api/v1/research`         | `papers`                                          |
| Alternative data | `/api/v1/alternative-data` | `sources`                                         |

Non-versioned `/api/*` aliases exist for backward compatibility. Full request and response shapes are in [docs/API.md](docs/API.md).

## Testing

```bash
# Backend (from code/backend)
pytest

# Web (from web-frontend)
npm test

# Mobile (from mobile-frontend)
npm test
```

The backend suite includes contract tests (`tests/test_frontend_contracts.py`) that pin the API shapes the clients consume, security tests (`tests/test_security.py`) covering password hashing and secret hardening, and market-data tests (`tests/test_market_data_service.py`) covering the live and synthetic paths. Some model tests require TensorFlow and are skipped if it is not installed.

## CI/CD Pipeline

GitHub Actions (`.github/workflows/cicd.yml`) runs five jobs on push, pull request, and manual dispatch:

| Job                          | Depends on          | What it does                                                                                          |
| :--------------------------- | :------------------ | :---------------------------------------------------------------------------------------------------- |
| Code Quality Checks          | -                   | Python formatter checks and a repository-wide Prettier check                                          |
| Backend Tests                | Code Quality Checks | Runs the pytest suite, a dedicated frontend/backend contract-test step, and uploads a coverage report |
| Backend Build                | Backend Tests       | Builds the backend Docker image with Buildx and uploads it as an artifact                             |
| Web-Frontend Test & Build    | Code Quality Checks | Runs the Vitest suite and produces the production web build                                           |
| Mobile-Frontend Test & Build | Code Quality Checks | Runs the Jest suite and produces the Expo web export                                                  |

## Documentation

| Document                                             | Contents                                |
| :--------------------------------------------------- | :-------------------------------------- |
| [docs/README.md](docs/README.md)                     | Documentation index                     |
| [docs/architecture.md](docs/architecture.md)         | System architecture                     |
| [docs/API.md](docs/API.md)                           | REST API reference                      |
| [docs/INSTALLATION.md](docs/INSTALLATION.md)         | Setup for all components                |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md)       | Environment variables and config        |
| [docs/USAGE.md](docs/USAGE.md)                       | Running and using the platform          |
| [docs/CLI.md](docs/CLI.md)                           | Helper scripts reference                |
| [docs/FEATURE_MATRIX.md](docs/FEATURE_MATRIX.md)     | Feature status, implemented vs planned  |
| [docs/BACKTEST_RESULTS.md](docs/BACKTEST_RESULTS.md) | How backtests work and how to read them |
| [docs/troubleshooting.md](docs/troubleshooting.md)   | Common issues and fixes                 |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)         | Contribution guide                      |
| [docs/examples/](docs/examples/)                     | Worked examples                         |

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
