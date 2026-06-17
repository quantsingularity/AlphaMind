# Installation

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer (npm included)
- Git
- Optional: Docker and Docker Compose

## 1. Clone

```bash
git clone https://github.com/quantsingularity/AlphaMind.git
cd AlphaMind
```

## 2. Backend

```bash
cd code/backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dependency set is large because it includes the research stack (PyTorch, TensorFlow, PyMC, QuantLib). If you only want to run the API and its tests, you can install the lighter subset:

```bash
pip install fastapi "uvicorn[standard]" "pydantic>=2.4" pydantic-settings \
  bcrypt PyJWT "python-jose[cryptography]" "passlib[bcrypt]" httpx \
  "SQLAlchemy>=2.0" "numpy<2.1" pandas greenlet aiosqlite yfinance \
  pytest pytest-asyncio pytest-cov
```

Run it:

```bash
python -m main
# Uvicorn running on http://0.0.0.0:8000 ; docs at http://localhost:8000/docs
```

On first run the SQLite database (`alphamind.db`) is created and seeded.

## 3. Web frontend

```bash
cd ../../web-frontend
npm install
npm run dev          # http://localhost:3000
```

The dev server proxies `/api` and `/health` to `http://localhost:8000`.

## 4. Mobile frontend

```bash
cd ../mobile-frontend
npm install
npm start            # then press w (web), a (Android), or i (iOS)
```

Expo prints a LAN URL such as `http://192.168.x.x:8081`. Opening the web build at that LAN address makes the app call the backend at the same host on port 8000, which avoids localhost-forwarding issues on WSL and VPNs.

## 5. Optional: Docker Compose

A Compose file in `code/backend` brings up the API together with PostgreSQL, Redis, and Kafka/Zookeeper:

```bash
cd code/backend
docker compose up --build
```

The API runs without these services too; Compose is for a fuller local environment.

## Database migrations

Alembic is configured in `code/backend`:

```bash
cd code/backend
alembic upgrade head
```

## Verifying the install

```bash
# Backend
cd code/backend && pytest

# Web
cd ../../web-frontend && npm test

# Mobile
cd ../mobile-frontend && npm test
```

If TensorFlow is not installed, the model-specific backend tests are skipped; the API, contract, security, and market-data tests still run.
