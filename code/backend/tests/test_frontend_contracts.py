"""
Frontend/backend contract tests.

These tests pin the exact response field names that the web and mobile
frontends read. They exist because the clients and the API drifted apart in
the past (fields renamed, endpoints moved), and unit tests on either side did
not catch it. If an endpoint's shape changes in a way that would break a
client, one of these tests fails.

Each assertion mirrors a specific field access in the frontend code.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def _assert_fields(obj: dict, fields: Iterable[str], where: str) -> None:
    missing = [f for f in fields if f not in obj]
    assert not missing, f"{where}: missing fields {missing}; got keys {sorted(obj)}"


def _first(items: Any, where: str) -> dict:
    assert isinstance(items, list) and items, f"{where}: expected non-empty list"
    assert isinstance(items[0], dict), f"{where}: expected list of objects"
    return items[0]


# ── Auth ────────────────────────────────────────────────────────────────────
def test_auth_register_and_login_contract():
    """Web AuthContext and mobile authSlice read { token, user }."""
    email = "contract_user@example.com"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "name": "Contract User", "password": "Sup3rSecret!"},
    )
    assert reg.status_code in (200, 201), reg.text
    body = reg.json()
    _assert_fields(body, ["token", "user"], "POST /api/auth/register")

    login = client.post(
        "/api/auth/login", json={"email": email, "password": "Sup3rSecret!"}
    )
    assert login.status_code == 200, login.text
    _assert_fields(login.json(), ["token", "user"], "POST /api/auth/login")


# ── Portfolio ─────────────────────────────────────────────────────────────--
def test_portfolio_summary_contract():
    """Web Dashboard + mobile HomeScreen read totalValue/cash/dailyPnL/totalPnL."""
    r = client.get("/api/v1/portfolio/")
    assert r.status_code == 200, r.text
    _assert_fields(
        r.json(),
        ["totalValue", "cash", "dailyPnL", "totalPnL", "allocation"],
        "GET /api/v1/portfolio/",
    )


def test_positions_contract():
    """Web Portfolio/Dashboard positions table fields."""
    r = client.get("/api/v1/portfolio/positions")
    assert r.status_code == 200, r.text
    pos = _first(r.json(), "GET /api/v1/portfolio/positions")
    _assert_fields(
        pos,
        ["ticker", "quantity", "entryPrice", "currentPrice", "unrealizedPnL", "weight"],
        "position",
    )


def test_holdings_contract():
    """Mobile PortfolioScreen reads symbol/value/weight."""
    r = client.get("/api/v1/portfolio/holdings")
    assert r.status_code == 200, r.text
    holding = _first(r.json(), "GET /api/v1/portfolio/holdings")
    _assert_fields(holding, ["symbol", "value", "weight"], "holding")


def test_performance_contract():
    """Web Dashboard equity curve + sharpe from performance.metrics."""
    r = client.get("/api/v1/portfolio/performance")
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_fields(body, ["equityCurve", "metrics"], "GET /portfolio/performance")
    point = _first(body["equityCurve"], "performance.equityCurve")
    _assert_fields(point, ["timestamp", "value"], "equityCurve point")
    _assert_fields(body["metrics"], ["sharpeRatio"], "performance.metrics")


# ── Strategies ────────────────────────────────────────────────────────────--
def test_strategies_contract():
    r = client.get("/api/v1/strategies/")
    assert r.status_code == 200, r.text
    strat = _first(r.json(), "GET /api/v1/strategies/")
    _assert_fields(strat, ["id", "name", "status", "performance"], "strategy")
    _assert_fields(
        strat["performance"],
        ["sharpeRatio", "maxDrawdown", "profitFactor", "winRate", "totalReturn"],
        "strategy.performance",
    )


def test_strategy_equity_curve_contract():
    """Web Strategies chart reads {day, value, benchmark}."""
    strat = _first(client.get("/api/v1/strategies/").json(), "strategies")
    r = client.get(f"/api/v1/strategies/{strat['id']}/equity-curve")
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_fields(body, ["equityCurve"], "equity-curve")
    _assert_fields(
        _first(body["equityCurve"], "equityCurve"),
        ["day", "value", "benchmark"],
        "equity-curve point",
    )


# ── Risk ─────────────────────────────────────────────────────────────────--
def test_risk_metrics_contract():
    r = client.get("/api/v1/risk/metrics")
    assert r.status_code == 200, r.text
    _assert_fields(
        r.json(),
        [
            "var",
            "cvar",
            "sharpeRatio",
            "sortinoRatio",
            "maxDrawdown",
            "beta",
            "correlation",
            "volatility",
        ],
        "GET /api/v1/risk/metrics",
    )


def test_stress_scenarios_contract():
    r = client.get("/api/v1/risk/stress-scenarios")
    assert r.status_code == 200, r.text
    sc = _first(r.json(), "GET /api/v1/risk/stress-scenarios")
    _assert_fields(
        sc, ["name", "pnl", "duration", "recovery", "portfolioImpact"], "scenario"
    )


def test_risk_radar_contract():
    r = client.get("/api/v1/risk/radar")
    assert r.status_code == 200, r.text
    _assert_fields(
        _first(r.json(), "GET /api/v1/risk/radar"), ["metric", "value"], "radar"
    )


# ── Backtest ─────────────────────────────────────────────────────────────--
def test_backtest_contract():
    """Web Backtest reads flat metric fields off the result."""
    strat = _first(client.get("/api/v1/strategies/").json(), "strategies")
    r = client.post(
        "/api/v1/backtest/",
        json={
            "strategyId": strat["id"],
            "startDate": "2023-01-01",
            "endDate": "2024-01-01",
            "initialCapital": 100000,
        },
    )
    assert r.status_code in (200, 201), r.text
    _assert_fields(
        r.json(),
        [
            "totalReturn",
            "annualisedReturn",
            "sharpeRatio",
            "sortinoRatio",
            "maxDrawdown",
            "winRate",
            "profitFactor",
            "finalCapital",
        ],
        "POST /api/v1/backtest/",
    )


# ── Market data ──────────────────────────────────────────────────────────--
def test_market_quotes_contract():
    """Web/mobile market screens read /quotes (not the bare root)."""
    r = client.get("/api/v1/market-data/quotes")
    assert r.status_code == 200, r.text
    q = _first(r.json(), "GET /api/v1/market-data/quotes")
    _assert_fields(
        q,
        ["ticker", "last", "bid", "ask", "volume", "open", "high", "low", "close"],
        "quote",
    )


def test_market_history_contract():
    r = client.get("/api/v1/market-data/historical/AAPL", params={"days": 30})
    assert r.status_code == 200, r.text
    bar = _first(r.json(), "GET /api/v1/market-data/historical/AAPL")
    _assert_fields(
        bar, ["timestamp", "open", "high", "low", "close", "volume"], "ohlcv"
    )


# ── Trading / orders ─────────────────────────────────────────────────────--
def test_trading_order_lifecycle_contract():
    """Web/mobile Trading reads the Order shape and can cancel pending orders."""
    created = client.post(
        "/api/v1/trading/orders",
        json={"ticker": "AAPL", "side": "BUY", "quantity": 10, "orderType": "MARKET"},
    )
    assert created.status_code in (200, 201), created.text
    order = created.json()
    _assert_fields(
        order,
        ["id", "ticker", "side", "quantity", "orderType", "status"],
        "POST /api/v1/trading/orders",
    )

    listing = client.get("/api/v1/trading/orders")
    assert listing.status_code == 200, listing.text
    assert isinstance(listing.json(), list)

    # MARKET orders fill instantly; LIMIT orders stay pending and are cancellable.
    pending = client.post(
        "/api/v1/trading/orders",
        json={
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 5,
            "orderType": "LIMIT",
            "price": 100.0,
        },
    )
    assert pending.status_code in (200, 201), pending.text
    pending_order = pending.json()
    assert pending_order["status"] == "pending", pending_order

    cancelled = client.delete(f"/api/v1/trading/orders/{pending_order['id']}")
    assert cancelled.status_code in (200, 204), cancelled.text


# ── Research ─────────────────────────────────────────────────────────────--
def test_research_papers_contract():
    r = client.get("/api/research/papers")
    assert r.status_code == 200, r.text
    paper = _first(r.json(), "GET /api/research/papers")
    _assert_fields(
        paper, ["id", "title", "authors", "abstract", "category", "year"], "paper"
    )


# ── Alternative data ─────────────────────────────────────────────────────--
def test_alternative_data_sources_contract():
    r = client.get("/api/v1/alternative-data/sources")
    assert r.status_code == 200, r.text
    src = _first(r.json(), "GET /api/v1/alternative-data/sources")
    _assert_fields(
        src, ["id", "name", "type", "status", "lastUpdate", "dataPoints"], "source"
    )


# ── CORS (cross-origin web client) ───────────────────────────────────────--
def test_cors_allows_web_client_origins():
    """The Expo web build and other local dev servers call the API cross-origin.
    A missing CORS header surfaces in the browser as a 'Network error'."""
    for origin in (
        "http://localhost:8081",
        "http://localhost:19006",
        "http://127.0.0.1:8081",
        "http://localhost:3000",
        "http://192.168.82.76:8081",
        "http://10.0.0.5:8081",
    ):
        r = client.get("/api/v1/portfolio/", headers={"Origin": origin})
        assert r.status_code == 200, r.text
        assert (
            r.headers.get("access-control-allow-origin") == origin
        ), f"missing/incorrect CORS header for origin {origin}"

    preflight = client.options(
        "/api/v1/trading/orders",
        headers={
            "Origin": "http://localhost:8081",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.status_code in (200, 204), preflight.text
    assert (
        preflight.headers.get("access-control-allow-origin") == "http://localhost:8081"
    )
