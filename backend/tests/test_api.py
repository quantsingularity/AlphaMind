"""
Tests for AlphaMind API endpoints.
"""

import os
import sys

from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "alphamind-api"
        assert "version" in data

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestTradingEndpoints:
    """Test trading endpoints."""

    def test_create_order(self):
        """Test order creation."""
        order_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "order_type": "market",
            "price": None,
        }
        response = client.post("/api/v1/trading/orders", json=order_data)
        assert response.status_code == 200
        data = response.json()
        assert "order_id" in data
        assert data["symbol"] == "AAPL"
        assert data["quantity"] == 100
        assert data["status"] == "pending"

    def test_get_orders(self):
        """Test getting all orders."""
        response = client.get("/api/v1/trading/orders")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_order_not_found(self):
        """Test getting non-existent order."""
        response = client.get("/api/v1/trading/orders/invalid-id")
        assert response.status_code == 404


class TestPortfolioEndpoints:
    """Test portfolio endpoints."""

    def test_get_portfolio(self):
        """Test getting portfolio."""
        response = client.get("/api/v1/portfolio/")
        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "cash" in data
        assert "positions" in data
        assert isinstance(data["positions"], list)

    def test_get_performance(self):
        """Test getting portfolio performance."""
        response = client.get("/api/v1/portfolio/performance")
        assert response.status_code == 200
        data = response.json()
        assert "daily_return" in data
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data


class TestMarketDataEndpoints:
    """Test market data endpoints."""

    def test_get_quote(self):
        """Test getting quote for symbol."""
        response = client.get("/api/v1/market-data/quote/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "price" in data
        assert "volume" in data
        assert "timestamp" in data

    def test_get_historical_data(self):
        """Test getting historical data."""
        response = client.get("/api/v1/market-data/historical/AAPL?days=30")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "data" in data
        assert isinstance(data["data"], list)


class TestStrategyEndpoints:
    """Test strategy endpoints."""

    def test_get_strategies(self):
        """Test getting all strategies."""
        response = client.get("/api/v1/strategies/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_run_backtest(self):
        """Test running backtest."""
        backtest_data = {
            "strategy_id": "momentum-v1",
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T00:00:00",
            "initial_capital": 100000.0,
        }
        response = client.post("/api/v1/strategies/backtest", json=backtest_data)
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_id"] == "momentum-v1"
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "trades" in data
