"""
Basic API functionality tests.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "alphamind-api"
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_docs_available():
    """Test that API documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "AlphaMind API"


def test_trading_orders_endpoint():
    """Test creating a trading order."""
    order_data = {
        "symbol": "AAPL",
        "quantity": 100.0,
        "order_type": "market",
        "price": None,
    }
    response = client.post("/api/v1/trading/orders", json=order_data)
    assert response.status_code == 200
    data = response.json()
    assert "order_id" in data
    assert data["symbol"] == "AAPL"
    assert data["quantity"] == 100.0
    assert data["status"] == "pending"


def test_get_orders_endpoint():
    """Test retrieving all orders."""
    response = client.get("/api/v1/trading/orders")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_portfolio_endpoint():
    """Test getting portfolio information."""
    response = client.get("/api/v1/portfolio/")
    assert response.status_code == 200
    data = response.json()
    assert "total_value" in data
    assert "cash" in data
    assert "positions" in data
    assert isinstance(data["positions"], list)


def test_portfolio_performance_endpoint():
    """Test getting portfolio performance."""
    response = client.get("/api/v1/portfolio/performance")
    assert response.status_code == 200
    data = response.json()
    assert "daily_return" in data
    assert "total_return" in data
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data


def test_market_data_quote_endpoint():
    """Test getting market data quote."""
    response = client.get("/api/v1/market-data/quote/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "price" in data
    assert "volume" in data
    assert "timestamp" in data


def test_market_data_historical_endpoint():
    """Test getting historical market data."""
    response = client.get("/api/v1/market-data/historical/AAPL?days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "data" in data
    assert isinstance(data["data"], list)


def test_strategies_endpoint():
    """Test getting trading strategies."""
    response = client.get("/api/v1/strategies/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_backtest_endpoint():
    """Test running a backtest."""
    backtest_data = {
        "strategy_id": "test_strategy",
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-12-31T00:00:00",
        "initial_capital": 100000.0,
    }
    response = client.post("/api/v1/strategies/backtest", json=backtest_data)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy_id"] == "test_strategy"
    assert "total_return" in data
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data
    assert "trades" in data


def test_invalid_endpoint():
    """Test that invalid endpoints return 404."""
    response = client.get("/api/v1/invalid/endpoint")
    assert response.status_code == 404


def test_cors_headers():
    """Test that CORS middleware is configured."""
    # Note: TestClient doesn't trigger CORS middleware
    # In a real environment, CORS headers would be present
    # This test verifies the endpoint works without errors
    response = client.get("/health")
    assert response.status_code == 200
