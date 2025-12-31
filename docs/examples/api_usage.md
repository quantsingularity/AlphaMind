# Example: API Usage

Complete examples of using the AlphaMind REST API for common tasks.

## Prerequisites

```bash
pip install httpx  # or requests
```

## Authentication

First, obtain an access token:

```python
import httpx

API_BASE = "http://localhost:8000"

# Login to get token
def login(username, password):
    client = httpx.Client(base_url=API_BASE)
    response = client.post("/api/v1/auth/login", json={
        "username": username,
        "password": password
    })
    return response.json()["access_token"]

# Get token (example)
# token = login("demo_user", "demo_pass")
token = "your_jwt_token_here"

# Create authenticated client
client = httpx.Client(
    base_url=API_BASE,
    headers={"Authorization": f"Bearer {token}"}
)
```

## Example 1: Get Portfolio

```python
# Fetch current portfolio
response = client.get("/api/v1/portfolio")
portfolio = response.json()

print("=== Portfolio Summary ===")
print(f"Total Value: ${portfolio['total_value']:,.2f}")
print(f"Cash: ${portfolio['cash']:,.2f}")
print(f"Positions: {len(portfolio['positions'])}")
print("\nPositions:")
for pos in portfolio['positions']:
    print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f} "
          f"(P&L: ${pos['pnl']:,.2f})")
```

**Output:**

```
=== Portfolio Summary ===
Total Value: $177,500.00
Cash: $27,500.00
Positions: 3

Positions:
  AAPL: 100 shares @ $151.00 (P&L: $100.00)
  GOOGL: 50 shares @ $2,850.00 (P&L: $2,500.00)
  MSFT: 75 shares @ $305.00 (P&L: $375.00)
```

## Example 2: Place Orders

```python
# Create a market order
def create_market_order(symbol, quantity):
    response = client.post("/api/v1/trading/orders", json={
        "symbol": symbol,
        "quantity": quantity,
        "order_type": "market"
    })
    return response.json()

# Create a limit order
def create_limit_order(symbol, quantity, price):
    response = client.post("/api/v1/trading/orders", json={
        "symbol": symbol,
        "quantity": quantity,
        "order_type": "limit",
        "price": price,
        "time_in_force": "gtc"
    })
    return response.json()

# Place orders
market_order = create_market_order("AAPL", 10)
print(f"Market Order: {market_order['order_id']} - {market_order['status']}")

limit_order = create_limit_order("GOOGL", 5, 2800.00)
print(f"Limit Order: {limit_order['order_id']} - {limit_order['status']}")
```

**Output:**

```
Market Order: ORD-1642243800.123 - pending
Limit Order: ORD-1642243801.456 - pending
```

## Example 3: Get Market Data

```python
# Fetch historical market data
def get_market_data(symbol, interval="1d", limit=30):
    response = client.get("/api/v1/market-data", params={
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    })
    return response.json()

# Get AAPL daily data
data = get_market_data("AAPL", interval="1d", limit=5)

print(f"=== {data['symbol']} Market Data ===")
for bar in data['data'][:3]:
    print(f"{bar['timestamp']}: "
          f"O: ${bar['open']:.2f} "
          f"H: ${bar['high']:.2f} "
          f"L: ${bar['low']:.2f} "
          f"C: ${bar['close']:.2f} "
          f"V: {bar['volume']:,}")
```

**Output:**

```
=== AAPL Market Data ===
2025-01-15T09:30:00Z: O: $150.00 H: $152.50 L: $149.50 C: $151.00 V: 52,418,900
2025-01-14T09:30:00Z: O: $148.50 H: $151.00 L: $148.00 C: $150.50 V: 48,325,100
2025-01-13T09:30:00Z: O: $147.00 H: $149.00 L: $146.50 C: $148.25 V: 51,234,800
```

## Example 4: Run Backtest

```python
import time

# Start a backtest
def start_backtest(strategy_id, start_date, end_date, symbols, initial_capital=100000):
    response = client.post(f"/api/v1/strategies/{strategy_id}/backtest", json={
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "symbols": symbols
    })
    return response.json()

# Check backtest status
def get_backtest_status(backtest_id):
    response = client.get(f"/api/v1/backtests/{backtest_id}")
    return response.json()

# Start backtest
backtest = start_backtest(
    strategy_id="momentum_v1",
    start_date="2020-01-01",
    end_date="2023-12-31",
    symbols=["AAPL", "GOOGL", "MSFT"]
)

print(f"Backtest started: {backtest['backtest_id']}")
print(f"Status: {backtest['status']}")
print(f"Estimated completion: {backtest['estimated_completion']}")

# Poll for completion (in practice, use webhooks)
# while True:
#     status = get_backtest_status(backtest['backtest_id'])
#     if status['status'] == 'completed':
#         print("Backtest complete!")
#         print(f"Results: {status['results']}")
#         break
#     time.sleep(30)
```

**Output:**

```
Backtest started: BT-20250115-001
Status: running
Estimated completion: 2025-01-15T10:35:00Z
```

## Example 5: Portfolio Performance Metrics

```python
# Get performance metrics
def get_performance(period="1m"):
    response = client.get("/api/v1/portfolio/performance", params={
        "period": period
    })
    return response.json()

# Fetch performance
perf = get_performance(period="1m")

print("=== Performance Metrics (1 Month) ===")
print(f"Total Return: {perf['total_return']:.2%}")
print(f"Daily Return: {perf['daily_return']:.2%}")
print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {perf['sortino_ratio']:.2f}")
print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
print(f"Volatility: {perf['volatility']:.2%}")
print(f"Beta: {perf['beta']:.2f}")
print(f"Alpha: {perf['alpha']:.4f}")
```

**Output:**

```
=== Performance Metrics (1 Month) ===
Total Return: 8.42%
Daily Return: 0.32%
Sharpe Ratio: 1.45
Sortino Ratio: 2.12
Max Drawdown: -5.23%
Volatility: 1.87%
Beta: 1.03
Alpha: 0.0021
```

## Example 6: Batch Operations

```python
# Submit multiple orders
def batch_create_orders(orders):
    results = []
    for order in orders:
        response = client.post("/api/v1/trading/orders", json=order)
        results.append(response.json())
    return results

# Create multiple orders
orders_to_create = [
    {"symbol": "AAPL", "quantity": 10, "order_type": "market"},
    {"symbol": "GOOGL", "quantity": 5, "order_type": "limit", "price": 2800.0},
    {"symbol": "MSFT", "quantity": 15, "order_type": "market"}
]

results = batch_create_orders(orders_to_create)

print("=== Batch Order Results ===")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['symbol']}: {result['order_id']} - {result['status']}")
```

**Output:**

```
=== Batch Order Results ===
1. AAPL: ORD-1642243900.123 - pending
2. GOOGL: ORD-1642243900.456 - pending
3. MSFT: ORD-1642243900.789 - pending
```

## Example 7: Error Handling

```python
from httpx import HTTPStatusError

def safe_api_call(func, *args, **kwargs):
    """Wrapper for safe API calls with error handling."""
    try:
        return func(*args, **kwargs)
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Authentication failed - token expired or invalid")
        elif e.response.status_code == 429:
            print("Rate limit exceeded - please wait")
        elif e.response.status_code == 500:
            print("Server error - please try again later")
        else:
            print(f"API error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Use safe wrapper
result = safe_api_call(get_market_data, "INVALID_SYMBOL")
if result is None:
    print("Failed to fetch data")
```

## Complete Trading Bot Example

```python
import time
from datetime import datetime

class SimpleTradingBot:
    """Simple automated trading bot using AlphaMind API."""

    def __init__(self, api_base, token):
        self.client = httpx.Client(
            base_url=api_base,
            headers={"Authorization": f"Bearer {token}"}
        )
        self.positions = {}

    def get_signal(self, symbol):
        """Generate trading signal based on price action."""
        # Fetch recent price data
        data = self.client.get("/api/v1/market-data", params={
            "symbol": symbol,
            "interval": "5m",
            "limit": 20
        }).json()

        if not data['data']:
            return 0

        # Simple momentum strategy
        prices = [bar['close'] for bar in data['data']]
        if len(prices) < 10:
            return 0

        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-10:]) / 10

        if short_ma > long_ma * 1.01:
            return 1  # Buy signal
        elif short_ma < long_ma * 0.99:
            return -1  # Sell signal
        return 0  # No signal

    def execute_trade(self, symbol, signal):
        """Execute trade based on signal."""
        if signal == 1 and symbol not in self.positions:
            # Buy
            order = self.client.post("/api/v1/trading/orders", json={
                "symbol": symbol,
                "quantity": 10,
                "order_type": "market"
            }).json()
            self.positions[symbol] = order['order_id']
            print(f"[{datetime.now()}] BUY {symbol}: {order['order_id']}")

        elif signal == -1 and symbol in self.positions:
            # Sell
            order = self.client.post("/api/v1/trading/orders", json={
                "symbol": symbol,
                "quantity": 10,
                "order_type": "market"
            }).json()
            del self.positions[symbol]
            print(f"[{datetime.now()}] SELL {symbol}: {order['order_id']}")

    def run(self, symbols, interval=300, duration=3600):
        """Run trading bot."""
        print(f"Starting trading bot for {symbols}")
        print(f"Checking every {interval}s for {duration}s\n")

        start_time = time.time()
        iteration = 0

        while time.time() - start_time < duration:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            for symbol in symbols:
                signal = self.get_signal(symbol)
                if signal != 0:
                    self.execute_trade(symbol, signal)

            time.sleep(interval)

        print("\nBot stopped")

# Initialize and run bot (example - commented out)
# bot = SimpleTradingBot(API_BASE, token)
# bot.run(symbols=["AAPL", "GOOGL"], interval=300, duration=3600)
print("Trading bot example (run() commented out for safety)")
```

## Cleanup

```python
# Close client when done
client.close()
```

## Next Steps

- Explore [API Reference](../API.md) for all endpoints
- Check [Configuration](../CONFIGURATION.md) for API settings
- Review [WebSocket API](../API.md#websocket-api) for real-time data
- Study [Error Handling](../API.md#error-handling) for robust integration
