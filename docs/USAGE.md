# AlphaMind Usage Guide

This guide covers typical usage patterns for AlphaMind, from basic operations to advanced workflows.

## Table of Contents

- [Starting AlphaMind](#starting-alphamind)
- [Library Usage](#library-usage)
- [API Usage](#api-usage)
- [Common Workflows](#common-workflows)
- [Advanced Usage](#advanced-usage)

## Starting AlphaMind

### Start All Services

The quickest way to start AlphaMind with all components:

```bash
# Start backend and frontend
./scripts/run_alphamind.sh

# Access points:
# - Web UI: http://localhost:3000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Redoc: http://localhost:8000/redoc
```

### Start Individual Components

#### Backend API Server

```bash
cd backend
source ../venv/bin/activate

# Development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Web Frontend

```bash
cd web-frontend

# Development mode
npm run dev

# Production build
npm run build
npm run preview
```

#### Mobile Frontend

```bash
cd mobile-frontend

# iOS (macOS only)
npm run ios

# Android
npm run android

# Expo (cross-platform)
npm start
```

## Library Usage

### Import AlphaMind Modules

AlphaMind can be used as a Python library for programmatic trading and analysis.

#### Example 1: Basic Portfolio Management

```python
from backend.risk_system.risk_aggregation.portfolio_risk import PortfolioRiskManager
import numpy as np

# Initialize portfolio risk manager
risk_manager = PortfolioRiskManager()

# Define portfolio positions
positions = {
    'AAPL': {'quantity': 100, 'price': 150.0},
    'GOOGL': {'quantity': 50, 'price': 2800.0},
    'MSFT': {'quantity': 75, 'price': 300.0}
}

# Calculate portfolio metrics
portfolio_value = sum(pos['quantity'] * pos['price'] for pos in positions.values())
print(f"Total Portfolio Value: ${portfolio_value:,.2f}")

# Expected output:
# Total Portfolio Value: $177,500.00
```

#### Example 2: Order Management

```python
from backend.execution_engine.order_management.order_manager import (
    OrderManager, Order, OrderType, OrderSide, OrderTimeInForce
)
import datetime

# Initialize order manager
order_manager = OrderManager()

# Create a market order
order = Order(
    order_id="ORD-001",
    instrument_id="AAPL",
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.MARKET,
    time_in_force=OrderTimeInForce.DAY
)

# Submit order
order_manager.submit_order(order)
print(f"Order {order.order_id} submitted: {order.side.value} {order.quantity} {order.instrument_id}")

# Expected output:
# Order ORD-001 submitted: buy 100 AAPL
```

#### Example 3: AI Model - DDPG Trading

```python
from backend.ai_models.ddpg_trading import DDPGTrader
import numpy as np

# Initialize DDPG trader
trader = DDPGTrader(
    state_dim=10,    # Number of features
    action_dim=1,    # Trading signal
    lr_actor=0.001,
    lr_critic=0.002
)

# Simulate market state (price, volume, indicators, etc.)
market_state = np.random.randn(10)

# Get trading action
action = trader.get_action(market_state, add_noise=True)
print(f"Trading Signal: {action[0]:.4f}")
print(f"Interpretation: {'BUY' if action[0] > 0 else 'SELL'} with strength {abs(action[0]):.2%}")

# Expected output:
# Trading Signal: 0.3421
# Interpretation: BUY with strength 34.21%
```

#### Example 4: Risk Management - Bayesian VaR

```python
from backend.risk_system.bayesian_var import BayesianVaR
import numpy as np

# Generate sample returns (simulated)
returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year

# Initialize Bayesian VaR calculator
var_calculator = BayesianVaR(returns=returns)

# Build probabilistic model (this may take a few minutes)
# var_calculator.build_model()

# Calculate Value at Risk at 95% confidence
# var_95 = var_calculator.calculate_var(alpha=0.05)
# print(f"95% VaR: {var_95:.4f}")

# Note: Actual execution requires PyMC3 and significant computation time
print("Bayesian VaR calculation requires full dependencies and ~5-10 minutes of computation")

# Expected output (when fully configured):
# 95% VaR: -0.0423  (meaning 5% chance of losing more than 4.23%)
```

#### Example 5: Alternative Data - Sentiment Analysis

```python
from backend.alternative_data.sentiment_analysis import SentimentAnalyzer

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

# Analyze news sentiment
news_text = "Apple Inc. reported record quarterly earnings, beating analyst expectations significantly."

# Get sentiment score
# sentiment = analyzer.analyze(news_text)
# print(f"Sentiment Score: {sentiment['score']:.2f}")
# print(f"Classification: {sentiment['label']}")

# Note: Actual execution may require API keys and NLP models
print("Sentiment analysis module - requires NLP model configuration")

# Expected output (when fully configured):
# Sentiment Score: 0.87
# Classification: Positive
```

## API Usage

### REST API Examples

AlphaMind provides a comprehensive REST API for programmatic access.

#### Health Check

```bash
curl http://localhost:8000/health

# Response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "timestamp": "2025-01-15T10:30:00Z"
# }
```

#### Get Portfolio

```bash
curl -X GET http://localhost:8000/api/v1/portfolio \
  -H "Content-Type: application/json"

# Response:
# {
#   "total_value": 100000.0,
#   "cash": 50000.0,
#   "positions": [],
#   "updated_at": "2025-01-15T10:30:00Z"
# }
```

#### Create Order

```bash
curl -X POST http://localhost:8000/api/v1/trading/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 100,
    "order_type": "market",
    "price": null
  }'

# Response:
# {
#   "order_id": "ORD-1642243800.0",
#   "symbol": "AAPL",
#   "quantity": 100,
#   "status": "pending",
#   "timestamp": "2025-01-15T10:30:00Z"
# }
```

#### Get Market Data

```bash
curl -X GET "http://localhost:8000/api/v1/market-data?symbol=AAPL&interval=1d&limit=30" \
  -H "Content-Type: application/json"

# Response:
# {
#   "symbol": "AAPL",
#   "data": [
#     {
#       "timestamp": "2025-01-15T09:30:00Z",
#       "open": 150.0,
#       "high": 152.5,
#       "low": 149.5,
#       "close": 151.0,
#       "volume": 1000000
#     }
#   ]
# }
```

### Python API Client Example

```python
import httpx
import json

# API configuration
API_BASE = "http://localhost:8000"

# Create HTTP client
client = httpx.Client(base_url=API_BASE)

# Health check
response = client.get("/health")
print(f"API Status: {response.json()['status']}")

# Get portfolio
response = client.get("/api/v1/portfolio")
portfolio = response.json()
print(f"Portfolio Value: ${portfolio['total_value']:,.2f}")

# Create order
order_data = {
    "symbol": "AAPL",
    "quantity": 100,
    "order_type": "market"
}
response = client.post("/api/v1/trading/orders", json=order_data)
order = response.json()
print(f"Order Created: {order['order_id']}")

# Close client
client.close()

# Expected output:
# API Status: healthy
# Portfolio Value: $100,000.00
# Order Created: ORD-1642243800.0
```

## Common Workflows

### Workflow 1: Backtest a Trading Strategy

```bash
# 1. Configure backtest parameters in backend/.env
# BACKTEST_START_DATE=2020-01-01
# BACKTEST_END_DATE=2023-12-31

# 2. Run backtest script (example - actual implementation may vary)
cd backend
python -m backend.alpha_research.portfolio_optimization

# 3. View results in generated reports
ls -lh backtest_results/
```

### Workflow 2: Train an AI Model

```python
from backend.ai_models.ddpg_trading import DDPGTrader
from backend.market_data.api_connectors import MarketDataConnector
import numpy as np

# 1. Load market data
data_connector = MarketDataConnector()
# market_data = data_connector.fetch_data('AAPL', '2020-01-01', '2023-12-31')

# 2. Prepare features
# features = prepare_features(market_data)

# 3. Initialize trader
trader = DDPGTrader(state_dim=10, action_dim=1)

# 4. Train model
print("Training DDPG model... (this may take hours)")
# for episode in range(1000):
#     state = reset_environment()
#     for step in range(252):  # Trading days
#         action = trader.get_action(state)
#         next_state, reward = environment_step(action)
#         trader.update(state, action, reward, next_state)
#         state = next_state

# 5. Save model
# trader.save_model('models/ddpg_aapl_v1.pkl')

print("Note: Full training requires market data and significant compute time")
```

### Workflow 3: Real-time Monitoring

```python
import asyncio
from backend.risk_system.risk_aggregation.real_time_monitoring import RealTimeMonitor

async def monitor_portfolio():
    # Initialize monitor
    monitor = RealTimeMonitor()

    # Start monitoring
    print("Starting real-time portfolio monitoring...")

    # Monitor for 60 seconds (example)
    # await monitor.start()
    # await asyncio.sleep(60)
    # await monitor.stop()

    print("Real-time monitoring requires configured data feeds")

# Run monitor
# asyncio.run(monitor_portfolio())
```

### Workflow 4: Generate Trading Signals

```python
from backend.alpha_research.factor_models.machine_learning_factors import MLFactorModel
import pandas as pd

# Initialize ML factor model
model = MLFactorModel()

# Load market data
# data = pd.read_csv('market_data.csv')

# Generate signals
# signals = model.generate_signals(data)

# Filter strong signals
# strong_signals = signals[abs(signals['score']) > 0.7]

# print(f"Generated {len(strong_signals)} strong trading signals")
# print(strong_signals.head())

print("Signal generation requires trained models and market data")
```

## Advanced Usage

### Custom Strategy Development

Create a custom trading strategy:

```python
from typing import Dict, Any
import pandas as pd

class CustomStrategy:
    """Example custom trading strategy."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.positions = {}

    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market data."""
        signals = pd.DataFrame(index=market_data.index)

        # Example: Simple moving average crossover
        short_ma = market_data['close'].rolling(window=20).mean()
        long_ma = market_data['close'].rolling(window=50).mean()

        # Buy signal when short MA crosses above long MA
        signals['signal'] = 0
        signals.loc[short_ma > long_ma, 'signal'] = 1
        signals.loc[short_ma < long_ma, 'signal'] = -1

        return signals

    def execute(self, signals: pd.DataFrame):
        """Execute trades based on signals."""
        for timestamp, row in signals.iterrows():
            if row['signal'] == 1:
                print(f"{timestamp}: BUY signal")
            elif row['signal'] == -1:
                print(f"{timestamp}: SELL signal")

# Usage
# strategy = CustomStrategy(params={'threshold': 0.02})
# signals = strategy.generate_signals(market_data)
# strategy.execute(signals)
```

### WebSocket Real-time Data

```python
import asyncio
import websockets
import json

async def stream_market_data():
    """Stream real-time market data via WebSocket."""
    uri = "ws://localhost:8000/ws/market-data"

    async with websockets.connect(uri) as websocket:
        # Subscribe to symbols
        await websocket.send(json.dumps({
            "action": "subscribe",
            "symbols": ["AAPL", "GOOGL", "MSFT"]
        }))

        # Receive data
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data['symbol']} @ ${data['price']}")

# Run WebSocket client
# asyncio.run(stream_market_data())
print("WebSocket streaming requires running AlphaMind API server")
```

### Parallel Backtesting

```bash
# Run multiple backtests in parallel
./scripts/run_tests.sh --component alpha_research --parallel

# Or use Python multiprocessing
python -m backend.research.parallel_backtest \
  --strategies momentum,mean_reversion,ml_signals \
  --workers 4 \
  --start-date 2020-01-01 \
  --end-date 2023-12-31
```

## Environment Variables for Usage

Key environment variables that affect runtime behavior:

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_DEBUG=false  # Set to false in production

# Trading Configuration
export INITIAL_CAPITAL=100000.0
export MAX_POSITION_SIZE=0.1  # Max 10% per position
export RISK_FREE_RATE=0.02

# Model Configuration
export MODEL_TRAINING_ENABLED=true
export MODEL_CACHE_DIR=/tmp/alphamind/models

# Performance
export MAX_WORKERS=4
export CACHE_TTL_SECONDS=3600
```

## Best Practices

1. **Always use virtual environments** to isolate dependencies
2. **Configure API keys** before running strategies that require external data
3. **Monitor resource usage** when running ML models or backtests
4. **Use test mode** before deploying to production
5. **Review logs** in `logs/` directory for debugging
6. **Backup data** regularly, especially trained models
7. **Use version control** for custom strategies and configurations

## Next Steps

- **Configuration**: See [CONFIGURATION.md](CONFIGURATION.md) for detailed settings
- **API Reference**: See [API.md](API.md) for complete API documentation
- **Examples**: Explore [EXAMPLES/](EXAMPLES/) for more code samples
- **Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand system design
- **Troubleshooting**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

## Getting Help

- **Documentation**: Browse [docs/README.md](README.md)
- **Issues**: Report problems at [GitHub Issues](https://github.com/quantsingularity/AlphaMind/issues)
- **Examples**: Study working examples in [EXAMPLES/](EXAMPLES/)
