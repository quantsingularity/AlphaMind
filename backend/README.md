# AlphaMind Backend

Institutional-Grade Quantitative AI Trading System - Backend API

## Overview

The AlphaMind backend is a production-ready FastAPI-based REST API that provides comprehensive trading, portfolio management, market data, and strategy backtesting capabilities.

## Features

- **FastAPI Framework**: High-performance async API with automatic documentation
- **Trading Operations**: Order management, execution tracking
- **Portfolio Management**: Real-time portfolio tracking and performance metrics
- **Market Data**: Real-time quotes and historical data access
- **Strategy Backtesting**: Comprehensive backtesting engine with performance metrics
- **Authentication**: JWT-based authentication system
- **Data Processing**: Advanced data pipeline with caching and monitoring
- **Risk Management**: Bayesian VaR, stress testing, risk aggregation
- **AI Models**: Reinforcement learning, transformers, generative models

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- PostgreSQL or MySQL (optional, for production)
- Redis (optional, for caching)

### Installation

1. **Clone and navigate to backend directory**:

   ```bash
   cd backend
   ```

2. **Create virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the server**:

   ```bash
   # Using the startup script
   ./start_backend.sh

   # Or directly with uvicorn
   python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Using Setup Script

The included `start_backend.sh` script automates the setup process:

```bash
chmod +x start_backend.sh
./start_backend.sh
```

This script will:

- Create a virtual environment if it doesn't exist
- Install all dependencies
- Load environment variables from `.env`
- Start the FastAPI server with hot reload

## API Documentation

Once the server is running, access the interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Health & Status

- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with service information

### Trading

- `POST /api/v1/trading/orders` - Create new order
- `GET /api/v1/trading/orders` - List all orders
- `GET /api/v1/trading/orders/{order_id}` - Get specific order

### Portfolio

- `GET /api/v1/portfolio/` - Get current portfolio
- `GET /api/v1/portfolio/performance` - Get performance metrics

### Market Data

- `GET /api/v1/market-data/quote/{symbol}` - Get real-time quote
- `GET /api/v1/market-data/historical/{symbol}` - Get historical data

### Strategies

- `GET /api/v1/strategies/` - List all strategies
- `POST /api/v1/strategies/backtest` - Run strategy backtest

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_api.py -v
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

## Project Structure

```
backend/
├── api/                      # FastAPI application
│   ├── main.py              # Main application entry point
│   ├── routers/             # API route handlers
│   ├── schemas/             # Pydantic models
│   ├── services/            # Business logic
│   └── dependencies/        # Dependency injection
├── core/                    # Core utilities
│   ├── config.py           # Configuration management
│   ├── exceptions.py       # Custom exceptions
│   └── logging.py          # Logging configuration
├── data_processing/        # Data processing modules
│   ├── caching.py         # Caching mechanisms
│   ├── pipeline.py        # ETL pipelines
│   ├── monitoring.py      # Performance monitoring
│   ├── parallel.py        # Parallel processing
│   └── streaming.py       # Stream processing
├── ai_models/             # AI/ML models
├── alpha_research/        # Alpha research tools
├── alternative_data/      # Alternative data processing
├── execution_engine/      # Order execution
├── market_data/          # Market data connectors
├── risk_system/          # Risk management
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── .env.example         # Environment template
└── README.md            # This file
```

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for complete list):

- `API_HOST`: API host address (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `SECRET_KEY`: JWT secret key
- `DB_HOST`, `DB_PORT`, `DB_NAME`: Database configuration
- `REDIS_HOST`, `REDIS_PORT`: Redis configuration
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Database Setup

For production use, configure a proper database:

**PostgreSQL**:

```bash
# Update .env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=alphamind
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

**MySQL**:

```bash
# Update .env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=alphamind
DB_USERNAME=your_user
DB_PASSWORD=your_password
```

## Development

### Code Style

The project follows PEP 8 style guidelines. Format code using:

```bash
black .
flake8 .
```

### Adding New Endpoints

1. Create a new router in `api/routers/`
2. Define Pydantic schemas in `api/schemas/`
3. Implement business logic in `api/services/`
4. Register the router in `api/main.py`
5. Add tests in `tests/`

### Example: Adding a New Endpoint

```python
# api/routers/new_feature.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class NewFeatureRequest(BaseModel):
    param1: str
    param2: int

@router.post("/new-feature")
async def new_feature(request: NewFeatureRequest):
    return {"result": "success"}
```

Then register in `api/main.py`:

```python
from api.routers import new_feature
app.include_router(new_feature.router, prefix="/api/v1/new-feature", tags=["new-feature"])
```

## Modules Overview

### Core Modules

- **config.py**: Configuration management with validation
- **exceptions.py**: Custom exception hierarchy
- **logging.py**: Structured logging setup

### Data Processing

- **caching.py**: Multi-level caching with TTL and LRU policies
- **pipeline.py**: ETL pipeline framework
- **monitoring.py**: Performance monitoring and alerting
- **parallel.py**: Parallel and distributed processing
- **streaming.py**: Real-time data stream processing

### AI Models

- **reinforcement_learning.py**: RL-based trading agents
- **transformer_timeseries/**: Transformer models for forecasting
- **generative_finance.py**: Generative models for synthetic data

### Market Data

- **api_connectors/**: Connectors for various data providers
  - Alpha Vantage
  - IEX Cloud
  - Bloomberg
  - FRED
- **live_feed.py**: Real-time market data feed
- **backtesting.py**: Historical backtesting engine

### Risk System

- **bayesian_var.py**: Bayesian Value at Risk
- **stress_testing.py**: Scenario-based stress testing
- **risk_aggregation/**: Portfolio-level risk aggregation

## Performance Optimization

### Caching

The backend implements multi-level caching:

```python
from data_processing.caching import cache_function

@cache_function(ttl_seconds=300)
def expensive_computation(symbol):
    # Your expensive computation
    return result
```

### Async Operations

All API endpoints are async for better performance:

```python
@router.get("/data/{symbol}")
async def get_data(symbol: str):
    # Async database query
    data = await fetch_data_async(symbol)
    return data
```

## Monitoring and Logging

### Logging

Logs are written to both console and file (if configured):

```python
from core.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing order", extra={"order_id": "123"})
```

### Performance Monitoring

```python
from data_processing.monitoring import PerformanceMonitor

monitor = PerformanceMonitor("trading")
monitor.register_metric("order_latency")
monitor.set_threshold("order_latency", "max", 100, AlertLevel.WARNING)
```

## Troubleshooting

### Common Issues

**Import Errors**:

```bash
# Ensure backend is in Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Port Already in Use**:

```bash
# Change port in .env or use different port
uvicorn api.main:app --port 8001
```

**Database Connection Errors**:

```bash
# Verify database is running
# Check credentials in .env
# Test connection manually
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment-Specific Settings

- Set `API_DEBUG=false` in production
- Use proper SECRET_KEY
- Configure database connection pooling
- Enable HTTPS/TLS
- Set appropriate CORS origins

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:

- GitHub Issues: https://github.com/abrar2030/AlphaMind/issues
- Documentation: http://localhost:8000/docs

## Changelog

### Version 1.0.0 (Current)

- ✅ FastAPI-based REST API
- ✅ Trading order management
- ✅ Portfolio tracking
- ✅ Market data endpoints
- ✅ Strategy backtesting
- ✅ JWT authentication
- ✅ Comprehensive test suite
- ✅ Data processing pipelines
- ✅ Performance monitoring
- ✅ API documentation
