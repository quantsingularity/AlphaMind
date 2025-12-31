# AlphaMind API Reference

Complete API reference for AlphaMind's REST API, WebSocket API, and GraphQL interface.

## Table of Contents

- [API Overview](#api-overview)
- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket API](#websocket-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## API Overview

**Base URL**: `http://localhost:8000` (development) or `https://api.alphamind.ai` (production)

**API Version**: v1

**Supported Formats**: JSON

**Interactive Documentation**:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

AlphaMind API uses JWT (JSON Web Token) for authentication.

### Obtain Access Token

| Parameter          | Description               |
| ------------------ | ------------------------- |
| **Endpoint**       | `POST /api/v1/auth/login` |
| **Content-Type**   | `application/json`        |
| **Authentication** | None (public endpoint)    |

**Request Body**:

```json
{
  "username": "user@example.com",
  "password": "your_password"
}
```

**Response**:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Using Authentication

Include the token in the `Authorization` header:

```bash
curl -X GET http://localhost:8000/api/v1/portfolio \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## REST API Endpoints

### Health Check

#### GET /health

Check API health status.

| Parameter | Type | Required? | Default | Description   | Example |
| --------- | ---- | --------: | ------: | ------------- | ------- |
| None      | -    |         - |       - | No parameters | -       |

**Example Request**:

```bash
curl http://localhost:8000/health
```

**Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "database": "connected",
  "cache": "active"
}
```

---

### Trading Endpoints

#### POST /api/v1/trading/orders

Create a new trading order.

| Parameter       | Type   | Required? | Default | Description                                         | Example    |
| --------------- | ------ | --------: | ------: | --------------------------------------------------- | ---------- |
| `symbol`        | string |       Yes |       - | Trading symbol                                      | `"AAPL"`   |
| `quantity`      | number |       Yes |       - | Number of shares                                    | `100`      |
| `order_type`    | string |       Yes |       - | Order type: `market`, `limit`, `stop`, `stop_limit` | `"market"` |
| `price`         | number |        No |  `null` | Limit price (required for `limit` orders)           | `150.50`   |
| `stop_price`    | number |        No |  `null` | Stop price (required for `stop` orders)             | `145.00`   |
| `time_in_force` | string |        No | `"day"` | Time in force: `day`, `gtc`, `ioc`, `fok`           | `"gtc"`    |

**Example Request**:

```bash
curl -X POST http://localhost:8000/api/v1/trading/orders \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol": "AAPL",
    "quantity": 100,
    "order_type": "limit",
    "price": 150.50,
    "time_in_force": "gtc"
  }'
```

**Response**:

```json
{
  "order_id": "ORD-1642243800.123",
  "symbol": "AAPL",
  "quantity": 100,
  "order_type": "limit",
  "price": 150.5,
  "status": "pending",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### GET /api/v1/trading/orders

List all orders.

| Parameter | Type    | Required? | Default | Description                                        | Example     |
| --------- | ------- | --------: | ------: | -------------------------------------------------- | ----------- |
| `status`  | string  |        No |  `null` | Filter by status: `pending`, `filled`, `cancelled` | `"pending"` |
| `symbol`  | string  |        No |  `null` | Filter by symbol                                   | `"AAPL"`    |
| `limit`   | integer |        No |   `100` | Maximum number of orders to return                 | `50`        |
| `offset`  | integer |        No |     `0` | Pagination offset                                  | `0`         |

**Example Request**:

```bash
curl -X GET "http://localhost:8000/api/v1/trading/orders?status=pending&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:

```json
{
  "orders": [
    {
      "order_id": "ORD-001",
      "symbol": "AAPL",
      "quantity": 100,
      "status": "pending",
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

#### GET /api/v1/trading/orders/{order_id}

Get specific order details.

| Parameter  | Type   | Required? | Default | Description               | Example     |
| ---------- | ------ | --------: | ------: | ------------------------- | ----------- |
| `order_id` | string |       Yes |       - | Order ID (path parameter) | `"ORD-001"` |

**Example Request**:

```bash
curl -X GET http://localhost:8000/api/v1/trading/orders/ORD-001 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:

```json
{
  "order_id": "ORD-001",
  "symbol": "AAPL",
  "quantity": 100,
  "order_type": "market",
  "status": "filled",
  "filled_quantity": 100,
  "avg_fill_price": 151.25,
  "timestamp": "2025-01-15T10:30:00Z",
  "filled_at": "2025-01-15T10:30:15Z"
}
```

---

### Portfolio Endpoints

#### GET /api/v1/portfolio

Get current portfolio.

| Parameter | Type | Required? | Default | Description   | Example |
| --------- | ---- | --------: | ------: | ------------- | ------- |
| None      | -    |         - |       - | No parameters | -       |

**Example Request**:

```bash
curl -X GET http://localhost:8000/api/v1/portfolio \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:

```json
{
  "total_value": 177500.0,
  "cash": 27500.0,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "avg_price": 150.0,
      "current_price": 151.0,
      "pnl": 100.0
    },
    {
      "symbol": "GOOGL",
      "quantity": 50,
      "avg_price": 2800.0,
      "current_price": 2850.0,
      "pnl": 2500.0
    }
  ],
  "updated_at": "2025-01-15T10:30:00Z"
}
```

#### GET /api/v1/portfolio/performance

Get portfolio performance metrics.

| Parameter | Type   | Required? | Default | Description                                             | Example |
| --------- | ------ | --------: | ------: | ------------------------------------------------------- | ------- |
| `period`  | string |        No |  `"1d"` | Performance period: `1d`, `1w`, `1m`, `3m`, `1y`, `all` | `"1m"`  |

**Example Request**:

```bash
curl -X GET "http://localhost:8000/api/v1/portfolio/performance?period=1m" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:

```json
{
  "period": "1m",
  "total_return": 0.0842,
  "daily_return": 0.0032,
  "sharpe_ratio": 1.45,
  "sortino_ratio": 2.12,
  "max_drawdown": -0.0523,
  "volatility": 0.0187,
  "beta": 1.03,
  "alpha": 0.0021
}
```

---

### Market Data Endpoints

#### GET /api/v1/market-data

Get market data for a symbol.

| Parameter    | Type    | Required? | Default | Description                                  | Example        |
| ------------ | ------- | --------: | ------: | -------------------------------------------- | -------------- |
| `symbol`     | string  |       Yes |       - | Trading symbol                               | `"AAPL"`       |
| `interval`   | string  |        No |  `"1d"` | Data interval: `1m`, `5m`, `15m`, `1h`, `1d` | `"1d"`         |
| `start_date` | string  |        No |  `-30d` | Start date (ISO 8601)                        | `"2025-01-01"` |
| `end_date`   | string  |        No |   `now` | End date (ISO 8601)                          | `"2025-01-15"` |
| `limit`      | integer |        No |   `100` | Maximum number of data points                | `30`           |

**Example Request**:

```bash
curl -X GET "http://localhost:8000/api/v1/market-data?symbol=AAPL&interval=1d&limit=5" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "data": [
    {
      "timestamp": "2025-01-15T09:30:00Z",
      "open": 150.0,
      "high": 152.5,
      "low": 149.5,
      "close": 151.0,
      "volume": 52418900,
      "vwap": 150.85
    },
    {
      "timestamp": "2025-01-14T09:30:00Z",
      "open": 148.5,
      "high": 151.0,
      "low": 148.0,
      "close": 150.5,
      "volume": 48325100,
      "vwap": 149.75
    }
  ]
}
```

---

### Strategy Endpoints

#### GET /api/v1/strategies

List available trading strategies.

**Example Request**:

```bash
curl -X GET http://localhost:8000/api/v1/strategies \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:

```json
{
  "strategies": [
    {
      "id": "momentum_v1",
      "name": "Momentum Strategy",
      "description": "Trend-following momentum strategy",
      "status": "active",
      "performance": {
        "sharpe_ratio": 1.85,
        "total_return": 0.234
      }
    },
    {
      "id": "mean_reversion_v1",
      "name": "Mean Reversion Strategy",
      "description": "Statistical arbitrage mean reversion",
      "status": "active",
      "performance": {
        "sharpe_ratio": 1.52,
        "total_return": 0.187
      }
    }
  ]
}
```

#### POST /api/v1/strategies/{strategy_id}/backtest

Run a backtest for a strategy.

| Parameter         | Type   | Required? |  Default | Description                    | Example             |
| ----------------- | ------ | --------: | -------: | ------------------------------ | ------------------- |
| `strategy_id`     | string |       Yes |        - | Strategy ID (path parameter)   | `"momentum_v1"`     |
| `start_date`      | string |       Yes |        - | Backtest start date            | `"2020-01-01"`      |
| `end_date`        | string |       Yes |        - | Backtest end date              | `"2023-12-31"`      |
| `initial_capital` | number |        No | `100000` | Starting capital               | `100000`            |
| `symbols`         | array  |        No |     `[]` | Symbols to trade (empty = all) | `["AAPL", "GOOGL"]` |

**Example Request**:

```bash
curl -X POST http://localhost:8000/api/v1/strategies/momentum_v1/backtest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "symbols": ["AAPL", "GOOGL", "MSFT"]
  }'
```

**Response**:

```json
{
  "backtest_id": "BT-20250115-001",
  "strategy_id": "momentum_v1",
  "status": "running",
  "progress": 0,
  "estimated_completion": "2025-01-15T10:35:00Z"
}
```

---

## WebSocket API

### Connect to WebSocket

**Endpoint**: `ws://localhost:8000/ws/market-data`

### Subscribe to Market Data

**Message Format**:

```json
{
  "action": "subscribe",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "interval": "1m"
}
```

### Receive Market Data

**Message Format**:

```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "timestamp": "2025-01-15T10:30:00Z",
  "price": 151.25,
  "volume": 1250,
  "bid": 151.24,
  "ask": 151.26
}
```

### Python WebSocket Example

```python
import asyncio
import websockets
import json

async def subscribe_market_data():
    uri = "ws://localhost:8000/ws/market-data"

    async with websockets.connect(uri) as websocket:
        # Subscribe
        await websocket.send(json.dumps({
            "action": "subscribe",
            "symbols": ["AAPL", "GOOGL"],
            "interval": "1m"
        }))

        # Receive data
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"{data['symbol']}: ${data['price']}")

asyncio.run(subscribe_market_data())
```

---

## Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400,
  "error_code": "INVALID_ORDER",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### HTTP Status Codes

| Code  | Meaning               | Description                       |
| ----- | --------------------- | --------------------------------- |
| `200` | OK                    | Request successful                |
| `201` | Created               | Resource created successfully     |
| `400` | Bad Request           | Invalid request parameters        |
| `401` | Unauthorized          | Missing or invalid authentication |
| `403` | Forbidden             | Insufficient permissions          |
| `404` | Not Found             | Resource not found                |
| `422` | Unprocessable Entity  | Validation error                  |
| `429` | Too Many Requests     | Rate limit exceeded               |
| `500` | Internal Server Error | Server error                      |
| `503` | Service Unavailable   | Service temporarily unavailable   |

### Common Error Codes

| Error Code              | Description               | Resolution                     |
| ----------------------- | ------------------------- | ------------------------------ |
| `INVALID_ORDER`         | Order validation failed   | Check order parameters         |
| `INSUFFICIENT_FUNDS`    | Not enough cash for order | Reduce order size or add funds |
| `MARKET_CLOSED`         | Market is closed          | Wait for market open           |
| `SYMBOL_NOT_FOUND`      | Symbol not recognized     | Verify symbol is correct       |
| `RATE_LIMIT_EXCEEDED`   | Too many requests         | Wait before retrying           |
| `AUTHENTICATION_FAILED` | Invalid credentials       | Check username/password        |
| `TOKEN_EXPIRED`         | JWT token expired         | Obtain new access token        |

---

## Rate Limiting

API requests are rate-limited to ensure fair usage:

| Tier             | Requests per Minute | Requests per Hour |
| ---------------- | ------------------- | ----------------- |
| **Free**         | 60                  | 1,000             |
| **Professional** | 300                 | 10,000            |
| **Enterprise**   | Unlimited           | Unlimited         |

**Rate Limit Headers**:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642243800
```

### Rate Limit Exceeded Response

```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "status_code": 429,
  "retry_after": 60
}
```

---

## API Versioning

AlphaMind API uses URL path versioning:

- Current version: `v1`
- Base path: `/api/v1/`
- When breaking changes are introduced, a new version will be created: `/api/v2/`

---

## Additional Resources

- **Interactive Docs**: Visit `/docs` endpoint for Swagger UI
- **Schema**: OpenAPI 3.0 schema available at `/openapi.json`
- **Examples**: See [EXAMPLES/](EXAMPLES/) for complete code examples
- **Client Libraries**: Python, JavaScript, and TypeScript clients available

## Next Steps

- **Configure API Keys**: See [CONFIGURATION.md](CONFIGURATION.md)
- **Try Examples**: Explore [EXAMPLES/api_usage.md](EXAMPLES/api_usage.md)
- **CLI Usage**: See [CLI.md](CLI.md) for command-line tools
