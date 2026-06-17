# Example: Using the REST API

These snippets assume the backend is running at `http://localhost:8000` (`python -m main` from `code/backend`). Interactive docs are at `/docs`.

## Shell (curl)

```bash
BASE=http://localhost:8000

# Register and capture the token
TOKEN=$(curl -s -X POST $BASE/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"trader@example.com","name":"Ada","password":"a-strong-password"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["token"])')

# Portfolio summary
curl -s $BASE/api/v1/portfolio/ | python -m json.tool

# Strategies, then an equity curve for the first one
curl -s $BASE/api/v1/strategies/ | python -m json.tool

# Market data
curl -s $BASE/api/v1/market-data/quotes | python -m json.tool
curl -s "$BASE/api/v1/market-data/historical/AAPL?days=30" | python -m json.tool

# Place a limit order, then cancel it
ORDER=$(curl -s -X POST $BASE/api/v1/trading/orders \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AAPL","side":"BUY","quantity":10,"orderType":"LIMIT","price":150}')
echo "$ORDER" | python -m json.tool
ID=$(echo "$ORDER" | python -c 'import sys,json; print(json.load(sys.stdin)["id"])')
curl -s -X DELETE $BASE/api/v1/trading/orders/$ID -o /dev/null -w '%{http_code}\n'
```

## Python (requests)

```python
import requests

BASE = "http://localhost:8000"

# Auth
r = requests.post(f"{BASE}/api/auth/login",
                  json={"email": "trader@example.com", "password": "a-strong-password"})
token = r.json()["token"]
headers = {"Authorization": f"Bearer {token}"}

# Portfolio and risk
portfolio = requests.get(f"{BASE}/api/v1/portfolio/").json()
risk = requests.get(f"{BASE}/api/v1/risk/metrics").json()
print("Total value:", portfolio["totalValue"])
print("VaR:", risk["var"], "Sharpe:", risk["sharpeRatio"])

# Run a backtest
strategies = requests.get(f"{BASE}/api/v1/strategies/").json()
result = requests.post(f"{BASE}/api/v1/backtest/", json={
    "strategyId": strategies[0]["id"],
    "startDate": "2023-01-01",
    "endDate": "2024-01-01",
    "initialCapital": 100000,
}).json()
print("Backtest return:", result["totalReturn"], "Sharpe:", result["sharpeRatio"])
```

See [../API.md](../API.md) for the complete set of endpoints and field shapes.
