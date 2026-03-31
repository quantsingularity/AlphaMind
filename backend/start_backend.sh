#!/usr/bin/env bash
set -euo pipefail

echo "=================================="
echo "AlphaMind Backend Startup"
echo "=================================="
echo ""

# ── Virtual environment ────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies (this may take a few minutes)..."
pip install --quiet -r requirements.txt

# ── Environment variables ──────────────────────────────────────────────────
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    # Use set -a / set +a rather than xargs which breaks on special characters
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
else
    echo "Warning: .env file not found."
    echo "Please copy .env.example to .env and configure appropriately."
fi

# ── Derived configuration ──────────────────────────────────────────────────
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${WORKERS:-1}"

echo ""
echo "Starting FastAPI backend with Uvicorn..."
echo "  Host    : http://${API_HOST}:${API_PORT}"
echo "  Docs    : http://${API_HOST}:${API_PORT}/docs"
echo "  ReDoc   : http://${API_HOST}:${API_PORT}/redoc"
echo "  Health  : http://${API_HOST}:${API_PORT}/health"
echo ""
echo "Press CTRL+C to stop."
echo ""

python3 -m uvicorn api.main:app \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"
