#!/bin/bash
# AlphaMind Backend Startup Script (FastAPI)

echo "=================================="
echo "AlphaMind Backend Startup"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Using default configuration."
    echo "Please copy .env.example to .env and configure appropriately."
fi

echo ""
echo "Starting FastAPI backend server with Uvicorn..."
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Alternative docs: http://localhost:8000/redoc"
echo "Health check endpoint: http://localhost:8000/health"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Start the server
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
