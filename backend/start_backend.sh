#!/bin/bash
# AlphaMind Backend Startup Script

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

echo ""
echo "Starting Flask backend server..."
echo "Server will be available at: http://localhost:5000"
echo "Health check endpoint: http://localhost:5000/health"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Start the server
python3 src/main.py
