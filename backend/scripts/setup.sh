#!/bin/bash
# AlphaMind Backend Setup Script

set -e

echo "=================================="
echo "AlphaMind Backend Setup"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

REQUIRED_MAJOR=3
REQUIRED_MINOR=10

MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Ask user which installation type
echo ""
echo "Select installation type:"
echo "1) Minimal (core API only, fastest)"
echo "2) Full (all features, may take longer)"
read -p "Enter choice [1-2] (default: 1): " INSTALL_TYPE
INSTALL_TYPE=${INSTALL_TYPE:-1}

if [ "$INSTALL_TYPE" = "1" ]; then
    echo "Installing minimal dependencies..."
    pip install -r requirements-minimal.txt --quiet
else
    echo "Installing full dependencies (this may take several minutes)..."
    pip install -r requirements.txt --quiet
fi

# Setup environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file to configure your environment"
else
    echo ".env file already exists"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data

# Make scripts executable
chmod +x start_backend.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file to configure your environment"
echo "2. Run: ./start_backend.sh (or: make run)"
echo "3. Access API at: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  ./start_backend.sh  - Start the backend"
echo "  make help           - Show all available commands"
echo "  make test           - Run tests"
echo ""
