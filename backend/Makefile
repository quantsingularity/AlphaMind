.PHONY: help install install-minimal install-dev clean test run dev lint format

help:
	@echo "AlphaMind Backend - Available Commands"
	@echo "======================================"
	@echo "make install          - Install all dependencies"
	@echo "make install-minimal  - Install minimal dependencies (core API only)"
	@echo "make install-dev      - Install with development tools"
	@echo "make run              - Start the backend server"
	@echo "make dev              - Start server in development mode with auto-reload"
	@echo "make test             - Run tests"
	@echo "make test-cov         - Run tests with coverage report"
	@echo "make lint             - Run code quality checks"
	@echo "make format           - Format code with black"
	@echo "make clean            - Clean up temporary files and caches"
	@echo ""

install:
	@echo "Installing AlphaMind backend dependencies..."
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Installation complete!"

install-minimal:
	@echo "Installing minimal dependencies..."
	@pip install --upgrade pip
	@pip install -r requirements-minimal.txt
	@echo "Minimal installation complete!"

install-dev:
	@echo "Installing with development tools..."
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Development installation complete!"

run:
	@echo "Starting AlphaMind backend..."
	@python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

dev:
	@echo "Starting AlphaMind backend in development mode..."
	@python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

test:
	@echo "Running tests..."
	@pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	@pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Running code quality checks..."
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting code..."
	@black . --line-length 100
	@echo "Code formatted!"

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf .mypy_cache/ 2>/dev/null || true
	@echo "Cleanup complete!"
