# AlphaMind Backend - Setup and Installation Guide

## Prerequisites

- Python 3.10 or 3.11
- Virtual environment tool (venv)
- At least 4GB RAM
- Internet connection for downloading dependencies

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and configure your settings
# nano .env  # or use your preferred editor
```

### 3. Start the Backend

```bash
# Using the startup script (recommended)
./start_backend.sh

# Or manually:
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- API Base URL: http://localhost:8000
- Interactive Documentation: http://localhost:8000/docs
- Alternative Documentation: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## Minimal Installation (Core API Only)

If you only need the core API without advanced features:

```bash
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv python-multipart numpy pandas pyyaml
```

## Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Docker Setup (Alternative)

```bash
# Build image
docker build -t alphamind-backend .

# Run container
docker run -p 8000:8000 alphamind-backend
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Port Already in Use

If port 8000 is already in use:

```bash
# Change port in .env file
API_PORT=8001

# Or specify when running
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Permission Issues

Make start script executable:

```bash
chmod +x start_backend.sh
```

## Environment Variables

Key environment variables (see .env.example for full list):

- `API_HOST`: API host address (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `SECRET_KEY`: Security secret key (change in production!)

## Optional Services

The backend can work with optional external services:

- **Redis**: For caching (optional)
- **PostgreSQL/MySQL**: For persistent storage (optional, uses in-memory by default)
- **Kafka**: For real-time data streaming (optional)

See docker-compose.yml (if provided) for running these services locally.

## Development Mode

For development with auto-reload:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## Production Considerations

1. Change `SECRET_KEY` in .env
2. Set `API_DEBUG=false`
3. Use proper CORS origins in .env
4. Use production WSGI server (Gunicorn with Uvicorn workers)
5. Set up proper logging
6. Configure external database
7. Use HTTPS/TLS
