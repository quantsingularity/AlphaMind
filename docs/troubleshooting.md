# AlphaMind Troubleshooting Guide

Common issues and solutions for AlphaMind installation, configuration, and operation.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [API Issues](#api-issues)
- [Frontend Issues](#frontend-issues)
- [Database Issues](#database-issues)
- [Getting Help](#getting-help)

## Installation Issues

### Python Version Errors

**Problem**: `ERROR: Python 3.10 or higher is required`

**Solution**:

```bash
# Check Python version
python --version

# If version is <3.10, install Python 3.10+
## Ubuntu/Debian
sudo apt-get install python3.10 python3.10-venv

## macOS (Homebrew)
brew install python@3.10

## Use specific Python version
python3.10 -m venv venv
source venv/bin/activate
```

---

### Missing System Dependencies

**Problem**: `error: command 'gcc' failed` or similar compilation errors

**Solution**:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libssl-dev libffi-dev

# macOS
xcode-select --install

# Install specific libraries
## For QuantLib
sudo apt-get install -y libboost-all-dev  # Ubuntu/Debian
brew install boost                          # macOS
```

---

### Permission Denied Errors

**Problem**: `Permission denied` when running scripts

**Solution**:

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run specific script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Never use sudo with pip
pip install --user package-name  # Install for user only
```

---

### Virtual Environment Issues

**Problem**: Can't activate virtual environment

**Solution**:

```bash
# Linux/macOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# If activation fails, recreate venv
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

---

## Configuration Issues

### Environment Variables Not Loading

**Problem**: Configuration not applied, using defaults

**Solution**:

```bash
# Verify .env file exists
ls backend/.env

# If missing, copy from example
cp backend/.env.example backend/.env

# Edit with your values
nano backend/.env

# Verify environment variables loaded
python -c "import os; print(os.getenv('API_HOST', 'NOT SET'))"

# Explicitly load .env
from dotenv import load_dotenv
load_dotenv('backend/.env')
```

---

### Invalid Secret Key

**Problem**: `SECRET_KEY is required` or authentication fails

**Solution**:

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to backend/.env
echo "SECRET_KEY=your-generated-key-here" >> backend/.env

# Or use openssl
openssl rand -base64 32
```

---

### Database Connection Refused

**Problem**: `Connection refused` when accessing database

**Solution**:

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list                # macOS

# Start PostgreSQL
sudo systemctl start postgresql   # Linux
brew services start postgresql    # macOS

# Verify connection settings
psql -h localhost -U alphamind -d alphamind

# Check .env file
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=alphamind
POSTGRES_PASSWORD=your_password
POSTGRES_DB=alphamind

# Test connection from Python
python -c "
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    user='alphamind',
    password='your_password',
    database='alphamind'
)
print('Connection successful!')
conn.close()
"
```

---

## Runtime Errors

### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'module_name'`

**Solution**:

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install missing dependencies
pip install -r backend/requirements.txt

# Install backend package in development mode
cd backend
python setup.py develop

# For optional dependencies
pip install tensorflow torch  # ML models
pip install pymc arviz        # Bayesian models

# Verify installation
python -c "import backend; print('Success')"
```

---

### Import Errors

**Problem**: `ImportError: cannot import name 'X' from 'Y'`

**Solution**:

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add backend to path (temporary)
export PYTHONPATH="${PYTHONPATH}:/path/to/AlphaMind/backend"

# Or modify script
import sys
sys.path.insert(0, '/path/to/AlphaMind/backend')

# Reinstall package
cd backend
pip uninstall alphamind-backend
python setup.py develop
```

---

### API Server Won't Start

**Problem**: `OSError: [Errno 98] Address already in use`

**Solution**:

```bash
# Check what's using port 8000
sudo lsof -i :8000
# Or
sudo netstat -tulpn | grep 8000

# Kill process using port
kill -9 PID  # Replace PID with actual process ID

# Or use different port
export API_PORT=8001
uvicorn api.main:app --port 8001

# Change in .env
API_PORT=8001
```

---

### Test Failures

**Problem**: Tests fail with import or dependency errors

**Solution**:

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/AlphaMind/backend"

# Run tests from project root
cd /path/to/AlphaMind
pytest tests/

# Run specific test
pytest tests/test_order_manager.py -v

# Skip slow tests
pytest -m "not slow"

# Note: Some tests require full dependencies
pip install -r backend/requirements.txt  # Full install
```

---

## Performance Issues

### Slow API Responses

**Problem**: API endpoints responding slowly

**Diagnosis**:

```bash
# Check API logs
tail -f /var/log/alphamind/backend.log

# Enable profiling
export ENABLE_PROFILING=true

# Check database query performance
# Add to backend/.env
LOG_LEVEL=DEBUG

# Monitor with curl
time curl http://localhost:8000/api/v1/portfolio
```

**Solutions**:

```bash
# Increase workers
export API_WORKERS=4

# Enable Redis caching
CACHE_TTL_SECONDS=3600

# Optimize database queries
# Add indexes to frequently queried columns

# Use connection pooling
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

---

### High Memory Usage

**Problem**: AlphaMind using too much memory

**Solutions**:

```bash
# Monitor memory usage
htop
# Or
ps aux | grep python | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Reduce workers
API_WORKERS=2

# Limit model cache
MODEL_CACHE_DIR=/tmp/alphamind/models
# Clear cache
rm -rf /tmp/alphamind/models/*

# Use Dask for large datasets
# Instead of pandas.read_csv()
import dask.dataframe as dd
df = dd.read_csv('large_file.csv')
```

---

### Training Takes Too Long

**Problem**: ML model training is extremely slow

**Solutions**:

```bash
# Use GPU if available
# Check GPU
nvidia-smi

# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Reduce training data
# Use sampling
sampled_data = data.sample(frac=0.1)

# Use fewer epochs
EPOCHS=10  # Instead of 100

# Parallelize training
from joblib import Parallel, delayed
# Use parallel processing

# Use pre-trained models
# Load checkpoint instead of training from scratch
```

---

## API Issues

### CORS Errors

**Problem**: Browser shows CORS errors

**Solution**:

```bash
# Add frontend URL to CORS_ORIGINS
# In backend/.env
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# For development (not production!)
CORS_ORIGINS=*

# Restart API server
# Changes require restart
```

---

### Authentication Fails

**Problem**: JWT token invalid or expired

**Solution**:

```bash
# Check token expiration
JWT_EXPIRATION_HOURS=24

# Verify secret key is set
echo $SECRET_KEY
# Should not be empty

# Re-authenticate
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'

# Check token format
# Token should start with "Bearer "
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

---

### Rate Limit Exceeded

**Problem**: `429 Too Many Requests`

**Solution**:

```bash
# Check rate limit headers
curl -I http://localhost:8000/api/v1/portfolio

# Response headers:
# X-RateLimit-Limit: 60
# X-RateLimit-Remaining: 0
# X-RateLimit-Reset: 1642243800

# Wait until reset time
# Or increase rate limits (if you control the server)

# Implement exponential backoff
import time
def retry_with_backoff(func, max_retries=5):
    for i in range(max_retries):
        try:
            return func()
        except RateLimitError:
            time.sleep(2 ** i)
    raise
```

---

## Frontend Issues

### Cannot Connect to Backend

**Problem**: Frontend shows "Network Error"

**Solution**:

```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend API configuration
# web-frontend/.env
REACT_APP_API_URL=http://localhost:8000

# Restart frontend
cd web-frontend
npm start

# Check browser console for errors
# Open Developer Tools (F12)
```

---

### WebSocket Connection Fails

**Problem**: Real-time data not updating

**Solution**:

```bash
# Test WebSocket connection
# Install wscat
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8000/ws/market-data

# Check WebSocket URL in frontend code
const ws = new WebSocket('ws://localhost:8000/ws/market-data');

# Verify backend WebSocket is enabled
# Check api/main.py includes WebSocket routes
```

---

## Database Issues

### Database Migration Errors

**Problem**: Migration fails or database schema out of sync

**Solution**:

```bash
# Check migration status
./scripts/db_migrate.sh status

# Run pending migrations
./scripts/db_migrate.sh upgrade

# If migration fails, rollback
./scripts/db_migrate.sh downgrade

# Reset database (CAUTION: destroys data)
dropdb alphamind
createdb alphamind
./scripts/db_migrate.sh upgrade
```

---

### InfluxDB Connection Issues

**Problem**: Cannot connect to InfluxDB

**Solution**:

```bash
# Check InfluxDB is running
influx ping

# Verify configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=alphamind
INFLUXDB_BUCKET=market_data

# Create token if missing
influx auth create \
  --org alphamind \
  --read-buckets \
  --write-buckets

# Test connection
influx bucket list --org alphamind
```

---

## Docker Issues

### Container Won't Start

**Problem**: Docker container exits immediately

**Solution**:

```bash
# Check container logs
docker logs alphamind-backend

# Check container status
docker ps -a

# Restart container
docker restart alphamind-backend

# Rebuild image
docker-compose build --no-cache
docker-compose up -d

# Check docker-compose configuration
docker-compose config
```

---

### Port Conflicts

**Problem**: "Port is already allocated"

**Solution**:

```bash
# Find process using port
sudo lsof -i :8000

# Stop conflicting service
docker stop $(docker ps -q --filter "publish=8000")

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Map host 8001 to container 8000
```

---

## Getting Help

### Diagnostic Information

When seeking help, provide:

```bash
# System information
uname -a
python --version
node --version
docker --version

# AlphaMind version
cd /path/to/AlphaMind
git describe --tags

# Error logs
tail -50 /var/log/alphamind/backend.log

# Configuration (redact secrets!)
cat backend/.env | grep -v "PASSWORD\|SECRET\|KEY"

# Test results
pytest --collect-only
```

### Common Commands

```bash
# Reset everything (CAUTION)
./scripts/setup_environment.sh --force-reinstall

# Check health
curl http://localhost:8000/health

# View logs
tail -f logs/alphamind.log

# Restart services
./scripts/run_alphamind.sh
```

### Support Channels

1. **Documentation**: Check [docs/](README.md) first
2. **GitHub Issues**: Search [existing issues](https://github.com/abrar2030/AlphaMind/issues)
3. **Create Issue**: Use issue templates for bug reports/feature requests
4. **Community**: GitHub Discussions for questions

### Issue Template

When creating an issue:

```markdown
**Environment**

- OS: Ubuntu 20.04
- Python: 3.10.8
- AlphaMind version: 1.0.0

**Problem Description**
Clear description of the issue

**Steps to Reproduce**

1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Logs**
```

Paste relevant logs here

```

**Attempted Solutions**
What you've already tried
```

## Prevention Best Practices

1. **Keep dependencies updated**: `pip install -U -r requirements.txt`
2. **Run tests before deploying**: `./scripts/run_tests.sh`
3. **Monitor logs**: Set up log monitoring in production
4. **Use version control**: Commit configuration changes
5. **Document changes**: Update documentation when modifying code
6. **Test in staging**: Never test directly in production
7. **Backup data**: Regular database backups

## Next Steps

- **Configuration**: See [CONFIGURATION.md](CONFIGURATION.md) for proper setup
- **Usage**: Check [USAGE.md](USAGE.md) for correct usage patterns
- **Architecture**: Understand system design in [ARCHITECTURE.md](ARCHITECTURE.md)
- **Contributing**: Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting fixes
