# AlphaMind Configuration Guide

Complete guide to configuring AlphaMind for development, testing, and production environments.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Environment Files](#environment-files)
- [Configuration Options](#configuration-options)
- [Security Configuration](#security-configuration)
- [Database Configuration](#database-configuration)
- [External Services](#external-services)
- [Advanced Configuration](#advanced-configuration)

## Configuration Overview

AlphaMind uses multiple configuration sources with the following precedence (highest to lowest):

1. **Environment variables** (highest priority)
2. **`.env` files** in component directories
3. **Default values** in code (lowest priority)

**Configuration Locations:**

| Component           | Configuration File                  | Purpose                         |
| ------------------- | ----------------------------------- | ------------------------------- |
| **Backend**         | `backend/.env`                      | API, database, trading settings |
| **Web Frontend**    | `web-frontend/.env`                 | API endpoints, app settings     |
| **Mobile Frontend** | `mobile-frontend/.env`              | API endpoints, app settings     |
| **Infrastructure**  | `infrastructure/terraform/*.tfvars` | Cloud infrastructure            |

## Environment Files

### Create Configuration Files

```bash
# Backend configuration
cp backend/.env.example backend/.env

# Web frontend configuration
cp web-frontend/.env.example web-frontend/.env

# Mobile frontend configuration
cp mobile-frontend/.env.example mobile-frontend/.env

# Edit files with your settings
nano backend/.env
```

### Environment-Specific Configuration

Create environment-specific files:

```bash
# Development
backend/.env.development

# Testing
backend/.env.test

# Production
backend/.env.production
```

## Configuration Options

### API Configuration

Core API server settings:

| Option        | Type    |   Default | Description                        | Where to Set          |
| ------------- | ------- | --------: | ---------------------------------- | --------------------- |
| `API_HOST`    | string  | `0.0.0.0` | API server host address            | `backend/.env`        |
| `API_PORT`    | integer |    `8000` | API server port                    | `backend/.env`        |
| `API_DEBUG`   | boolean |    `true` | Enable debug mode                  | `backend/.env`        |
| `API_RELOAD`  | boolean |    `true` | Enable auto-reload on code changes | `backend/.env`        |
| `API_WORKERS` | integer |       `1` | Number of API worker processes     | `backend/.env` or ENV |

**Example Configuration:**

```bash
# Development
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
API_RELOAD=true

# Production
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=false
API_WORKERS=4
```

---

### Security Configuration

Authentication and security settings:

| Option                 | Type    |        Default | Description                            | Where to Set   |
| ---------------------- | ------- | -------------: | -------------------------------------- | -------------- |
| `SECRET_KEY`           | string  | **(required)** | Secret key for JWT signing             | `backend/.env` |
| `JWT_ALGORITHM`        | string  |        `HS256` | JWT signing algorithm                  | `backend/.env` |
| `JWT_EXPIRATION_HOURS` | integer |           `24` | JWT token expiration time (hours)      | `backend/.env` |
| `CORS_ORIGINS`         | string  |            `*` | Allowed CORS origins (comma-separated) | `backend/.env` |

**Generate Secret Key:**

```bash
# Generate secure random key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Or use openssl
openssl rand -base64 32
```

**Example Configuration:**

```bash
# NEVER use default keys in production!
SECRET_KEY=your-randomly-generated-secret-key-change-this

# JWT Configuration
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# CORS - restrict in production
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

---

### Database Configuration

Database connection settings:

| Option              | Type    |     Default | Description              | Where to Set          |
| ------------------- | ------- | ----------: | ------------------------ | --------------------- |
| `DB_USERNAME`       | string  |      `root` | Database username        | `backend/.env` or ENV |
| `DB_PASSWORD`       | string  |  `password` | Database password        | `backend/.env` or ENV |
| `DB_HOST`           | string  | `localhost` | Database host            | `backend/.env`        |
| `DB_PORT`           | integer |      `3306` | Database port (MySQL)    | `backend/.env`        |
| `DB_NAME`           | string  | `alphamind` | Database name            | `backend/.env`        |
| `POSTGRES_USER`     | string  | `alphamind` | PostgreSQL username      | `backend/.env` or ENV |
| `POSTGRES_PASSWORD` | string  |  `password` | PostgreSQL password      | `backend/.env` or ENV |
| `POSTGRES_HOST`     | string  | `localhost` | PostgreSQL host          | `backend/.env`        |
| `POSTGRES_PORT`     | integer |      `5432` | PostgreSQL port          | `backend/.env`        |
| `POSTGRES_DB`       | string  | `alphamind` | PostgreSQL database name | `backend/.env`        |

**Example Configuration:**

```bash
# PostgreSQL (Recommended)
POSTGRES_USER=alphamind
POSTGRES_PASSWORD=secure_password_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=alphamind

# Connection pool settings (optional)
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
```

---

### Redis Configuration

Caching and session storage:

| Option              | Type    |     Default | Description                  | Where to Set          |
| ------------------- | ------- | ----------: | ---------------------------- | --------------------- |
| `REDIS_HOST`        | string  | `localhost` | Redis server host            | `backend/.env`        |
| `REDIS_PORT`        | integer |      `6379` | Redis server port            | `backend/.env`        |
| `REDIS_DB`          | integer |         `0` | Redis database number        | `backend/.env`        |
| `REDIS_PASSWORD`    | string  |        `""` | Redis password (if required) | `backend/.env` or ENV |
| `CACHE_TTL_SECONDS` | integer |      `3600` | Default cache TTL            | `backend/.env`        |

**Example Configuration:**

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty if no password
CACHE_TTL_SECONDS=3600
```

---

### InfluxDB Configuration

Time-series data storage:

| Option            | Type   |                 Default | Description                   | Where to Set          |
| ----------------- | ------ | ----------------------: | ----------------------------- | --------------------- |
| `INFLUXDB_URL`    | string | `http://localhost:8086` | InfluxDB URL                  | `backend/.env`        |
| `INFLUXDB_TOKEN`  | string |          **(required)** | InfluxDB authentication token | `backend/.env` or ENV |
| `INFLUXDB_ORG`    | string |             `alphamind` | InfluxDB organization         | `backend/.env`        |
| `INFLUXDB_BUCKET` | string |           `market_data` | Default bucket name           | `backend/.env`        |

**Example Configuration:**

```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-influxdb-token-here
INFLUXDB_ORG=alphamind
INFLUXDB_BUCKET=market_data
```

---

### Kafka Configuration

Message streaming (optional):

| Option                    | Type   |          Default | Description            | Where to Set   |
| ------------------------- | ------ | ---------------: | ---------------------- | -------------- |
| `KAFKA_BOOTSTRAP_SERVERS` | string | `localhost:9092` | Kafka broker addresses | `backend/.env` |
| `KAFKA_GROUP_ID`          | string |      `alphamind` | Consumer group ID      | `backend/.env` |

**Example Configuration:**

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=alphamind
```

---

## External Services

### Market Data API Keys

Configure external market data providers:

| Option                  | Type   | Default | Description           | Where to Set          |
| ----------------------- | ------ | ------: | --------------------- | --------------------- |
| `ALPHA_VANTAGE_API_KEY` | string |    `""` | Alpha Vantage API key | `backend/.env` or ENV |
| `IEX_CLOUD_API_KEY`     | string |    `""` | IEX Cloud API key     | `backend/.env` or ENV |
| `POLYGON_API_KEY`       | string |    `""` | Polygon.io API key    | `backend/.env` or ENV |
| `FRED_API_KEY`          | string |    `""` | FRED API key          | `backend/.env` or ENV |
| `BLOOMBERG_API_KEY`     | string |    `""` | Bloomberg API key     | `backend/.env` or ENV |

**Example Configuration:**

```bash
# Market Data Providers
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
IEX_CLOUD_API_KEY=your_iex_cloud_key
POLYGON_API_KEY=your_polygon_key
FRED_API_KEY=your_fred_key
```

**Obtaining API Keys:**

| Provider      | Free Tier      | Sign Up URL                                       |
| ------------- | -------------- | ------------------------------------------------- |
| Alpha Vantage | 5 requests/min | https://www.alphavantage.co/support/#api-key      |
| IEX Cloud     | Limited        | https://iexcloud.io/pricing/                      |
| Polygon.io    | Limited        | https://polygon.io/pricing                        |
| FRED          | Unlimited      | https://fred.stlouisfed.org/docs/api/api_key.html |

---

### Cloud Provider Configuration

#### Google Cloud Platform (GCP)

| Option                 | Type   | Default | Description                  | Where to Set          |
| ---------------------- | ------ | ------: | ---------------------------- | --------------------- |
| `GCP_PROJECT_ID`       | string |    `""` | GCP project ID               | `backend/.env` or ENV |
| `GCP_CREDENTIALS_PATH` | string |    `""` | Path to service account JSON | `backend/.env` or ENV |

**Example Configuration:**

```bash
GCP_PROJECT_ID=alphamind-prod
GCP_CREDENTIALS_PATH=/path/to/credentials.json
```

#### Amazon Web Services (AWS)

| Option                  | Type   |     Default | Description    | Where to Set      |
| ----------------------- | ------ | ----------: | -------------- | ----------------- |
| `AWS_ACCESS_KEY_ID`     | string |        `""` | AWS access key | ENV only (secure) |
| `AWS_SECRET_ACCESS_KEY` | string |        `""` | AWS secret key | ENV only (secure) |
| `AWS_REGION`            | string | `us-east-1` | AWS region     | `backend/.env`    |

**Example Configuration:**

```bash
# Set via environment (more secure)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
```

---

## Trading Configuration

Trading-specific settings:

| Option              | Type    |    Default | Description                       | Where to Set   |
| ------------------- | ------- | ---------: | --------------------------------- | -------------- |
| `INITIAL_CAPITAL`   | float   | `100000.0` | Starting capital for backtests    | `backend/.env` |
| `MAX_POSITION_SIZE` | float   |      `0.1` | Maximum position size (10% = 0.1) | `backend/.env` |
| `RISK_FREE_RATE`    | float   |     `0.02` | Risk-free rate for calculations   | `backend/.env` |
| `MAX_WORKERS`       | integer |        `4` | Maximum parallel workers          | `backend/.env` |

**Example Configuration:**

```bash
# Trading Settings
INITIAL_CAPITAL=100000.0
MAX_POSITION_SIZE=0.1  # Max 10% per position
RISK_FREE_RATE=0.02    # 2% annual risk-free rate

# Performance
MAX_WORKERS=4
```

---

## Logging Configuration

Logging and monitoring:

| Option       | Type   | Default | Description                                        | Where to Set          |
| ------------ | ------ | ------: | -------------------------------------------------- | --------------------- |
| `LOG_LEVEL`  | string |  `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `backend/.env` or ENV |
| `LOG_FILE`   | string |    `""` | Log file path (empty = stdout)                     | `backend/.env`        |
| `LOG_FORMAT` | string |  `json` | Log format: `json`, `text`                         | `backend/.env`        |

**Example Configuration:**

```bash
# Development
LOG_LEVEL=DEBUG
LOG_FILE=  # Log to stdout

# Production
LOG_LEVEL=INFO
LOG_FILE=/var/log/alphamind/backend.log
LOG_FORMAT=json
```

---

## Model Configuration

AI/ML model settings:

| Option                   | Type    |                 Default | Description           | Where to Set   |
| ------------------------ | ------- | ----------------------: | --------------------- | -------------- |
| `MODEL_TRAINING_ENABLED` | boolean |                  `true` | Enable model training | `backend/.env` |
| `MODEL_CACHE_DIR`        | string  | `/tmp/alphamind/models` | Model cache directory | `backend/.env` |
| `MODEL_AUTO_UPDATE`      | boolean |                 `false` | Auto-update models    | `backend/.env` |

**Example Configuration:**

```bash
MODEL_TRAINING_ENABLED=true
MODEL_CACHE_DIR=/var/cache/alphamind/models
MODEL_AUTO_UPDATE=false
```

---

## Feature Flags

Enable/disable specific features:

| Option                       | Type    | Default | Description                      | Where to Set   |
| ---------------------------- | ------- | ------: | -------------------------------- | -------------- |
| `SATELLITE_DATA_ENABLED`     | boolean | `false` | Enable satellite data processing | `backend/.env` |
| `SEC_FILINGS_ENABLED`        | boolean |  `true` | Enable SEC filings analysis      | `backend/.env` |
| `SENTIMENT_ANALYSIS_ENABLED` | boolean |  `true` | Enable sentiment analysis        | `backend/.env` |

**Example Configuration:**

```bash
# Alternative Data Features
SATELLITE_DATA_ENABLED=false  # Requires additional setup
SEC_FILINGS_ENABLED=true
SENTIMENT_ANALYSIS_ENABLED=true
```

---

## Backtesting Configuration

Backtesting parameters:

| Option                | Type   |      Default | Description                      | Where to Set   |
| --------------------- | ------ | -----------: | -------------------------------- | -------------- |
| `BACKTEST_START_DATE` | string | `2020-01-01` | Backtest start date (YYYY-MM-DD) | `backend/.env` |
| `BACKTEST_END_DATE`   | string | `2023-12-31` | Backtest end date (YYYY-MM-DD)   | `backend/.env` |
| `BACKTEST_COMMISSION` | float  |      `0.001` | Commission rate (0.1% = 0.001)   | `backend/.env` |

**Example Configuration:**

```bash
BACKTEST_START_DATE=2020-01-01
BACKTEST_END_DATE=2023-12-31
BACKTEST_COMMISSION=0.001  # 0.1% commission
```

---

## Performance Configuration

System performance tuning:

| Option                    | Type    | Default | Description                    | Where to Set   |
| ------------------------- | ------- | ------: | ------------------------------ | -------------- |
| `REQUEST_TIMEOUT_SECONDS` | integer |    `30` | HTTP request timeout           | `backend/.env` |
| `MAX_CONNECTIONS`         | integer |   `100` | Maximum concurrent connections | `backend/.env` |
| `ENABLE_PROFILING`        | boolean | `false` | Enable performance profiling   | `backend/.env` |

**Example Configuration:**

```bash
REQUEST_TIMEOUT_SECONDS=30
MAX_CONNECTIONS=100
ENABLE_PROFILING=false  # Enable for debugging only
```

---

## Configuration Validation

Validate your configuration:

```bash
# Check configuration
cd backend
python -c "
from core.config import config_manager
config_manager.load_from_env('ALPHAMIND_')
errors = config_manager.validate()
if errors:
    print('Configuration errors:', errors)
else:
    print('Configuration valid!')
"
```

---

## Best Practices

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use environment variables for secrets** - Don't put secrets in files
3. **Use different keys per environment** - Dev, staging, prod should differ
4. **Rotate secrets regularly** - Especially in production
5. **Document custom settings** - Add comments in `.env` files
6. **Validate configuration** - Run validation before deployment
7. **Use secure storage** - Consider HashiCorp Vault, AWS Secrets Manager

---

## Environment-Specific Examples

### Development Environment

```bash
# backend/.env.development
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
API_RELOAD=true
LOG_LEVEL=DEBUG
SECRET_KEY=dev-secret-key-not-for-production
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Production Environment

```bash
# backend/.env.production
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=false
API_WORKERS=4
LOG_LEVEL=INFO
LOG_FILE=/var/log/alphamind/backend.log
SECRET_KEY=production-secret-from-secure-storage
CORS_ORIGINS=https://app.alphamind.ai
```

---

## Troubleshooting Configuration

### Check Current Configuration

```python
from core.config import config_manager
print(config_manager.get_all())
```

### Common Issues

| Issue                             | Solution                             |
| --------------------------------- | ------------------------------------ |
| "Configuration validation failed" | Check required fields are set        |
| "Database connection refused"     | Verify database credentials and host |
| "API key invalid"                 | Check external service API keys      |
| "Secret key too weak"             | Generate secure random key           |
| "CORS error in browser"           | Add frontend URL to `CORS_ORIGINS`   |

---

## Next Steps

- **Installation**: See [INSTALLATION.md](INSTALLATION.md) for setup
- **Usage**: See [USAGE.md](USAGE.md) for running AlphaMind
- **Security**: Review security best practices
- **Deployment**: See [deployment.md](../docs/deployment.md) for production setup
