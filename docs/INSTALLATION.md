# AlphaMind Installation Guide

This guide covers all installation methods for AlphaMind across different platforms and deployment scenarios.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Quick Setup (Recommended)](#quick-setup-recommended)
  - [Manual Installation](#manual-installation)
  - [Docker Installation](#docker-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verification](#verification)
- [Next Steps](#next-steps)

## System Requirements

| Component   | Minimum                          | Recommended              | Notes                                      |
| ----------- | -------------------------------- | ------------------------ | ------------------------------------------ |
| **OS**      | Linux, macOS 10.15+, Windows 10+ | Ubuntu 20.04+, macOS 12+ | WSL2 required for Windows                  |
| **Python**  | 3.10                             | 3.10 or 3.11             | Python 3.12 supported but not fully tested |
| **Node.js** | 16.x                             | 18.x LTS                 | Required for frontend components           |
| **Memory**  | 8 GB RAM                         | 16 GB+ RAM               | 32 GB for ML model training                |
| **Storage** | 10 GB free                       | 50 GB+ free              | Additional space for market data           |
| **CPU**     | 4 cores                          | 8+ cores                 | Multi-core for parallel processing         |
| **GPU**     | None (optional)                  | CUDA-compatible GPU      | NVIDIA GPU for ML training acceleration    |
| **Network** | Stable internet                  | Low-latency connection   | Required for real-time data feeds          |

### Additional Requirements

- **C++ Compiler**: Required for QuantLib and certain dependencies
  - Linux: `gcc` 9.0+
  - macOS: Xcode Command Line Tools or `gcc` via Homebrew
  - Windows: Visual Studio 2019+ or MinGW-w64
- **Docker** (optional): For containerized deployment
- **Git**: For cloning the repository

## Installation Methods

### Quick Setup (Recommended)

The automated setup script handles all dependencies and configuration:

```bash
# 1. Clone the repository
git clone https://github.com/quantsingularity/AlphaMind.git
cd AlphaMind

# 2. Run the setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh --type development

# 3. Activate the environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

**Script Options:**

| Option              | Description                                        | Default       |
| ------------------- | -------------------------------------------------- | ------------- |
| `--type TYPE`       | Setup type: `development`, `testing`, `production` | `development` |
| `--verbose`         | Enable detailed output                             | Disabled      |
| `--skip-python`     | Skip Python environment setup                      | Disabled      |
| `--skip-node`       | Skip Node.js environment setup                     | Disabled      |
| `--skip-docker`     | Skip Docker setup verification                     | Disabled      |
| `--force-reinstall` | Force reinstall all components                     | Disabled      |

**Example - Production Setup:**

```bash
./scripts/setup_environment.sh --type production --verbose
```

### Manual Installation

#### Backend Setup

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install core dependencies
pip install -r requirements-minimal.txt

# 5. Install full dependencies (optional)
pip install -r requirements.txt

# 6. Install backend package in development mode
python setup.py develop

# 7. Verify installation
python -c "import backend; print('Backend installed successfully')"
```

#### Web Frontend Setup

```bash
# 1. Navigate to web frontend directory
cd web-frontend

# 2. Install dependencies
npm install

# 3. Verify installation
npm run build
```

#### Mobile Frontend Setup (Optional)

```bash
# 1. Navigate to mobile frontend directory
cd mobile-frontend

# 2. Install dependencies
npm install

# 3. For iOS (macOS only)
cd ios && pod install && cd ..

# 4. Verify installation
npm run doctor  # React Native diagnostics
```

### Docker Installation

Docker provides isolated, reproducible deployments:

```bash
# 1. Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# 2. Build containers
./scripts/docker_build.sh

# 3. Start services
docker-compose up -d

# 4. Verify services
docker-compose ps

# 5. View logs
docker-compose logs -f alphamind-backend
```

**Docker Compose Services:**

| Service             | Port | Description            |
| ------------------- | ---- | ---------------------- |
| `alphamind-backend` | 8000 | FastAPI backend server |
| `alphamind-web`     | 3000 | React web frontend     |
| `postgres`          | 5432 | PostgreSQL database    |
| `redis`             | 6379 | Redis cache            |
| `influxdb`          | 8086 | Time-series database   |
| `kafka`             | 9092 | Message broker         |

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential \
    libssl-dev libffi-dev \
    nodejs npm \
    git

# Install C++ compiler for QuantLib
sudo apt-get install -y g++ cmake

# Proceed with installation
./scripts/setup_environment.sh
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.10 node git

# Install Xcode Command Line Tools
xcode-select --install

# Proceed with installation
./scripts/setup_environment.sh
```

### Windows (WSL2)

```powershell
# 1. Install WSL2 (run in PowerShell as Administrator)
wsl --install

# 2. Restart computer

# 3. Launch Ubuntu from Start Menu

# 4. Inside WSL2 Ubuntu terminal:
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv build-essential nodejs npm git

# 5. Clone and install AlphaMind
git clone https://github.com/quantsingularity/AlphaMind.git
cd AlphaMind
./scripts/setup_environment.sh
```

## Installation Matrix

| OS / Platform       | Recommended Installation Method | Notes                                      |
| ------------------- | ------------------------------- | ------------------------------------------ |
| **Ubuntu 20.04+**   | Quick setup script              | Native support, best performance           |
| **macOS 12+**       | Quick setup script              | Native support, requires Xcode CLI         |
| **Windows 10/11**   | WSL2 + Quick setup              | Use WSL2 for best compatibility            |
| **Docker (Any OS)** | Docker Compose                  | Isolated environment, slower performance   |
| **Cloud (GCP/AWS)** | Docker or manual                | See [deployment.md](../docs/deployment.md) |

## Configuration

After installation, configure AlphaMind:

```bash
# 1. Copy example configuration
cp backend/.env.example backend/.env

# 2. Edit configuration (see CONFIGURATION.md for details)
nano backend/.env  # or use your preferred editor

# 3. Set required API keys (minimum)
# - API_HOST and API_PORT
# - SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
# - Database credentials
# - External API keys (Alpha Vantage, IEX Cloud, etc.)
```

See [CONFIGURATION.md](CONFIGURATION.md) for complete configuration documentation.

## Verification

### Verify Backend

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Check Python packages
pip list | grep -E "fastapi|pandas|numpy"

# Run health check
cd backend
python -c "from api.main import app; print('✓ Backend API imports successfully')"

# Start API server (test)
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/docs to see API documentation
```

### Verify Frontend

```bash
cd web-frontend

# Check Node packages
npm list --depth=0 | grep -E "react|typescript"

# Start development server (test)
npm run dev
# Visit http://localhost:3000 to see web interface
```

### Run Tests

```bash
# Quick test suite
./scripts/run_tests.sh --type unit --fail-fast

# Full test suite (takes several minutes)
./scripts/run_tests.sh --type all
```

## Troubleshooting Installation

### Python Version Issues

```bash
# If default python is not 3.10+
python3.10 -m venv venv
source venv/bin/activate
python --version  # Should show 3.10+
```

### Missing System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-dev build-essential

# macOS
xcode-select --install
brew install python@3.10
```

### QuantLib Compilation Errors

```bash
# Ubuntu/Debian
sudo apt-get install -y libboost-all-dev

# macOS
brew install boost

# Then reinstall
pip install --no-cache-dir QuantLib
```

### Permission Errors

```bash
# Linux/macOS - Make scripts executable
chmod +x scripts/*.sh

# Avoid using sudo with pip
pip install --user package-name
```

### Docker Issues

```bash
# Reset Docker environment
docker-compose down -v
docker system prune -a
./scripts/docker_build.sh
```

For additional troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Next Steps

After successful installation:

1. **Configure**: Complete configuration in `backend/.env` - see [CONFIGURATION.md](CONFIGURATION.md)
2. **Learn**: Read [USAGE.md](USAGE.md) for common workflows
3. **Explore**: Try examples in [EXAMPLES/](EXAMPLES/)
4. **Develop**: Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
5. **Deploy**: See [deployment.md](../docs/deployment.md) for production deployment

## Getting Help

- **Issues**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Documentation**: Browse [docs/README.md](README.md)
- **Community**: Open an issue at [GitHub Issues](https://github.com/quantsingularity/AlphaMind/issues)
