# AlphaMind CLI Reference

Command-line interface tools and scripts for AlphaMind.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Commands](#core-commands)
- [Script Reference](#script-reference)
- [Environment Variables](#environment-variables)

## Overview

AlphaMind provides several CLI tools for setup, testing, deployment, and operations. All scripts are located in the `scripts/` directory.

## Installation

Make all scripts executable:

```bash
chmod +x scripts/*.sh
```

## Core Commands

### setup_environment.sh

Unified environment setup and dependency management.

| Command                | Arguments   | Description                    | Example                                             |
| ---------------------- | ----------- | ------------------------------ | --------------------------------------------------- |
| `setup_environment.sh` | `[OPTIONS]` | Set up development environment | `./scripts/setup_environment.sh --type development` |

**Options:**

| Flag                | Description                                        | Default       | Example             |
| ------------------- | -------------------------------------------------- | ------------- | ------------------- |
| `--type TYPE`       | Setup type: `development`, `testing`, `production` | `development` | `--type production` |
| `--verbose`         | Enable verbose output                              | Disabled      | `--verbose`         |
| `--skip-python`     | Skip Python environment setup                      | Disabled      | `--skip-python`     |
| `--skip-node`       | Skip Node.js environment setup                     | Disabled      | `--skip-node`       |
| `--skip-docker`     | Skip Docker setup                                  | Disabled      | `--skip-docker`     |
| `--force-reinstall` | Force reinstall of components                      | Disabled      | `--force-reinstall` |
| `--help`            | Show help message                                  | -             | `--help`            |

**Examples:**

```bash
# Development setup with verbose output
./scripts/setup_environment.sh --type development --verbose

# Production setup, skip Docker
./scripts/setup_environment.sh --type production --skip-docker

# Force reinstall everything
./scripts/setup_environment.sh --force-reinstall

# Show help
./scripts/setup_environment.sh --help
```

**Expected Output:**

```
===========================================================
 AlphaMind Environment Setup
===========================================================
[INFO] Detected OS: Linux (Ubuntu)
[INFO] Checking Python version...
[SUCCESS] Python 3.10.8 found
[INFO] Creating virtual environment...
[SUCCESS] Virtual environment created
[INFO] Installing Python dependencies...
[SUCCESS] All dependencies installed
[INFO] Setup complete! Activate with: source venv/bin/activate
```

---

### run_alphamind.sh

Start AlphaMind application (backend and frontend).

| Command            | Arguments | Description        | Example                      |
| ------------------ | --------- | ------------------ | ---------------------------- |
| `run_alphamind.sh` | None      | Start all services | `./scripts/run_alphamind.sh` |

**Example:**

```bash
./scripts/run_alphamind.sh
```

**Expected Output:**

```
Starting AlphaMind application...
Creating Python virtual environment...
Starting backend server...
Backend running with PID: 12345
Starting frontend...
Frontend running with PID: 12346
AlphaMind application is running!
Access the application at: http://localhost:3000
Press Ctrl+C to stop all services
```

---

### run_tests.sh

Comprehensive automated testing pipeline.

| Command        | Arguments   | Description    | Example                              |
| -------------- | ----------- | -------------- | ------------------------------------ |
| `run_tests.sh` | `[OPTIONS]` | Run test suite | `./scripts/run_tests.sh --type unit` |

**Options:**

| Flag               | Description                                    | Default          | Example                   |
| ------------------ | ---------------------------------------------- | ---------------- | ------------------------- |
| `--type TYPE`      | Test type: `unit`, `integration`, `e2e`, `all` | `all`            | `--type unit`             |
| `--component COMP` | Specific component to test                     | All              | `--component risk_system` |
| `--verbose`        | Enable verbose output                          | Disabled         | `--verbose`               |
| `--no-coverage`    | Disable coverage reporting                     | Enabled          | `--no-coverage`           |
| `--report-dir DIR` | Directory for test reports                     | `./test-reports` | `--report-dir ./reports`  |
| `--parallel`       | Run tests in parallel                          | Disabled         | `--parallel`              |
| `--fail-fast`      | Stop on first failure                          | Disabled         | `--fail-fast`             |
| `--help`           | Show help message                              | -                | `--help`                  |

**Examples:**

```bash
# Run all tests
./scripts/run_tests.sh

# Run unit tests only
./scripts/run_tests.sh --type unit

# Test specific component
./scripts/run_tests.sh --component ai_models --verbose

# Fast failure mode
./scripts/run_tests.sh --fail-fast --parallel

# Integration tests with custom report directory
./scripts/run_tests.sh --type integration --report-dir ./my-reports
```

---

### test_components.sh

Test specific AlphaMind components.

| Command              | Arguments          | Description               | Example                                                |
| -------------------- | ------------------ | ------------------------- | ------------------------------------------------------ |
| `test_components.sh` | `--component NAME` | Test a specific component | `./scripts/test_components.sh --component risk_engine` |

**Available Components:**

| Component          | Description                 | Example                        |
| ------------------ | --------------------------- | ------------------------------ |
| `ai_models`        | AI/ML models and training   | `--component ai_models`        |
| `risk_system`      | Risk management system      | `--component risk_system`      |
| `execution_engine` | Order execution engine      | `--component execution_engine` |
| `market_data`      | Market data connectors      | `--component market_data`      |
| `alternative_data` | Alternative data processing | `--component alternative_data` |
| `api`              | REST API endpoints          | `--component api`              |
| `all`              | All components              | `--component all`              |

**Example:**

```bash
./scripts/test_components.sh --component risk_system --verbose
```

---

### lint_code.sh

Code quality and linting automation.

| Command        | Arguments   | Description   | Example                        |
| -------------- | ----------- | ------------- | ------------------------------ |
| `lint_code.sh` | `[OPTIONS]` | Lint codebase | `./scripts/lint_code.sh --fix` |

**Options:**

| Flag               | Description                      | Default          | Example                  |
| ------------------ | -------------------------------- | ---------------- | ------------------------ |
| `--type TYPE`      | Lint type: `python`, `js`, `all` | `all`            | `--type python`          |
| `--component COMP` | Specific component to lint       | All              | `--component backend`    |
| `--fix`            | Auto-fix issues where possible   | Disabled         | `--fix`                  |
| `--verbose`        | Enable verbose output            | Disabled         | `--verbose`              |
| `--report-dir DIR` | Directory for lint reports       | `./lint-reports` | `--report-dir ./reports` |
| `--strict`         | Fail on any lint error           | Disabled         | `--strict`               |
| `--help`           | Show help message                | -                | `--help`                 |

**Examples:**

```bash
# Lint all code
./scripts/lint_code.sh

# Lint and auto-fix Python code
./scripts/lint_code.sh --type python --fix

# Strict mode for CI/CD
./scripts/lint_code.sh --strict

# Lint specific component
./scripts/lint_code.sh --component web-frontend --type js
```

---

### build.sh

Build AlphaMind components.

| Command    | Arguments   | Description   | Example                               |
| ---------- | ----------- | ------------- | ------------------------------------- |
| `build.sh` | `[OPTIONS]` | Build project | `./scripts/build.sh --env production` |

**Options:**

| Flag               | Description                                         | Default           | Example                    |
| ------------------ | --------------------------------------------------- | ----------------- | -------------------------- |
| `--env ENV`        | Environment: `development`, `staging`, `production` | `development`     | `--env production`         |
| `--component COMP` | Specific component to build                         | All               | `--component web-frontend` |
| `--clean`          | Clean build (remove artifacts)                      | Disabled          | `--clean`                  |
| `--verbose`        | Enable verbose output                               | Disabled          | `--verbose`                |
| `--report-dir DIR` | Directory for build reports                         | `./build-reports` | `--report-dir ./reports`   |
| `--no-optimize`    | Disable optimization                                | Enabled           | `--no-optimize`            |
| `--no-cache`       | Disable build caching                               | Enabled           | `--no-cache`               |
| `--analyze`        | Generate build analysis                             | Disabled          | `--analyze`                |

**Examples:**

```bash
# Production build
./scripts/build.sh --env production --clean

# Development build with analysis
./scripts/build.sh --env development --analyze

# Build specific component
./scripts/build.sh --component backend --verbose
```

---

### docker_build.sh

Build Docker containers.

| Command           | Arguments   | Description         | Example                     |
| ----------------- | ----------- | ------------------- | --------------------------- |
| `docker_build.sh` | `[OPTIONS]` | Build Docker images | `./scripts/docker_build.sh` |

**Example:**

```bash
# Build all Docker images
./scripts/docker_build.sh

# Build and push to registry
./scripts/docker_build.sh --push --registry gcr.io/my-project
```

---

### deploy_automation.sh

Automated deployment with rollback support.

| Command                | Arguments   | Description        | Example                                        |
| ---------------------- | ----------- | ------------------ | ---------------------------------------------- |
| `deploy_automation.sh` | `[OPTIONS]` | Deploy application | `./scripts/deploy_automation.sh --env staging` |

**Options:**

| Flag               | Description                                         | Default         | Example               |
| ------------------ | --------------------------------------------------- | --------------- | --------------------- |
| `--env ENV`        | Environment: `development`, `staging`, `production` | `development`   | `--env production`    |
| `--component COMP` | Specific component to deploy                        | All             | `--component backend` |
| `--verbose`        | Enable verbose output                               | Disabled        | `--verbose`           |
| `--log-dir DIR`    | Directory for deployment logs                       | `./deploy-logs` | `--log-dir ./logs`    |
| `--dry-run`        | Simulate deployment                                 | Disabled        | `--dry-run`           |
| `--rollback`       | Rollback to previous version                        | Disabled        | `--rollback`          |
| `--skip-build`     | Skip build step                                     | Disabled        | `--skip-build`        |
| `--skip-tests`     | Skip tests before deployment                        | Disabled        | `--skip-tests`        |
| `--force`          | Force deploy even if tests fail                     | Disabled        | `--force`             |

**Examples:**

```bash
# Deploy to staging
./scripts/deploy_automation.sh --env staging

# Production deployment with verbose logging
./scripts/deploy_automation.sh --env production --verbose

# Dry run (simulation)
./scripts/deploy_automation.sh --env production --dry-run

# Rollback to previous version
./scripts/deploy_automation.sh --env production --rollback

# Quick deploy (skip tests)
./scripts/deploy_automation.sh --env development --skip-tests
```

---

### db_migrate.sh

Database migration management.

| Command         | Arguments  | Description                | Example                           |
| --------------- | ---------- | -------------------------- | --------------------------------- |
| `db_migrate.sh` | `[ACTION]` | Manage database migrations | `./scripts/db_migrate.sh upgrade` |

**Actions:**

| Action      | Description                  | Example                                           |
| ----------- | ---------------------------- | ------------------------------------------------- |
| `upgrade`   | Apply all pending migrations | `./scripts/db_migrate.sh upgrade`                 |
| `downgrade` | Rollback last migration      | `./scripts/db_migrate.sh downgrade`               |
| `status`    | Show migration status        | `./scripts/db_migrate.sh status`                  |
| `create`    | Create new migration         | `./scripts/db_migrate.sh create "Add user table"` |

---

### release.sh

Create and publish releases.

| Command      | Arguments | Description    | Example                       |
| ------------ | --------- | -------------- | ----------------------------- |
| `release.sh` | `VERSION` | Create release | `./scripts/release.sh v1.2.0` |

**Example:**

```bash
# Create version 1.2.0 release
./scripts/release.sh v1.2.0

# Create and push to GitHub
./scripts/release.sh v1.2.0 --push
```

---

## Script Reference Table

Complete reference of all scripts:

| Script                 | Purpose               | Typical Usage             | Frequency            |
| ---------------------- | --------------------- | ------------------------- | -------------------- |
| `setup_environment.sh` | Environment setup     | Initial setup, onboarding | Once per environment |
| `run_alphamind.sh`     | Start application     | Development, testing      | Daily                |
| `run_tests.sh`         | Run test suite        | Development, CI/CD        | Multiple times daily |
| `test_components.sh`   | Component testing     | Feature development       | Multiple times daily |
| `lint_code.sh`         | Code quality checks   | Pre-commit, CI/CD         | Multiple times daily |
| `build.sh`             | Build application     | Deployment preparation    | Several times daily  |
| `docker_build.sh`      | Build containers      | Container deployment      | Before deployment    |
| `deploy_automation.sh` | Deploy application    | Release deployment        | Per release          |
| `db_migrate.sh`        | Database migrations   | Schema changes            | Per migration        |
| `release.sh`           | Create releases       | Version releases          | Per version          |
| `start_dev.sh`         | Start dev environment | Quick development start   | Daily                |

---

## Environment Variables

CLI scripts respect these environment variables:

| Variable              | Description              | Default       | Example              |
| --------------------- | ------------------------ | ------------- | -------------------- |
| `ALPHAMIND_ENV`       | Environment type         | `development` | `production`         |
| `ALPHAMIND_VERBOSE`   | Enable verbose output    | `false`       | `true`               |
| `ALPHAMIND_LOG_LEVEL` | Logging level            | `INFO`        | `DEBUG`              |
| `ALPHAMIND_ROOT`      | Project root directory   | Auto-detected | `/path/to/AlphaMind` |
| `ALPHAMIND_VENV`      | Virtual environment path | `venv`        | `/path/to/venv`      |

**Usage:**

```bash
# Set environment type
export ALPHAMIND_ENV=production

# Enable verbose mode
export ALPHAMIND_VERBOSE=true

# Run script with environment
./scripts/build.sh
```

---

## Exit Codes

All scripts use standard exit codes:

| Code  | Meaning           | Description                    |
| ----- | ----------------- | ------------------------------ |
| `0`   | Success           | Command completed successfully |
| `1`   | General Error     | Unspecified error occurred     |
| `2`   | Usage Error       | Invalid arguments or options   |
| `126` | Permission Error  | Script not executable          |
| `127` | Command Not Found | Required command missing       |
| `130` | User Interrupt    | User pressed Ctrl+C            |

**Example:**

```bash
./scripts/run_tests.sh
echo $?  # Check exit code
# 0 = tests passed
# 1 = tests failed
```

---

## Best Practices

1. **Always run from project root**: `cd /path/to/AlphaMind && ./scripts/script.sh`
2. **Check exit codes**: Use `$?` to verify script success
3. **Use verbose mode** during debugging: `--verbose`
4. **Read help first**: Use `--help` to understand options
5. **Use dry-run** before production: `--dry-run` for critical operations
6. **Review logs**: Check log files in report directories

---

## Next Steps

- **Configuration**: See [CONFIGURATION.md](CONFIGURATION.md) for environment setup
- **API Usage**: See [API.md](API.md) for programmatic access
- **Examples**: Try [EXAMPLES/](EXAMPLES/) for complete workflows
- **Troubleshooting**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for issues
