# AlphaMind Automation Scripts

This package contains a set of comprehensive automation scripts designed to improve the development workflow, testing, deployment, and maintenance processes for the AlphaMind repository.

## Scripts Overview

1. **setup_environment.sh** - Unified environment setup and dependency management
2. **run_tests.sh** - Comprehensive automated testing pipeline
3. **lint_code.sh** - Code quality and linting automation
4. **optimized_build.sh** - Build process optimization
5. **deploy_automation.sh** - Deployment automation with rollback support

## Installation

1. Extract the zip file to your AlphaMind repository root directory
2. Make sure all scripts are executable:
   ```
   chmod +x scripts/*.sh
   ```

## Usage Instructions

### Environment Setup

```bash
./scripts/setup_environment.sh [OPTIONS]
```

Options:

- `--type TYPE` - Setup type: development, testing, or production (default: development)
- `--verbose` - Enable verbose output
- `--skip-python` - Skip Python environment setup
- `--skip-node` - Skip Node.js environment setup
- `--skip-docker` - Skip Docker setup
- `--force-reinstall` - Force reinstallation of components
- `--help` - Show help message

### Running Tests

```bash
./scripts/run_tests.sh [OPTIONS]
```

Options:

- `--type TYPE` - Test type: unit, integration, e2e, or all (default: all)
- `--component COMPONENT` - Specific component to test (e.g., ai_models, risk_system)
- `--verbose` - Enable verbose output
- `--no-coverage` - Disable coverage reporting
- `--report-dir DIR` - Directory for test reports (default: ./test-reports)
- `--parallel` - Run tests in parallel
- `--fail-fast` - Stop on first failure
- `--help` - Show help message

### Linting Code

```bash
./scripts/lint_code.sh [OPTIONS]
```

Options:

- `--type TYPE` - Lint type: python, js, all (default: all)
- `--component COMPONENT` - Specific component to lint (e.g., backend, web-frontend)
- `--fix` - Automatically fix issues where possible
- `--verbose` - Enable verbose output
- `--report-dir DIR` - Directory for lint reports (default: ./lint-reports)
- `--strict` - Fail on any lint error (strict mode)
- `--help` - Show help message

### Building the Project

```bash
./scripts/build.sh [OPTIONS]
```

Options:

- `--env ENV` - Build environment: development, staging, production (default: development)
- `--component COMPONENT` - Specific component to build (e.g., backend, web-frontend)
- `--clean` - Perform a clean build (remove previous build artifacts)
- `--verbose` - Enable verbose output
- `--report-dir DIR` - Directory for build reports (default: ./build-reports)
- `--no-optimize` - Disable optimization steps
- `--no-cache` - Disable build caching
- `--analyze` - Generate build analysis reports
- `--help` - Show help message

### Deploying the Project

```bash
./scripts/deploy_automation.sh [OPTIONS]
```

Options:

- `--env ENV` - Deployment environment: development, staging, production (default: development)
- `--component COMPONENT` - Specific component to deploy (e.g., backend, web-frontend)
- `--verbose` - Enable verbose output
- `--log-dir DIR` - Directory for deployment logs (default: ./deploy-logs)
- `--dry-run` - Simulate deployment without making changes
- `--rollback` - Rollback to previous deployment
- `--skip-build` - Skip build step
- `--skip-tests` - Skip tests before deployment
- `--force` - Force deployment even if tests fail
- `--help` - Show help message

## Workflow Integration

These scripts can be integrated into your development workflow as follows:

1. **Initial Setup**: When starting work on the project, run `setup_environment.sh` to set up your development environment.
2. **Development Cycle**: During development, use `lint_code.sh` to ensure code quality and `run_tests.sh` to verify functionality.
3. **Building**: When ready to build, use `optimized_build.sh` to create optimized builds for your target environment.
4. **Deployment**: Use `deploy_automation.sh` to deploy your application to the desired environment.

## CI/CD Integration

These scripts are designed to be easily integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up environment
        run: ./scripts/setup_environment.sh --type testing
      - name: Lint code
        run: ./scripts/lint_code.sh --strict
      - name: Run tests
        run: ./scripts/run_tests.sh --type all

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up environment
        run: ./scripts/setup_environment.sh --type production
      - name: Build project
        run: ./scripts/optimized_build.sh --env production

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: ./scripts/deploy_automation.sh --env production --skip-build --skip-tests
```

## Customization

Each script can be customized by editing the configuration files in the `config` directory. For deployment-specific configurations, see the `config/deploy` directory.

## Requirements

- Bash shell (Linux, macOS, or Windows with WSL)
- Python 3.10+ (for Python-related scripts)
- Node.js 16+ (for JavaScript-related scripts)
- Docker (optional, for containerized builds and deployments)

## Troubleshooting

If you encounter any issues with the scripts:

1. Run the script with the `--verbose` flag for more detailed output
2. Check the log files in the respective report directories
3. Ensure all dependencies are installed correctly
4. Verify that the script has execute permissions

## License

These scripts are provided under the same license as the AlphaMind repository.
