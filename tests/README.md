# AlphaMind Tests

[![Test Coverage](https://img.shields.io/badge/coverage-78%25-yellowgreen)](https://github.com/abrar2030/AlphaMind/tree/main/tests)
[![License](https://img.shields.io/github/license/abrar2030/AlphaMind)](https://github.com/abrar2030/AlphaMind/blob/main/LICENSE)

## Overview

The tests directory contains comprehensive test suites for the AlphaMind quantitative trading system. These tests ensure the reliability, correctness, and performance of the system's components, from AI models to order execution and risk management. With approximately 78% test coverage, the test suite provides confidence in the system's functionality and helps prevent regressions during development.

## Directory Contents

- `test_attention.py` - Tests for attention mechanisms in AI models
- `test_auth.py` - Authentication and authorization tests
- `test_generative_finance.py` - Tests for generative models in financial applications
- `test_order_manager.py` - Order management system tests
- `test_portfolio.py` - Portfolio management tests
- `test_portfolio_risk.py` - Portfolio risk assessment tests
- `test_position_limits.py` - Position sizing and limits tests
- `test_real_time_monitoring.py` - Real-time monitoring system tests
- `test_risk_execution_integration.py` - Integration tests for risk and execution systems
- `test_sentiment.py` - Sentiment analysis model tests

## Test Categories

### AI Model Tests

- **Attention Mechanisms**: Tests for transformer-based models and attention layers
- **Generative Models**: Tests for synthetic data generation and scenario modeling
- **Sentiment Analysis**: Tests for NLP-based sentiment extraction from financial texts

### Trading System Tests

- **Order Management**: Tests for order creation, routing, and execution
- **Portfolio Management**: Tests for portfolio construction and rebalancing
- **Position Limits**: Tests for position sizing and risk limits enforcement

### Risk Management Tests

- **Portfolio Risk**: Tests for risk metrics calculation and monitoring
- **Risk-Execution Integration**: Tests for integration between risk controls and execution

### Infrastructure Tests

- **Authentication**: Tests for user authentication and authorization
- **Real-time Monitoring**: Tests for monitoring and alerting systems

## Running Tests

### Prerequisites

- Python 3.10+
- Required dependencies installed (`pip install -r requirements.txt`)
- Test database configured (if applicable)

### Running All Tests

```bash
# From the project root
cd AlphaMind
pytest tests/

# With coverage report
pytest tests/ --cov=alphamind --cov-report=term-missing
```

### Running Specific Test Categories

```bash
# Run AI model tests
pytest tests/test_attention.py tests/test_generative_finance.py tests/test_sentiment.py

# Run trading system tests
pytest tests/test_order_manager.py tests/test_portfolio.py tests/test_position_limits.py

# Run risk management tests
pytest tests/test_portfolio_risk.py tests/test_risk_execution_integration.py
```

### Running Individual Tests

```bash
# Run a specific test file
pytest tests/test_portfolio.py

# Run a specific test class
pytest tests/test_portfolio.py::TestPortfolioOptimization

# Run a specific test method
pytest tests/test_portfolio.py::TestPortfolioOptimization::test_efficient_frontier
```

## Test Configuration

Tests can be configured using pytest configuration files or environment variables:

- `pytest.ini` - Global pytest configuration
- `conftest.py` - Test fixtures and setup
- Environment variables - Test-specific configuration

## Continuous Integration

Tests are automatically run in the CI/CD pipeline on:
- Pull requests to the main branch
- Direct pushes to the main branch
- Scheduled runs (nightly builds)

The CI pipeline reports test results and coverage metrics, which are displayed in the GitHub repository.

## Writing New Tests

When adding new features or fixing bugs, follow these guidelines for writing tests:

1. **Test Coverage**: Aim for comprehensive coverage of new code
2. **Test Organization**: Place tests in the appropriate file based on functionality
3. **Test Independence**: Ensure tests can run independently and in any order
4. **Test Fixtures**: Use fixtures for common setup and teardown
5. **Mocking**: Use mocks for external dependencies and services

Example test structure:

```python
def test_portfolio_optimization(setup_portfolio):
    # Arrange
    portfolio = setup_portfolio
    target_return = 0.1
    
    # Act
    optimized = portfolio.optimize(target_return=target_return)
    
    # Assert
    assert optimized.expected_return >= target_return
    assert optimized.sharpe_ratio > portfolio.sharpe_ratio
```

## Performance Testing

Performance tests ensure the system meets latency and throughput requirements:

- **Benchmark Tests**: Measure execution time for critical operations
- **Load Tests**: Evaluate system performance under high load
- **Scalability Tests**: Assess performance with increasing data or user load

## Test Data

Test data is provided through:
- Fixtures defined in `conftest.py`
- Static test data files in the `tests/data` directory
- Dynamically generated test data

## Contributing

When contributing to the test suite:

1. Add tests for new features or bug fixes
2. Maintain or improve test coverage
3. Ensure tests are properly documented
4. Follow the existing test style and organization

For more details, see the [Contributing Guidelines](../docs/CONTRIBUTING.md).

## Related Documentation

- [Development Guide](../docs/development-guide.md)
- [CI/CD Pipeline](../docs/deployment.md)
- [Troubleshooting Guide](../docs/troubleshooting.md)
