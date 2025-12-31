# Contributing to AlphaMind

Thank you for your interest in contributing to AlphaMind! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.10+ installed
- Node.js 16+ installed
- Git configured with your GitHub account
- Familiarity with the AlphaMind architecture ([ARCHITECTURE.md](ARCHITECTURE.md))

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/AlphaMind.git
cd AlphaMind

# Add upstream remote
git remote add upstream https://github.com/abrar2030/AlphaMind.git

# Create a feature branch
git checkout -b feature/your-feature-name
```

### Development Setup

```bash
# Run the setup script
./scripts/setup_environment.sh --type development

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -e "backend[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
# Feature branch
git checkout -b feature/add-new-strategy

# Bug fix branch
git checkout -b fix/order-validation-bug

# Documentation branch
git checkout -b docs/update-api-reference
```

### 2. Make Changes

- Write clean, readable code
- Follow code standards (see below)
- Add tests for new functionality
- Update documentation

### 3. Run Tests

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific component tests
./scripts/test_components.sh --component ai_models

# Run with coverage
pytest --cov=backend --cov-report=html
```

### 4. Lint Code

```bash
# Lint all code
./scripts/lint_code.sh

# Auto-fix issues
./scripts/lint_code.sh --fix

# Lint specific component
./scripts/lint_code.sh --component backend --type python
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add momentum trading strategy"

# Pre-commit hooks will run automatically
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
feat(ai_models): add SAC reinforcement learning agent

Implements Soft Actor-Critic (SAC) algorithm for trading.
Includes entropy regularization for exploration.

Closes #123

fix(execution): handle order rejection correctly

Previously, rejected orders caused system crash.
Now properly handles rejection and logs error.

Fixes #456
```

## Code Standards

### Python Code Style

Follow [PEP 8](https://pep8.org/) style guide:

| Guideline       | Rule                               | Example                                                      |
| --------------- | ---------------------------------- | ------------------------------------------------------------ |
| **Line Length** | Max 88 characters (Black default)  | -                                                            |
| **Indentation** | 4 spaces                           | -                                                            |
| **Imports**     | Group: stdlib, third-party, local  | `import os`<br>`import numpy`<br>`from backend import utils` |
| **Naming**      | snake_case for functions/variables | `calculate_sharpe_ratio()`                                   |
| **Naming**      | PascalCase for classes             | `class OrderManager:`                                        |
| **Naming**      | UPPER_CASE for constants           | `MAX_POSITION_SIZE = 0.1`                                    |
| **Type Hints**  | Use type hints                     | `def process(data: pd.DataFrame) -> dict:`                   |
| **Docstrings**  | Google style docstrings            | See example below                                            |

**Docstring Example:**

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio for a returns series.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0.02)

    Returns:
        Sharpe ratio (annualized)

    Raises:
        ValueError: If returns series is empty

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
        Sharpe Ratio: 1.45
    """
    if len(returns) == 0:
        raise ValueError("Returns series cannot be empty")

    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
```

### JavaScript/TypeScript Style

Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript):

- Use ESLint and Prettier for formatting
- Prefer `const` over `let`, never use `var`
- Use arrow functions for callbacks
- Use async/await over promises
- Add TypeScript types for all functions

### Code Organization

```python
# File structure
# 1. Module docstring
# 2. Imports (grouped)
# 3. Constants
# 4. Classes
# 5. Functions
# 6. Main execution (if __name__ == "__main__")

"""
Module for portfolio optimization.

This module implements modern portfolio theory algorithms
including mean-variance optimization and risk parity.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Local imports
from backend.risk_system.portfolio_risk import PortfolioRiskManager
from backend.core.config import config_manager

# Constants
MAX_WEIGHT = 0.3
MIN_WEIGHT = 0.01

# Logger
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Optimize portfolio weights."""

    def __init__(self):
        self.risk_manager = PortfolioRiskManager()

    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Optimize portfolio weights."""
        # Implementation
        pass


def calculate_portfolio_variance(weights, cov_matrix):
    """Calculate portfolio variance."""
    return weights @ cov_matrix @ weights


if __name__ == "__main__":
    # Example usage
    optimizer = PortfolioOptimizer()
```

## Testing Guidelines

### Test Coverage Requirements

- **Minimum coverage**: 70% overall
- **Critical components**: 90%+ coverage (execution engine, risk system)
- All new features must include tests

### Writing Tests

```python
import pytest
from backend.risk_system.bayesian_var import BayesianVaR


class TestBayesianVaR:
    """Test suite for Bayesian VaR calculator."""

    @pytest.fixture
    def sample_returns(self):
        """Fixture providing sample return data."""
        return np.random.normal(0.001, 0.02, 252)

    def test_initialization(self, sample_returns):
        """Test BayesianVaR initialization."""
        var_calc = BayesianVaR(returns=sample_returns)
        assert var_calc.returns is not None
        assert len(var_calc.returns) == 252

    def test_build_model_creates_trace(self, sample_returns):
        """Test that build_model generates trace."""
        var_calc = BayesianVaR(returns=sample_returns)
        var_calc.build_model()
        assert var_calc.trace is not None

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_calculate_var_different_alphas(self, sample_returns, alpha):
        """Test VaR calculation with different confidence levels."""
        var_calc = BayesianVaR(returns=sample_returns)
        var_calc.build_model()
        var_value = var_calc.calculate_var(alpha=alpha)

        assert isinstance(var_value, float)
        assert var_value < 0  # VaR should be negative

    def test_calculate_var_without_model_raises_error(self, sample_returns):
        """Test that VaR calculation fails without building model."""
        var_calc = BayesianVaR(returns=sample_returns)

        with pytest.raises(ValueError, match="Call build_model"):
            var_calc.calculate_var()
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_bayesian_var.py

# Specific test
pytest tests/test_bayesian_var.py::TestBayesianVaR::test_initialization

# With coverage
pytest --cov=backend --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Documentation

### Documentation Requirements

Every contribution must include:

1. **Code documentation** (docstrings)
2. **API documentation** (if adding/changing endpoints)
3. **User documentation** (if adding features)
4. **Examples** (for major features)

### Documentation Style

- Use **Markdown** for all documentation
- Include **code examples** with expected output
- Add **tables** for structured information
- Use **diagrams** for complex concepts (Mermaid preferred)
- Provide **working examples** that can be copy-pasted

### Updating Documentation

When making changes:

```bash
# Update relevant docs
nano docs/API.md           # If changing API
nano docs/USAGE.md         # If adding features
nano docs/CONFIGURATION.md # If adding config options

# Add examples
nano docs/EXAMPLES/your_feature.md

# Update feature matrix
nano docs/FEATURE_MATRIX.md

# Update architecture if needed
nano docs/ARCHITECTURE.md
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is linted and formatted
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] Branch is up-to-date with main

```bash
# Update your branch
git fetch upstream
git rebase upstream/main

# Push to your fork
git push origin feature/your-feature-name
```

### Submitting a Pull Request

1. **Go to GitHub** and create a pull request from your fork
2. **Fill out the PR template** completely
3. **Link related issues** (e.g., "Closes #123")
4. **Request reviewers** if you know who should review
5. **Be responsive** to review comments

### PR Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Changes Made

- Added feature X
- Fixed bug Y
- Updated documentation Z

## Testing

Describe testing performed:

- Unit tests added/updated
- Integration tests added/updated
- Manual testing performed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Commented code in hard-to-understand areas
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated and passing
- [ ] Dependent changes merged and published

## Related Issues

Closes #123
```

### Review Process

1. **Automated checks** run (tests, linting)
2. **Code review** by maintainers
3. **Address feedback** in new commits
4. **Approval** by at least one maintainer
5. **Merge** by maintainer

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints
- Maintain professionalism

### Getting Help

- **Documentation**: Start with [docs/README.md](README.md)
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: Check [EXAMPLES/](EXAMPLES/) for code samples

### Reporting Bugs

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:

1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**

- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.8]
- AlphaMind version: [e.g., 1.0.0]

**Logs**
```

Paste relevant logs

```

**Additional Context**
Any other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Description of the problem.

**Proposed Solution**
How you would like to see it solved.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other relevant information.
```

## License

By contributing to AlphaMind, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to AlphaMind! 🚀
