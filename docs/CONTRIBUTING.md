# Contributing to AlphaMind

Thank you for your interest in contributing to AlphaMind! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment details (OS, browser, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancements! Please create an issue with:
- A clear, descriptive title
- Detailed description of the proposed enhancement
- Any relevant examples, mockups, or references
- Explanation of why this enhancement would be valuable

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   
   cd ../web-frontend
   npm install
   
   cd ../mobile-frontend
   yarn install
   ```
3. Run tests to ensure everything is working:
   ```
   cd tests
   pytest
   
   cd ../web-frontend
   npm test
   
   cd ../mobile-frontend
   yarn test
   ```

## Coding Standards

- Follow PEP 8 for Python code
- Use ESLint for JavaScript code
- Write meaningful commit messages
- Include tests for new features
- Update documentation for API changes

## Testing

- All new features should include tests
- Run the existing test suite before submitting a PR
- Aim to maintain or improve the current test coverage

## Documentation

- Update the README.md if necessary
- Document new features, APIs, or changes to existing functionality
- Keep code comments clear and up-to-date

## Review Process

- All PRs require at least one review before merging
- Address review comments promptly
- CI checks must pass before merging

Thank you for contributing to AlphaMind!
