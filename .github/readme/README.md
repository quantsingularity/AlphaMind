# GitHub Workflows Documentation

This document provides a comprehensive overview of the GitHub Actions workflows configured for the AlphaMind project. These workflows automate the continuous integration and deployment processes, ensuring code quality and reliability across all components of the application.

## Overview

The AlphaMind project utilizes GitHub Actions to automate testing, linting, and build processes. The workflow configuration is designed to support a multi-component architecture consisting of a Python backend, a web frontend, and a mobile frontend. This integrated approach ensures that all parts of the application maintain consistent quality standards.

## Workflow Files

Currently, the repository contains one workflow file:

- **ci-cd.yml**: Handles continuous integration for the AlphaMind project across all components.

## CI/CD Workflow (ci-cd.yml)

### Trigger Events

The CI/CD workflow is triggered on the following events:

- **Push events** to the `main`, `master`, and `develop` branches
- **Pull request events** targeting the `main`, `master`, and `develop` branches

This configuration ensures that code changes are validated both during development (via pull requests) and when merging into primary branches.

### Execution Environment

All jobs in this workflow run on the latest Ubuntu environment (`ubuntu-latest`), which provides a consistent and up-to-date execution environment.

### Job Structure

The workflow contains a single job named `lint-build-test` that performs linting, dependency installation, and testing for all components of the application. The job is structured to process each component sequentially.

### Backend Processing Steps

1. **Checkout Code**: Uses `actions/checkout@v4` to fetch the repository content.
2. **Set up Python**: Configures Python 3.11 using `actions/setup-python@v5` with pip caching enabled to speed up dependency installation.
3. **Install Backend Dependencies**: Upgrades pip and installs dependencies from `backend/requirements.txt`.
4. **Lint Backend (Flake8)**: Runs Flake8 on the backend code to ensure adherence to Python coding standards.
5. **Lint Backend (Black Check)**: Verifies that the backend code conforms to Black formatting standards without making changes.
6. **Test Backend**: Executes pytest from the `tests` directory to run all backend tests.

### Web Frontend Processing Steps

1. **Set up Node.js**: Configures Node.js version 20 using `actions/setup-node@v4` with npm caching enabled.
2. **Install Web Frontend Dependencies**: Uses `npm ci` to install dependencies based on the `package-lock.json` file, ensuring consistent installations.
3. **Test Web Frontend**: Runs the test suite for the web frontend using `npm test`.

Note: There is a commented-out linting step for the web frontend. This step can be uncommented and configured once linting rules are established for the web component.

### Mobile Frontend Processing Steps

1. **Set up Node.js**: Configures Node.js version 20 with yarn caching enabled.
2. **Install Mobile Frontend Dependencies**: Uses `yarn install --frozen-lockfile` to ensure consistent dependency installation.
3. **Test Mobile Frontend**: Executes the test suite for the mobile frontend using `yarn test`.

Similar to the web frontend, there is a commented-out linting step that can be enabled once linting configurations are established for the mobile component.

## Workflow Optimization Features

The workflow incorporates several optimization features to improve performance and reliability:

1. **Dependency Caching**: All three components (backend, web frontend, and mobile frontend) utilize dependency caching to speed up the workflow execution. The backend uses pip caching, the web frontend uses npm caching, and the mobile frontend uses yarn caching.

2. **Specific Working Directories**: Commands for each component are executed in their respective directories using the `working-directory` parameter, ensuring proper context for each operation.

3. **Consistent Environment**: By using specific versions for Python (3.11) and Node.js (20), the workflow ensures consistent behavior across different workflow runs.

## Extending the Workflow

The workflow is designed to be extensible. Here are some potential enhancements:

1. **Enable Linting for Frontend Components**: Uncomment and configure the linting steps for both web and mobile frontends once linting rules are established.

2. **Add Deployment Steps**: Extend the workflow to include deployment to staging or production environments based on branch or tag triggers.

3. **Add Code Coverage Reporting**: Incorporate code coverage tools and reporting to monitor test coverage.

4. **Implement Security Scanning**: Add security scanning tools to identify potential vulnerabilities in the codebase or dependencies.

5. **Add Performance Testing**: Incorporate performance testing to ensure the application meets performance requirements.

## Best Practices for Contributors

When working with this workflow, consider the following best practices:

1. **Run Tests Locally**: Before pushing changes, run tests locally to catch issues early.

2. **Review Workflow Logs**: When a workflow fails, review the logs to understand the cause of failure.

3. **Keep Dependencies Updated**: Regularly update dependencies to benefit from bug fixes and security patches.

4. **Maintain Consistency**: Ensure that local development environments match the workflow environment to avoid "works on my machine" issues.

5. **Document Changes**: When modifying the workflow, document the changes and their purpose to help other contributors understand the workflow configuration.

## Troubleshooting

If you encounter issues with the workflow, consider the following troubleshooting steps:

1. **Check Dependency Versions**: Ensure that dependency versions in your local environment match those in the workflow.

2. **Verify Environment Variables**: If the workflow uses environment variables, verify that they are correctly configured.

3. **Review Recent Changes**: Check recent changes to the codebase that might have affected the workflow.

4. **Inspect Failed Steps**: In the GitHub Actions UI, inspect the logs of failed steps to identify specific errors.

This documentation aims to provide a comprehensive understanding of the GitHub workflow configuration for the AlphaMind project. By following these guidelines, contributors can effectively work with and extend the automated CI/CD processes.
