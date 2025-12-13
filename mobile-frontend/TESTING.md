# Testing Guide

## Running Tests

### Run All Tests

```bash
npm test
```

### Run Tests with Coverage

```bash
npm run test:coverage
```

### Run Tests in Watch Mode

```bash
npm run test:watch
```

### Run Specific Test Suite

```bash
npm test -- --testPathPattern="authService"
```

## Test Structure

- `__tests__/components/` - Component unit tests
- `__tests__/screens/` - Screen component tests
- `__tests__/services/` - API service tests
- `__tests__/store/` - Redux store and slice tests
- `__tests__/utils/` - Utility function tests
- `__tests__/integration/` - Integration tests

## Test Coverage

Current tests cover:

- Authentication service and flows
- Redux store slices (auth, portfolio, settings)
- Configuration constants
- API integration

## Known Testing Limitations

Due to React Native's complex testing environment:

1. Some UI component tests require additional setup for React Native Paper components
2. Flow type checking in React Native core may cause test failures
3. Integration tests work best with mocked API responses

## Adding New Tests

When adding new features, ensure:

1. Unit tests for services and utilities
2. Redux slice tests for new state management
3. Integration tests for critical user flows
4. At least 70% code coverage for new code
