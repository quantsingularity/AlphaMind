# AlphaMind Mobile Frontend

[![License](https://img.shields.io/github/license/abrar2030/AlphaMind)](https://github.com/abrar2030/AlphaMind/blob/main/LICENSE)

## Overview

The mobile-frontend directory contains the React Native mobile application for AlphaMind, providing traders with on-the-go access to portfolio monitoring, alerts, and trading capabilities. This cross-platform mobile application delivers a responsive and intuitive interface for monitoring investments and executing trades from any mobile device.

## Directory Contents

- `App.js` - Main application component and entry point
- `app.json` - Application configuration
- `assets/` - Static assets including images, fonts, and other resources
- `eslint.config.js` - ESLint configuration for code quality
- `index.js` - Application registration and initialization
- `navigation/` - Navigation configuration and components
- `package.json` - Dependencies and scripts configuration
- `screens/` - Screen components for different application views
- `tests/` - Test files for components and functionality
- `yarn.lock` - Yarn dependency lock file

## Features

The mobile application provides the following key features:

- **Portfolio Monitoring**: Real-time tracking of portfolio performance
- **Alert System**: Customizable alerts for price movements and events
- **Trade Execution**: Mobile trading capabilities (in progress)
- **Performance Reporting**: Visual analytics and performance metrics
- **User Authentication**: Secure login and account management

## Technology Stack

- **Framework**: React Native
- **State Management**: Redux
- **Navigation**: React Navigation
- **UI Components**: React Native Paper
- **Charts**: Victory Native
- **Testing**: Jest and React Native Testing Library

## Getting Started

### Prerequisites

- Node.js 16+
- Yarn or npm
- React Native CLI
- Android Studio (for Android development)
- Xcode (for iOS development)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abrar2030/AlphaMind.git
   cd AlphaMind/mobile-frontend
   ```

2. Install dependencies:
   ```bash
   yarn install
   # or
   npm install
   ```

3. Start the Metro bundler:
   ```bash
   npx react-native start
   ```

4. Run on Android:
   ```bash
   npx react-native run-android
   ```

5. Run on iOS:
   ```bash
   npx react-native run-ios
   ```

## Development Workflow

### Code Structure

- **Screens**: Individual application screens in the `screens/` directory
- **Components**: Reusable UI components
- **Navigation**: Screen navigation configuration in the `navigation/` directory
- **Services**: API and data services
- **Redux**: State management with actions, reducers, and selectors
- **Utilities**: Helper functions and utilities

### Testing

Run tests using Jest:

```bash
yarn test
# or
npm test
```

The `tests/` directory contains:
- Unit tests for components
- Integration tests for screens
- End-to-end tests using Detox

### Code Quality

Maintain code quality using ESLint:

```bash
yarn lint
# or
npm run lint
```

## Building for Production

### Android

```bash
cd android
./gradlew assembleRelease
```

The APK will be generated at `android/app/build/outputs/apk/release/app-release.apk`

### iOS

1. Open the project in Xcode:
   ```bash
   cd ios
   open AlphaMind.xcworkspace
   ```

2. Select "Product" > "Archive" from the menu
3. Follow the Xcode distribution workflow

## Troubleshooting

Common issues and solutions:

- **Metro bundler issues**: Clear cache with `npx react-native start --reset-cache`
- **Build failures**: Ensure all dependencies are installed and compatible
- **Device connection issues**: Check USB debugging settings and device drivers
- **iOS simulator problems**: Update Xcode and CocoaPods

## Contributing

When contributing to the mobile frontend:

1. Follow the React Native best practices
2. Maintain consistent code style
3. Write tests for new features
4. Update documentation for significant changes
5. Ensure cross-platform compatibility

For more details, see the [Contributing Guidelines](../docs/CONTRIBUTING.md).

## Related Documentation

- [Main README](../README.md) - Project overview
- [API Documentation](../docs/api-documentation.md) - Backend API reference
- [Development Guide](../docs/development-guide.md) - Development workflow
- [Troubleshooting Guide](../docs/troubleshooting.md) - Common issues and solutions
