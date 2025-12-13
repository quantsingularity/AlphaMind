# AlphaMind Mobile Frontend

[![License](https://img.shields.io/github/license/abrar2030/AlphaMind)](https://github.com/abrar2030/AlphaMind/blob/main/LICENSE)

## Overview

The mobile-frontend directory contains the React Native mobile application for AlphaMind, providing traders with on-the-go access to portfolio monitoring, alerts, and trading capabilities. This cross-platform mobile application delivers a responsive and intuitive interface for monitoring investments and executing trades from any mobile device.

## Features

The mobile application provides the following key features:

- **Authentication**: Complete login and registration system with JWT token management
- **Portfolio Monitoring**: Real-time tracking of portfolio performance with KPI cards
- **Dashboard**: Visual analytics and performance metrics with refresh capability
- **Research Access**: Browse and access research papers and publications
- **Settings Management**: Theme preferences, notification settings, and user profile
- **State Management**: Redux Toolkit for centralized state management
- **Offline Support**: Mock data mode for development without backend
- **API Integration**: Axios-based API client with interceptors for authentication

## Technology Stack

- **Framework**: React Native (Expo)
- **State Management**: Redux Toolkit
- **Navigation**: React Navigation (Bottom Tabs, Native Stack)
- **UI Components**: React Native Paper (Material Design 3)
- **Charts**: Victory Native, React Native Chart Kit
- **API Client**: Axios
- **Testing**: Jest, React Native Testing Library
- **Code Quality**: ESLint, Prettier

## Directory Structure

```
mobile-frontend/
├── __tests__/                 # Test files
│   ├── components/           # Component tests
│   ├── screens/              # Screen tests
│   ├── services/             # Service tests
│   ├── store/                # Redux tests
│   └── integration/          # Integration tests
├── assets/                    # Static assets (images, fonts)
├── components/                # Reusable UI components
│   ├── ErrorMessage.js
│   ├── KPICard.js
│   └── LoadingSpinner.js
├── constants/                 # Application constants
│   ├── config.js             # API and app configuration
│   └── theme.js              # Theme definitions
├── hooks/                     # Custom React hooks
├── navigation/                # Navigation configuration
│   ├── AppNavigator.js       # Main app navigator
│   └── AuthNavigator.js      # Authentication navigator
├── screens/                   # Screen components
│   ├── DocumentationScreen.js
│   ├── FeaturesScreen.js
│   ├── HomeScreen.js
│   ├── LoginScreen.js
│   ├── RegisterScreen.js
│   ├── ResearchScreen.js
│   └── SettingsScreen.js
├── services/                  # API and external services
│   ├── api.js                # Axios instance with interceptors
│   ├── authService.js        # Authentication API
│   ├── portfolioService.js   # Portfolio API
│   └── researchService.js    # Research API
├── store/                     # Redux store
│   ├── index.js              # Store configuration
│   └── slices/               # Redux slices
│       ├── authSlice.js
│       ├── portfolioSlice.js
│       └── settingsSlice.js
├── utils/                     # Utility functions
├── App.js                     # Root component
├── index.js                   # App entry point
├── app.json                   # Expo configuration
├── babel.config.js            # Babel configuration
├── jest.config.js             # Jest configuration
├── jest.setup.js              # Jest setup file
├── package.json               # Dependencies and scripts
├── .env.example               # Environment variables template
└── .env                       # Environment variables (create from .env.example)
```

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn
- Expo CLI
- iOS Simulator (macOS) or Android Emulator/Device

### Installation

1. Navigate to the mobile-frontend directory:

   ```bash
   cd mobile-frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Create environment file:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` to configure your settings:
   - Set `API_BASE_URL` to your backend URL (default: http://localhost:5000)
   - Set `ENABLE_MOCK_DATA=true` for offline development

### Running the Application

#### Development Mode

Start the Expo development server:

```bash
npm start
```

This will open the Expo DevTools in your browser. From there, you can:

- Press `i` to open iOS Simulator
- Press `a` to open Android Emulator
- Scan QR code with Expo Go app on your phone

#### Run on Specific Platform

```bash
# iOS
npm run ios

# Android
npm run android

# Web
npm run web
```

### Backend Integration

The mobile app integrates with the AlphaMind backend API.

#### Running with Backend

1. **Start the backend server** (from repository root):

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python src/main.py
   ```

   The backend should start on `http://localhost:5000`

2. **Configure mobile app** to connect to backend:
   - Edit `.env` and set `ENABLE_MOCK_DATA=false`
   - Set `API_BASE_URL=http://localhost:5000` (or your backend URL)
   - For physical devices, use your computer's IP address instead of localhost

3. **Start mobile app**:
   ```bash
   npm start
   ```

#### Running without Backend (Mock Data)

For development without backend:

- Set `ENABLE_MOCK_DATA=true` in `.env`
- The app will use mock data from services (portfolioService, researchService)

## Testing

### Run Tests

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch

# Run specific test
npm test -- --testPathPattern="authService"
```

### Test Coverage

Current test suites:

- ✅ Authentication service tests
- ✅ Redux store slice tests
- ✅ Configuration tests
- ✅ API integration tests

See `TESTING.md` for detailed testing documentation.

## Code Quality

### Linting

```bash
# Check for linting errors
npm run lint

# Auto-fix linting errors
npm run lint:fix
```

### Formatting

```bash
# Check code formatting
npm run format:check

# Auto-format code
npm run format
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# API Configuration
API_BASE_URL=http://localhost:5000
API_TIMEOUT=30000

# Feature Flags
ENABLE_MOCK_DATA=false        # Use mock data instead of API
ENABLE_OFFLINE_MODE=true      # Enable offline capabilities

# Environment
NODE_ENV=development
```

### Theme Customization

Themes are defined in `constants/theme.js`. You can customize:

- Light theme colors
- Dark theme colors
- Component-specific styling

Users can switch themes in Settings screen.

## Key Features Implementation

### Authentication

- JWT token-based authentication
- Login and registration screens
- Auto-login with stored credentials
- Token refresh handling
- Secure token storage with AsyncStorage

### State Management

- Redux Toolkit for centralized state
- Async thunks for API calls
- Slices for auth, portfolio, and settings
- Persistent state with AsyncStorage

### API Integration

- Axios client with interceptors
- Automatic token injection
- Error handling and retry logic
- Mock data support for offline development

### Navigation

- Tab-based navigation for main screens
- Stack navigation for authentication flow
- Protected routes requiring authentication
- Deep linking support

## Building for Production

### Android

```bash
# Generate APK
expo build:android

# Or using EAS Build (recommended)
eas build --platform android
```

### iOS

```bash
# Generate IPA
expo build:ios

# Or using EAS Build (recommended)
eas build --platform ios
```

## Troubleshooting

### Common Issues

1. **Metro bundler cache issues**:

   ```bash
   expo start -c
   ```

2. **Node modules issues**:

   ```bash
   rm -rf node_modules
   npm install
   ```

3. **iOS Simulator not opening**:

   ```bash
   # Ensure Xcode is installed
   xcode-select --install
   ```

4. **Android Emulator issues**:
   - Ensure Android Studio is installed
   - Create an AVD (Android Virtual Device)
   - Start emulator before running `npm run android`

5. **Backend connection issues**:
   - Check backend is running on correct port
   - For physical devices, use computer's IP instead of localhost
   - Check firewall settings

## Performance Optimization

- Redux state normalized and optimized
- Memoized selectors and components
- Lazy loading for heavy screens
- Image optimization with caching
- Pull-to-refresh for data updates

## Accessibility

- Semantic labeling for screen readers
- Sufficient color contrast
- Touch target sizes (minimum 44x44)
- Keyboard navigation support

## Security

- Secure token storage with AsyncStorage
- No sensitive data in Redux state
- HTTPS enforcement for API calls
- Token expiration handling
- Input validation and sanitization

## Contributing

1. Follow existing code structure and patterns
2. Write tests for new features
3. Run linting and formatting before committing
4. Update documentation for new features
5. Follow React Native and Redux best practices

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/abrar2030/AlphaMind/blob/main/LICENSE) file for details.

## Support

For issues and questions:

- GitHub Issues: https://github.com/abrar2030/AlphaMind/issues
- Documentation: See `/docs` in repository root

## Changelog

### v1.0.0 (Current)

- ✅ Complete authentication system (login/register)
- ✅ Redux state management implementation
- ✅ API integration with backend
- ✅ Portfolio dashboard with real-time KPIs
- ✅ Research papers browsing
- ✅ Settings with theme and notification preferences
- ✅ Mock data support for offline development
- ✅ Comprehensive test suite
- ✅ Production-ready build configuration
