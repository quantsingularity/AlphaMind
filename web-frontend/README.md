# AlphaMind Web Frontend

[![License](https://img.shields.io/github/license/abrar2030/AlphaMind)](https://github.com/abrar2030/AlphaMind/blob/main/LICENSE)

## Overview

The web-frontend directory contains the web interface for the AlphaMind quantitative trading system. This responsive web application provides traders and analysts with a comprehensive dashboard for strategy development, backtesting, performance analytics, and risk visualization. The web interface serves as the primary interaction point for users to leverage AlphaMind's advanced AI-driven trading capabilities.

## Directory Contents

- `about.html` - About page with project information
- `css/` - Stylesheets and design assets
- `docs/` - Frontend-specific documentation
- `documentation.html` - User documentation page
- `eslint.config.js` - ESLint configuration for code quality
- `features.html` - Feature showcase page
- `images/` - Image assets for the web interface
- `index.html` - Main landing page and entry point
- `js/` - JavaScript modules and application logic
- `package.json` - Dependencies and scripts configuration
- `package-lock.json` - Dependency lock file
- `research.html` - Research and methodology page
- `tests/` - Frontend test files

## Features

The web interface provides the following key features:

- **Interactive Dashboard**: Real-time portfolio monitoring and analytics
- **Strategy Builder**: Visual interface for creating and modifying trading strategies
- **Backtesting Interface**: Historical performance testing of strategies
- **Performance Analytics**: Detailed performance metrics and visualizations
- **Risk Visualization**: Interactive risk assessment tools (in progress)

## Technology Stack

- **Framework**: React with TypeScript
- **State Management**: Redux
- **Data Visualization**: D3.js, TradingView
- **UI Components**: Tailwind CSS, Styled Components
- **API Communication**: Axios, GraphQL
- **Authentication**: OAuth2, JWT
- **Testing**: Jest, React Testing Library, Cypress

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/abrar2030/AlphaMind.git
   cd AlphaMind/web-frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:

   ```bash
   npm start
   # or
   yarn start
   ```

4. Access the application at http://localhost:3000

## Development Workflow

### Code Structure

- **Components**: Reusable UI components
- **Pages**: Top-level page components
- **Services**: API and data services
- **Store**: Redux state management
- **Utils**: Helper functions and utilities
- **Hooks**: Custom React hooks
- **Assets**: Static assets and resources

### Building for Production

```bash
npm run build
# or
yarn build
```

The build artifacts will be stored in the `build/` directory.

### Testing

```bash
# Run all tests
npm test
# or
yarn test

# Run with coverage report
npm test -- --coverage
# or
yarn test --coverage

# Run end-to-end tests
npm run e2e
# or
yarn e2e
```

## Documentation

The `docs/` directory contains frontend-specific documentation:

- **API Reference**: API endpoints and usage
- **Component Documentation**: UI component specifications
- **User Guides**: End-user documentation
- **Tutorials**: Step-by-step guides for common tasks
- **Example Notebooks**: Interactive examples

## Responsive Design

The web interface is designed to be fully responsive, providing optimal user experience across:

- Desktop computers
- Tablets
- Mobile devices

## Accessibility

The application follows WCAG 2.1 guidelines to ensure accessibility:

- Semantic HTML
- ARIA attributes
- Keyboard navigation
- Screen reader compatibility
- Sufficient color contrast

## Browser Compatibility

The web interface supports:

- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

## Performance Optimization

The application implements several performance optimizations:

- Code splitting
- Lazy loading
- Memoization
- Asset optimization
- Caching strategies
