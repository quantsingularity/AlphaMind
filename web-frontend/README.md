# AlphaMind Web Frontend

## Overview

The web-frontend is a modern React/TypeScript application providing a comprehensive interface for the AlphaMind quantitative trading system. This responsive web application offers traders and analysts real-time portfolio monitoring, strategy management, backtesting capabilities, and risk visualization.

## Technology Stack

- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite
- **Routing**: React Router v7
- **State Management**: TanStack Query (React Query)
- **Styling**: Tailwind CSS
- **Charts**: Recharts, D3.js
- **HTTP Client**: Axios
- **Testing**: Vitest, React Testing Library
- **Code Quality**: ESLint, Prettier

## Features

- **Real-time Dashboard**: Live portfolio monitoring with performance metrics
- **Strategy Management**: Create, configure, and monitor trading strategies
- **Portfolio Tracking**: View positions, P&L, and asset allocation
- **Backtesting Interface**: Test strategies against historical data
- **Risk Analytics**: Comprehensive risk metrics and visualizations
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Type Safety**: Full TypeScript support throughout the application

## Getting Started

### Prerequisites

- Node.js 16+ (recommended: Node.js 18 or higher)
- npm or yarn package manager

### Installation

1. Navigate to the web-frontend directory:

   ```bash
   cd web-frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   # or
   yarn install
   ```

3. Copy environment configuration:

   ```bash
   cp .env.example .env
   ```

4. Configure environment variables in `.env`:

   ```env
   VITE_API_BASE_URL=http://localhost:5000
   VITE_ENV=development
   VITE_ENABLE_MOCK_DATA=true
   ```

### Development

Start the development server:

```bash
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:5173`

### Building for Production

Build the application:

```bash
npm run build
# or
yarn build
```

Preview the production build:

```bash
npm run start
# or
yarn start
```

## Testing

### Run all tests:

```bash
npm test
# or
yarn test
```

### Run tests with UI:

```bash
npm run test:ui
# or
yarn test:ui
```

### Generate coverage report:

```bash
npm run test:coverage
# or
yarn test:coverage
```

## Project Structure

```
web-frontend/
├── src/
│   ├── assets/           # Static assets (images, styles)
│   ├── components/       # Reusable UI components
│   │   ├── Layout.tsx
│   │   └── Layout.test.tsx
│   ├── hooks/            # Custom React hooks
│   │   ├── useStrategies.ts
│   │   └── usePortfolio.ts
│   ├── pages/            # Page components
│   │   ├── Home.tsx
│   │   ├── Dashboard.tsx
│   │   ├── Strategies.tsx
│   │   ├── Portfolio.tsx
│   │   ├── Backtest.tsx
│   │   ├── Documentation.tsx
│   │   └── About.tsx
│   ├── services/         # API services
│   │   └── api.ts
│   ├── types/            # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/            # Utility functions
│   │   ├── format.ts
│   │   └── format.test.ts
│   ├── test/             # Test setup and utilities
│   │   └── setup.ts
│   ├── App.tsx           # Main application component
│   ├── main.tsx          # Application entry point
│   └── index.css         # Global styles
├── docs/                 # Documentation
├── public/               # Public static files
├── .env.example          # Environment variables template
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite configuration
├── vitest.config.ts      # Vitest configuration
├── tailwind.config.js    # Tailwind CSS configuration
└── README.md             # This file
```

## API Integration

The frontend communicates with the backend via REST API. The API service is configured in `src/services/api.ts`.

### API Endpoints

- `GET /health` - Health check
- `GET /api/strategies` - List all strategies
- `GET /api/portfolio` - Get portfolio data
- `GET /api/positions` - Get current positions
- `POST /api/orders` - Place new orders
- `POST /api/backtest` - Run backtest
- `GET /api/risk/metrics` - Get risk metrics

### Backend Setup

Before starting the frontend, ensure the backend is running:

```bash
# In the backend directory
cd ../backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

The backend should start on `http://localhost:5000`

## Development Guidelines

### Code Style

- Use TypeScript for all new code
- Follow React best practices and hooks patterns
- Use functional components with hooks
- Implement proper error handling
- Write tests for new features

### Component Guidelines

- Keep components small and focused
- Use composition over inheritance
- Extract reusable logic into custom hooks
- Implement proper prop types
- Add accessibility attributes

### Testing Guidelines

- Write unit tests for utilities and hooks
- Write component tests for UI components
- Test user interactions and edge cases
- Aim for > 80% code coverage
- Use meaningful test descriptions

## Linting and Formatting

Run linter:

```bash
npm run lint
```

Fix linting issues:

```bash
npm run lint:fix
```

## Browser Support

- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

## Performance Optimization

The application implements several performance optimizations:

- Code splitting with React.lazy and Suspense
- Memoization with React.memo and useMemo
- Efficient data fetching with TanStack Query
- Optimized re-renders with React.useCallback
- Asset optimization with Vite

## Troubleshooting

### Common Issues

**Port already in use:**

```bash
# Change port in vite.config.ts or use --port flag
npm run dev -- --port 3001
```

**Module not found errors:**

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Type errors:**

```bash
# Restart TypeScript server in your IDE
# Or run type check manually
npx tsc --noEmit
```

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Write/update tests
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/quantsingularity/AlphaMind/blob/main/LICENSE) file for details.
