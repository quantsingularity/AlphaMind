import { configureStore } from "@reduxjs/toolkit";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react-native";
import { Provider as PaperProvider } from "react-native-paper";
import { Provider } from "react-redux";
import App from "../../App";
import authReducer from "../../store/slices/authSlice";
import portfolioReducer from "../../store/slices/portfolioSlice";
import settingsReducer from "../../store/slices/settingsSlice";

jest.mock("../../services/api", () => ({
  default: {
    post: jest.fn(),
    get: jest.fn(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  },
}));

jest.mock("expo-constants", () => ({
  default: {
    expoConfig: {
      extra: { apiBaseUrl: "http://localhost:5000" },
    },
  },
}));

describe("App integration", () => {
  it("renders auth screen when not authenticated", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Welcome to AlphaMind")).toBeTruthy();
    });
  });

  it("shows login and register navigation options", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Login")).toBeTruthy();
    });
  });
});

describe("Auth flow", () => {
  const createTestStore = (preloadedState) =>
    configureStore({
      reducer: {
        auth: authReducer,
        portfolio: portfolioReducer,
        settings: settingsReducer,
      },
      preloadedState,
    });

  it("shows dashboard when authenticated", async () => {
    const store = createTestStore({
      auth: {
        user: { id: 1, name: "Test User", email: "test@example.com" },
        isAuthenticated: true,
        loading: false,
        error: null,
      },
      portfolio: {
        data: null,
        performance: [],
        holdings: [],
        loading: false,
        performanceLoading: false,
        holdingsLoading: false,
        error: null,
        lastUpdated: null,
      },
      settings: {
        theme: "light",
        notifications: {
          tradeAlerts: true,
          researchUpdates: true,
          priceAlerts: true,
        },
        displayPreferences: {
          currency: "USD",
          decimalPlaces: 2,
          chartType: "line",
        },
      },
    });

    render(
      <Provider store={store}>
        <PaperProvider>
          <App />
        </PaperProvider>
      </Provider>,
    );

    await waitFor(() => {
      expect(screen.getByText("AlphaMind Dashboard")).toBeTruthy();
    });
  });
});
