import React from "react";
import { render, screen, waitFor } from "@testing-library/react-native";
import { Provider } from "react-redux";
import { Provider as PaperProvider } from "react-native-paper";
import { configureStore } from "@reduxjs/toolkit";
import HomeScreen from "../../screens/HomeScreen";
import authReducer from "../../store/slices/authSlice";
import portfolioReducer from "../../store/slices/portfolioSlice";
import settingsReducer from "../../store/slices/settingsSlice";

const mockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      auth: authReducer,
      portfolio: portfolioReducer,
      settings: settingsReducer,
    },
    preloadedState: initialState,
  });
};

const renderWithProviders = (component, store) => {
  return render(
    <Provider store={store}>
      <PaperProvider>{component}</PaperProvider>
    </Provider>,
  );
};

describe("HomeScreen", () => {
  it("renders loading state initially", () => {
    const store = mockStore({
      auth: { user: null, isAuthenticated: false, loading: false },
      portfolio: { data: null, loading: true, error: null },
      settings: { theme: "light" },
    });

    renderWithProviders(<HomeScreen />, store);
    expect(screen.getByText("Loading portfolio...")).toBeTruthy();
  });

  it("renders dashboard with portfolio data", async () => {
    const store = mockStore({
      auth: {
        user: { name: "John Doe" },
        isAuthenticated: true,
        loading: false,
      },
      portfolio: {
        data: {
          value: 1250345.67,
          dailyPnL: 15678.9,
          dailyPnLPercent: 1.27,
          sharpeRatio: 2.35,
          activeStrategies: 12,
        },
        loading: false,
        error: null,
      },
      settings: { theme: "light" },
    });

    renderWithProviders(<HomeScreen />, store);

    await waitFor(() => {
      expect(screen.getByText("AlphaMind Dashboard")).toBeTruthy();
      expect(screen.getByText("Welcome back, John Doe!")).toBeTruthy();
      expect(screen.getByText("Portfolio Value")).toBeTruthy();
    });
  });

  it("renders error state", () => {
    const store = mockStore({
      auth: { user: null, isAuthenticated: false, loading: false },
      portfolio: {
        data: null,
        loading: false,
        error: "Failed to fetch portfolio",
      },
      settings: { theme: "light" },
    });

    renderWithProviders(<HomeScreen />, store);
    expect(screen.getByText("Failed to fetch portfolio")).toBeTruthy();
  });
});
