import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react-native";
import { Provider } from "react-redux";
import { Provider as PaperProvider } from "react-native-paper";
import { NavigationContainer } from "@react-navigation/native";
import App from "../../App";
import store from "../../store";
import * as authService from "../../services/authService";

jest.mock("../../services/authService");

describe("App Integration Flow", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("shows login screen when not authenticated", async () => {
    authService.authService.isAuthenticated = jest.fn().mockResolvedValue(false);

    render(
      <Provider store={store}>
        <PaperProvider>
          <NavigationContainer>
            <App />
          </NavigationContainer>
        </PaperProvider>
      </Provider>,
    );

    await waitFor(() => {
      expect(screen.queryByText("Welcome to AlphaMind")).toBeTruthy();
    });
  });

  it("completes login flow successfully", async () => {
    authService.authService.isAuthenticated = jest.fn().mockResolvedValue(false);
    authService.authService.login = jest.fn().mockResolvedValue({
      token: "test-token",
      user: { id: 1, email: "test@example.com", name: "Test User" },
    });

    render(
      <Provider store={store}>
        <PaperProvider>
          <NavigationContainer>
            <App />
          </NavigationContainer>
        </PaperProvider>
      </Provider>,
    );

    await waitFor(() => {
      expect(screen.queryByText("Welcome to AlphaMind")).toBeTruthy();
    });

    // Fill in login form
    const emailInput = screen.getByLabelText("Email");
    const passwordInput = screen.getByLabelText("Password");

    fireEvent.changeText(emailInput, "test@example.com");
    fireEvent.changeText(passwordInput, "password123");

    // Submit login
    const loginButton = screen.getByText("Login");
    fireEvent.press(loginButton);

    // Should navigate to main app after successful login
    await waitFor(
      () => {
        expect(authService.authService.login).toHaveBeenCalledWith(
          "test@example.com",
          "password123",
        );
      },
      { timeout: 3000 },
    );
  });
});
