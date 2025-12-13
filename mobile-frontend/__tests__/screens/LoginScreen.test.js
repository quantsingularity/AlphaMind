import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react-native";
import { Provider } from "react-redux";
import { Provider as PaperProvider } from "react-native-paper";
import { NavigationContainer } from "@react-navigation/native";
import { configureStore } from "@reduxjs/toolkit";
import LoginScreen from "../../screens/LoginScreen";
import authReducer from "../../store/slices/authSlice";

const mockNavigate = jest.fn();

jest.mock("@react-navigation/native", () => {
  const actualNav = jest.requireActual("@react-navigation/native");
  return {
    ...actualNav,
    useNavigation: () => ({
      navigate: mockNavigate,
    }),
  };
});

const mockStore = configureStore({
  reducer: {
    auth: authReducer,
  },
});

const renderWithProviders = (component) => {
  return render(
    <Provider store={mockStore}>
      <PaperProvider>
        <NavigationContainer>{component}</NavigationContainer>
      </PaperProvider>
    </Provider>,
  );
};

describe("LoginScreen", () => {
  beforeEach(() => {
    mockNavigate.mockClear();
  });

  it("renders login form correctly", () => {
    renderWithProviders(<LoginScreen navigation={{ navigate: mockNavigate }} />);

    expect(screen.getByText("Welcome to AlphaMind")).toBeTruthy();
    expect(screen.getByText("Login to access your trading dashboard")).toBeTruthy();
    expect(screen.getByText("Login")).toBeTruthy();
  });

  it("allows input in email and password fields", () => {
    renderWithProviders(<LoginScreen navigation={{ navigate: mockNavigate }} />);

    const emailInput = screen.getByLabelText("Email");
    const passwordInput = screen.getByLabelText("Password");

    fireEvent.changeText(emailInput, "test@example.com");
    fireEvent.changeText(passwordInput, "password123");

    expect(emailInput.props.value).toBe("test@example.com");
    expect(passwordInput.props.value).toBe("password123");
  });

  it("login button is disabled with empty fields", () => {
    renderWithProviders(<LoginScreen navigation={{ navigate: mockNavigate }} />);

    const loginButton = screen.getByText("Login").parent;
    expect(loginButton.props.accessibilityState.disabled).toBe(true);
  });

  it("navigates to register screen", () => {
    renderWithProviders(<LoginScreen navigation={{ navigate: mockNavigate }} />);

    const registerButton = screen.getByText("Don't have an account? Register");
    fireEvent.press(registerButton);

    expect(mockNavigate).toHaveBeenCalledWith("Register");
  });
});
