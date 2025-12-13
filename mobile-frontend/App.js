import React, { useEffect } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { Provider as PaperProvider } from "react-native-paper";
import { Provider as ReduxProvider } from "react-redux";
import { useDispatch, useSelector } from "react-redux";
import { StatusBar } from "expo-status-bar";
import store from "./store";
import { lightTheme, darkTheme } from "./constants/theme";
import { checkAuth } from "./store/slices/authSlice";
import AuthNavigator from "./navigation/AuthNavigator";
import AppNavigator from "./navigation/AppNavigator";
import LoadingSpinner from "./components/LoadingSpinner";

function AppContent() {
  const dispatch = useDispatch();
  const { isAuthenticated, loading } = useSelector((state) => state.auth);
  const { theme: themePreference } = useSelector((state) => state.settings);

  useEffect(() => {
    dispatch(checkAuth());
  }, [dispatch]);

  // Determine theme based on preference
  const theme = themePreference === "dark" ? darkTheme : lightTheme;

  if (loading) {
    return <LoadingSpinner message="Loading AlphaMind..." />;
  }

  return (
    <PaperProvider theme={theme}>
      <StatusBar style={themePreference === "dark" ? "light" : "dark"} />
      <NavigationContainer>
        {isAuthenticated ? <AppNavigator /> : <AuthNavigator />}
      </NavigationContainer>
    </PaperProvider>
  );
}

export default function App() {
  return (
    <ReduxProvider store={store}>
      <AppContent />
    </ReduxProvider>
  );
}
