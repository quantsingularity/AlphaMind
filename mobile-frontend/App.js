import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { Provider as PaperProvider } from "react-native-paper";
import AppNavigator from "./navigation/AppNavigator";

// Optional: Define a custom theme or use the default
// import { DefaultTheme } from 'react-native-paper';
// const theme = {
//   ...DefaultTheme,
//   colors: {
//     ...DefaultTheme.colors,
//     primary: 'tomato',
//     accent: 'yellow',
//   },
// };

export default function App() {
  return (
    <PaperProvider>
      {/* Wrap the app with PaperProvider */}
      <NavigationContainer>
        <AppNavigator />
      </NavigationContainer>
    </PaperProvider>
  );
}

