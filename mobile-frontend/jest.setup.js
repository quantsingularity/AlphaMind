// Mock AsyncStorage
jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

// Mock React Native Vector Icons
jest.mock("react-native-vector-icons/MaterialCommunityIcons", () => "Icon");

// Mock expo-status-bar
jest.mock("expo-status-bar", () => ({
  StatusBar: "StatusBar",
}));

// Mock expo-constants
jest.mock("expo-constants", () => ({
  default: {
    expoConfig: {
      extra: {
        apiBaseUrl: "http://localhost:5000",
      },
    },
  },
}));

// Mock react-native-svg for Victory Native
jest.mock("react-native-svg", () => {
  const React = require("react");
  return {
    Svg: ({ children }) => React.createElement("Svg", {}, children),
    Circle: "Circle",
    G: "G",
    Path: "Path",
    Rect: "Rect",
    Line: "Line",
    Text: "Text",
    TSpan: "TSpan",
  };
});

// Suppress console warnings during tests
global.console = {
  ...console,
  warn: jest.fn(),
};
