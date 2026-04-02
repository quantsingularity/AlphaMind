import "@testing-library/react-native/extend-expect";

jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

jest.mock("react-native-vector-icons/MaterialCommunityIcons", () => "Icon");

jest.mock("expo-status-bar", () => ({
  StatusBar: "StatusBar",
}));

jest.mock("expo-constants", () => ({
  default: {
    expoConfig: {
      extra: {
        apiBaseUrl: "http://localhost:5000",
      },
    },
  },
}));

jest.mock("react-native-svg", () => {
  const React = require("react");
  const MockSvg = ({ children }) => React.createElement("Svg", {}, children);
  return {
    Svg: MockSvg,
    Circle: "Circle",
    G: "G",
    Path: "Path",
    Rect: "Rect",
    Line: "Line",
    Text: "Text",
    TSpan: "TSpan",
    Defs: "Defs",
    LinearGradient: "LinearGradient",
    Stop: "Stop",
  };
});

jest.mock("react-native-paper", () => {
  const RealModule = jest.requireActual("react-native-paper");
  return RealModule;
});

global.console = {
  ...console,
  warn: jest.fn(),
  error: jest.fn(),
};
