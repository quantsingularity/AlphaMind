import babelParser from "@babel/eslint-parser";
import js from "@eslint/js";
import reactPlugin from "eslint-plugin-react";
import reactNativePlugin from "eslint-plugin-react-native";

export default [
  {
    ignores: ["node_modules/**"],
  },
  js.configs.recommended,
  {
    files: ["**/*.{js,jsx}"], // Apply rules to JS and JSX files
    languageOptions: {
      parser: babelParser,
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
        requireConfigFile: false, // Important for @babel/eslint-parser
        babelOptions: {
          presets: ["@babel/preset-react"], // Or your specific Babel config
        },
      },
      globals: {
        // Define global variables if needed (e.g., for Jest tests)
        __DEV__: true,
        jest: true,
        expect: true,
        test: true,
        it: true,
        describe: true,
        beforeEach: true,
        afterEach: true,
      },
    },
    plugins: {
      react: reactPlugin,
      "react-native": reactNativePlugin,
    },
    rules: {
      // Start with recommended rules for React and React Native
      ...reactPlugin.configs.recommended.rules,
      ...reactNativePlugin.configs.all.rules,
      // Add specific rule overrides if necessary
      "react/react-in-jsx-scope": "off", // Often not needed with modern React
      "react-native/no-unused-styles": "warn",
      "react-native/split-platform-components": "warn",
      "react-native/no-inline-styles": "warn",
      "react-native/no-color-literals": "warn",
      "react-native/no-raw-text": "warn", // Can be strict, adjust as needed
    },
    settings: {
      react: {
        version: "detect", // Automatically detect React version
      },
    },
  },
];
