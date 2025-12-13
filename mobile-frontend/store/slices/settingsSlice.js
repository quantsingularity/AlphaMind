import { createSlice } from "@reduxjs/toolkit";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { STORAGE_KEYS } from "../../constants/config";

const initialState = {
  theme: "system", // 'light', 'dark', 'system'
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
};

const settingsSlice = createSlice({
  name: "settings",
  initialState,
  reducers: {
    setTheme: (state, action) => {
      state.theme = action.payload;
      // Persist to AsyncStorage
      AsyncStorage.setItem(
        STORAGE_KEYS.SETTINGS,
        JSON.stringify({ ...state, theme: action.payload }),
      );
    },
    setNotifications: (state, action) => {
      state.notifications = { ...state.notifications, ...action.payload };
      // Persist to AsyncStorage
      AsyncStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(state));
    },
    setDisplayPreferences: (state, action) => {
      state.displayPreferences = { ...state.displayPreferences, ...action.payload };
      // Persist to AsyncStorage
      AsyncStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(state));
    },
    loadSettings: (state, action) => {
      return { ...state, ...action.payload };
    },
    resetSettings: () => {
      AsyncStorage.removeItem(STORAGE_KEYS.SETTINGS);
      return initialState;
    },
  },
});

export const { setTheme, setNotifications, setDisplayPreferences, loadSettings, resetSettings } =
  settingsSlice.actions;
export default settingsSlice.reducer;
