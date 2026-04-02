import AsyncStorage from "@react-native-async-storage/async-storage";
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { STORAGE_KEYS } from "../../constants/config";

const initialState = {
  theme: "system",
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

export const loadSavedSettings = createAsyncThunk(
  "settings/loadSaved",
  async () => {
    const saved = await AsyncStorage.getItem(STORAGE_KEYS.SETTINGS);
    return saved ? JSON.parse(saved) : null;
  },
);

export const persistSettings = createAsyncThunk(
  "settings/persist",
  async (_, { getState }) => {
    const settings = getState().settings;
    await AsyncStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
  },
);

const settingsSlice = createSlice({
  name: "settings",
  initialState,
  reducers: {
    setTheme: (state, action) => {
      state.theme = action.payload;
    },
    setNotifications: (state, action) => {
      state.notifications = { ...state.notifications, ...action.payload };
    },
    setDisplayPreferences: (state, action) => {
      state.displayPreferences = {
        ...state.displayPreferences,
        ...action.payload,
      };
    },
    resetSettings: () => {
      return initialState;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(loadSavedSettings.fulfilled, (state, action) => {
      if (action.payload) {
        return { ...state, ...action.payload };
      }
    });
  },
});

export const {
  setTheme,
  setNotifications,
  setDisplayPreferences,
  resetSettings,
} = settingsSlice.actions;
export default settingsSlice.reducer;
