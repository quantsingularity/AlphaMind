import AsyncStorage from "@react-native-async-storage/async-storage";
import { createListenerMiddleware } from "@reduxjs/toolkit";
import { STORAGE_KEYS } from "../constants/config";
import {
  setDisplayPreferences,
  setNotifications,
  setTheme,
} from "./slices/settingsSlice";

export const settingsListenerMiddleware = createListenerMiddleware();

const persistCurrentSettings = async (getState) => {
  try {
    const settings = getState().settings;
    await AsyncStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
  } catch (error) {
    console.error("Failed to persist settings:", error);
  }
};

settingsListenerMiddleware.startListening({
  actionCreator: setTheme,
  effect: async (_, listenerApi) => {
    await persistCurrentSettings(listenerApi.getState);
  },
});

settingsListenerMiddleware.startListening({
  actionCreator: setNotifications,
  effect: async (_, listenerApi) => {
    await persistCurrentSettings(listenerApi.getState);
  },
});

settingsListenerMiddleware.startListening({
  actionCreator: setDisplayPreferences,
  effect: async (_, listenerApi) => {
    await persistCurrentSettings(listenerApi.getState);
  },
});
