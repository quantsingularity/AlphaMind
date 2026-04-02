import { configureStore } from "@reduxjs/toolkit";
import { settingsListenerMiddleware } from "./settingsListener";
import authReducer from "./slices/authSlice";
import portfolioReducer from "./slices/portfolioSlice";
import settingsReducer from "./slices/settingsSlice";

export const store = configureStore({
  reducer: {
    auth: authReducer,
    portfolio: portfolioReducer,
    settings: settingsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ["persist/PERSIST"],
      },
    }).prepend(settingsListenerMiddleware.middleware),
});

export default store;
