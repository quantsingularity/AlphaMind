import { configureStore } from "@reduxjs/toolkit";
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
        // Ignore these action types
        ignoredActions: ["persist/PERSIST"],
      },
    }),
});

export default store;
