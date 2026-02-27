import Constants from "expo-constants";

// API Configuration
export const API_BASE_URL =
  Constants.expoConfig?.extra?.apiBaseUrl ||
  process.env.API_BASE_URL ||
  "http://localhost:5000";

export const API_TIMEOUT = 30000;

// Feature Flags
export const ENABLE_MOCK_DATA = process.env.ENABLE_MOCK_DATA === "true";
export const ENABLE_OFFLINE_MODE = process.env.ENABLE_OFFLINE_MODE !== "false";

// AsyncStorage Keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: "@alphamind/auth_token",
  USER_DATA: "@alphamind/user_data",
  THEME_PREFERENCE: "@alphamind/theme_preference",
  SETTINGS: "@alphamind/settings",
};

// API Endpoints
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: "/api/auth/login",
    REGISTER: "/api/auth/register",
    LOGOUT: "/api/auth/logout",
    REFRESH: "/api/auth/refresh",
    PROFILE: "/api/auth/profile",
  },
  PORTFOLIO: {
    LIST: "/api/portfolio",
    DETAILS: (id) => `/api/portfolio/${id}`,
    PERFORMANCE: "/api/portfolio/performance",
    HOLDINGS: "/api/portfolio/holdings",
  },
  STRATEGIES: {
    LIST: "/api/strategies",
    DETAILS: (id) => `/api/strategies/${id}`,
    PERFORMANCE: (id) => `/api/strategies/${id}/performance`,
  },
  MARKET: {
    QUOTES: "/api/market/quotes",
    CHART: "/api/market/chart",
  },
  RESEARCH: {
    PAPERS: "/api/research/papers",
    DETAILS: (id) => `/api/research/papers/${id}`,
  },
};
