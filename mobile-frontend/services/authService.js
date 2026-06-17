import AsyncStorage from "@react-native-async-storage/async-storage";
import { API_ENDPOINTS, STORAGE_KEYS } from "../constants/config";
import api from "./api";

// Persist a session so the request interceptor and checkAuth can read it.
const persistSession = async (token, user) => {
  await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
  if (user) {
    await AsyncStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(user));
  }
};

// The API response interceptor stamps connectivity failures (timeout, no
// response, unexpected) with status 0. Real backend rejections carry the HTTP
// status (401, 400, ...), so we only fall back to a demo session when the
// server genuinely could not be reached. This mirrors the web client and lets
// sign up / sign in work even with no backend running.
const isOffline = (error) => Boolean(error) && error.status === 0;

const deriveName = (email) => {
  if (!email) return "AlphaMind Trader";
  const handle = String(email)
    .split("@")[0]
    .replace(/[._-]+/g, " ")
    .trim();
  if (!handle) return "AlphaMind Trader";
  return handle
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const makeDemoSession = async (user) => {
  const token = `demo-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  await persistSession(token, user);
  return { token, user, demo: true };
};

export const authService = {
  login: async (email, password) => {
    try {
      const response = await api.post(API_ENDPOINTS.AUTH.LOGIN, {
        email,
        password,
      });

      const { token, user } = response.data;

      if (!token) {
        throw new Error("No authentication token received from server");
      }

      await persistSession(token, user);
      return response.data;
    } catch (error) {
      if (isOffline(error)) {
        return makeDemoSession({
          id: `demo-${Date.now()}`,
          name: deriveName(email),
          email: email || "demo@alphamind.io",
          demo: true,
        });
      }
      throw error;
    }
  },

  register: async (userData) => {
    try {
      const response = await api.post(API_ENDPOINTS.AUTH.REGISTER, userData);

      const { token, user } = response.data;

      if (!token) {
        throw new Error("No authentication token received from server");
      }

      await persistSession(token, user);
      return response.data;
    } catch (error) {
      if (isOffline(error)) {
        return makeDemoSession({
          id: `demo-${Date.now()}`,
          name: userData?.name || deriveName(userData?.email),
          email: userData?.email || "demo@alphamind.io",
          demo: true,
        });
      }
      throw error;
    }
  },

  logout: async () => {
    try {
      await api.post(API_ENDPOINTS.AUTH.LOGOUT);
    } catch (error) {
      console.error("Logout API error:", error);
    } finally {
      await AsyncStorage.multiRemove([
        STORAGE_KEYS.AUTH_TOKEN,
        STORAGE_KEYS.USER_DATA,
      ]);
    }
  },

  getProfile: async () => {
    const response = await api.get(API_ENDPOINTS.AUTH.PROFILE);
    return response.data;
  },

  isAuthenticated: async () => {
    try {
      const token = await AsyncStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
      return !!token;
    } catch (_error) {
      return false;
    }
  },

  getUserData: async () => {
    try {
      const userData = await AsyncStorage.getItem(STORAGE_KEYS.USER_DATA);
      return userData ? JSON.parse(userData) : null;
    } catch (_error) {
      return null;
    }
  },

  // Used on app startup. A real backend session is restored so the user stays
  // logged in; an offline demo session is intentionally NOT restored (and is
  // cleared) so the app always starts on the homepage rather than dropping the
  // user straight into a stale demo dashboard.
  getPersistedSession: async () => {
    try {
      const token = await AsyncStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
      if (!token) {
        return { isAuthenticated: false, user: null };
      }
      if (token.startsWith("demo-")) {
        await AsyncStorage.multiRemove([
          STORAGE_KEYS.AUTH_TOKEN,
          STORAGE_KEYS.USER_DATA,
        ]);
        return { isAuthenticated: false, user: null };
      }
      const userData = await AsyncStorage.getItem(STORAGE_KEYS.USER_DATA);
      return {
        isAuthenticated: true,
        user: userData ? JSON.parse(userData) : null,
      };
    } catch (_error) {
      return { isAuthenticated: false, user: null };
    }
  },

  refreshToken: async () => {
    const response = await api.post(API_ENDPOINTS.AUTH.REFRESH);
    const { token } = response.data;
    if (token) {
      await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
    }
    return response.data;
  },
};

export default authService;
