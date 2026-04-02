import AsyncStorage from "@react-native-async-storage/async-storage";
import { API_ENDPOINTS, STORAGE_KEYS } from "../constants/config";
import api from "./api";

export const authService = {
  login: async (email, password) => {
    const response = await api.post(API_ENDPOINTS.AUTH.LOGIN, {
      email,
      password,
    });

    const { token, user } = response.data;

    if (!token) {
      throw new Error("No authentication token received from server");
    }

    await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
    if (user) {
      await AsyncStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(user));
    }

    return response.data;
  },

  register: async (userData) => {
    const response = await api.post(API_ENDPOINTS.AUTH.REGISTER, userData);

    const { token, user } = response.data;

    if (!token) {
      throw new Error("No authentication token received from server");
    }

    await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
    if (user) {
      await AsyncStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(user));
    }

    return response.data;
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

  refreshToken: async () => {
    const response = await api.post(API_ENDPOINTS.AUTH.REFRESH);
    const { token } = response.data;
    if (token) {
      await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
    }
    return response.data;
  },
};
