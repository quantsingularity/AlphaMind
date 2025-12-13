import api from "./api";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { API_ENDPOINTS, STORAGE_KEYS } from "../constants/config";

export const authService = {
  /**
   * Login user with email and password
   */
  login: async (email, password) => {
    try {
      const response = await api.post(API_ENDPOINTS.AUTH.LOGIN, {
        email,
        password,
      });

      if (response.data.token) {
        await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, response.data.token);
        if (response.data.user) {
          await AsyncStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(response.data.user));
        }
      }

      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Register new user
   */
  register: async (userData) => {
    try {
      const response = await api.post(API_ENDPOINTS.AUTH.REGISTER, userData);

      if (response.data.token) {
        await AsyncStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, response.data.token);
        if (response.data.user) {
          await AsyncStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(response.data.user));
        }
      }

      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Logout current user
   */
  logout: async () => {
    try {
      await api.post(API_ENDPOINTS.AUTH.LOGOUT);
    } catch (error) {
      console.error("Logout API error:", error);
    } finally {
      // Clear local storage regardless of API call success
      await AsyncStorage.multiRemove([STORAGE_KEYS.AUTH_TOKEN, STORAGE_KEYS.USER_DATA]);
    }
  },

  /**
   * Get current user profile
   */
  getProfile: async () => {
    try {
      const response = await api.get(API_ENDPOINTS.AUTH.PROFILE);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Check if user is authenticated
   */
  isAuthenticated: async () => {
    try {
      const token = await AsyncStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
      return !!token;
    } catch (error) {
      return false;
    }
  },

  /**
   * Get stored user data
   */
  getUserData: async () => {
    try {
      const userData = await AsyncStorage.getItem(STORAGE_KEYS.USER_DATA);
      return userData ? JSON.parse(userData) : null;
    } catch (error) {
      return null;
    }
  },
};
