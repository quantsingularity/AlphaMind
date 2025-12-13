import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { API_BASE_URL, API_TIMEOUT, STORAGE_KEYS } from "../constants/config";

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  async (config) => {
    try {
      const token = await AsyncStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (error) {
      console.error("Error retrieving auth token:", error);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response) {
      // Server responded with error status
      const { status } = error.response;

      if (status === 401) {
        // Unauthorized - clear token and redirect to login
        await AsyncStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
        await AsyncStorage.removeItem(STORAGE_KEYS.USER_DATA);
      }

      return Promise.reject({
        message: error.response.data?.message || "Server error occurred",
        status,
        data: error.response.data,
      });
    } else if (error.request) {
      // Request made but no response received
      return Promise.reject({
        message: "Network error - please check your connection",
        status: 0,
      });
    } else {
      // Something else happened
      return Promise.reject({
        message: error.message || "An unexpected error occurred",
        status: 0,
      });
    }
  },
);

export default api;
