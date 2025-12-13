import AsyncStorage from "@react-native-async-storage/async-storage";
import { authService } from "../../services/authService";
import api from "../../services/api";

jest.mock("../../services/api");
jest.mock("@react-native-async-storage/async-storage");

describe("AuthService", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("login", () => {
    it("should login successfully and store token", async () => {
      const mockResponse = {
        data: {
          token: "test-token",
          user: { id: 1, email: "test@example.com", name: "Test User" },
        },
      };

      api.post.mockResolvedValue(mockResponse);
      AsyncStorage.setItem.mockResolvedValue();

      const result = await authService.login("test@example.com", "password123");

      expect(api.post).toHaveBeenCalledWith("/api/auth/login", {
        email: "test@example.com",
        password: "password123",
      });
      expect(AsyncStorage.setItem).toHaveBeenCalledWith("@alphamind/auth_token", "test-token");
      expect(result).toEqual(mockResponse.data);
    });

    it("should throw error on login failure", async () => {
      api.post.mockRejectedValue(new Error("Invalid credentials"));

      await expect(authService.login("test@example.com", "wrong")).rejects.toThrow(
        "Invalid credentials",
      );
    });
  });

  describe("isAuthenticated", () => {
    it("should return true when token exists", async () => {
      AsyncStorage.getItem.mockResolvedValue("test-token");

      const result = await authService.isAuthenticated();

      expect(result).toBe(true);
      expect(AsyncStorage.getItem).toHaveBeenCalledWith("@alphamind/auth_token");
    });

    it("should return false when token does not exist", async () => {
      AsyncStorage.getItem.mockResolvedValue(null);

      const result = await authService.isAuthenticated();

      expect(result).toBe(false);
    });
  });

  describe("logout", () => {
    it("should clear storage on logout", async () => {
      api.post.mockResolvedValue({});
      AsyncStorage.multiRemove.mockResolvedValue();

      await authService.logout();

      expect(api.post).toHaveBeenCalledWith("/api/auth/logout");
      expect(AsyncStorage.multiRemove).toHaveBeenCalledWith([
        "@alphamind/auth_token",
        "@alphamind/user_data",
      ]);
    });

    it("should clear storage even if API call fails", async () => {
      api.post.mockRejectedValue(new Error("Network error"));
      AsyncStorage.multiRemove.mockResolvedValue();

      await authService.logout();

      expect(AsyncStorage.multiRemove).toHaveBeenCalled();
    });
  });
});
