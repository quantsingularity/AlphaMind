import AsyncStorage from "@react-native-async-storage/async-storage";
import { authService } from "../../services/authService";

jest.mock("../../services/api", () => ({
  __esModule: true,
  default: {
    post: jest.fn(),
    get: jest.fn(),
  },
}));

import api from "../../services/api";

describe("authService", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    AsyncStorage.clear();
  });

  describe("login", () => {
    it("stores token and user data on successful login", async () => {
      const mockResponse = {
        data: {
          token: "test-token-123",
          user: { id: 1, email: "test@example.com", name: "Test User" },
        },
      };
      api.post.mockResolvedValueOnce(mockResponse);

      const result = await authService.login("test@example.com", "password");

      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        "@alphamind/auth_token",
        "test-token-123",
      );
      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        "@alphamind/user_data",
        JSON.stringify(mockResponse.data.user),
      );
      expect(result).toEqual(mockResponse.data);
    });

    it("throws error when no token in response", async () => {
      api.post.mockResolvedValueOnce({ data: { user: { id: 1 } } });

      await expect(
        authService.login("test@example.com", "password"),
      ).rejects.toThrow("No authentication token received from server");
    });

    it("propagates API errors", async () => {
      api.post.mockRejectedValueOnce({ message: "Invalid credentials" });

      await expect(
        authService.login("test@example.com", "wrong-password"),
      ).rejects.toEqual({ message: "Invalid credentials" });
    });
  });

  describe("offline demo fallback", () => {
    it("falls back to a demo session on a connectivity error during login", async () => {
      // The api interceptor stamps connectivity failures with status 0.
      api.post.mockRejectedValueOnce({
        message: "Network error - please check your connection",
        status: 0,
      });

      const result = await authService.login("abrar@example.com", "secret");

      expect(result.token).toMatch(/^demo-/);
      expect(result.user.email).toBe("abrar@example.com");
      expect(result.demo).toBe(true);
      expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        "@alphamind/auth_token",
        result.token,
      );
    });

    it("falls back to a demo session on a connectivity error during register", async () => {
      api.post.mockRejectedValueOnce({
        message: "Network error - please check your connection",
        status: 0,
      });

      const result = await authService.register({
        name: "Abrar Ahmed",
        email: "abrar@example.com",
        password: "secret",
      });

      expect(result.token).toMatch(/^demo-/);
      expect(result.user.name).toBe("Abrar Ahmed");
      expect(result.user.email).toBe("abrar@example.com");
      expect(result.demo).toBe(true);
    });

    it("does NOT fall back when the server actively rejects (e.g. 401)", async () => {
      api.post.mockRejectedValueOnce({
        message: "Invalid credentials",
        status: 401,
      });
      await expect(
        authService.login("test@example.com", "wrong"),
      ).rejects.toEqual({ message: "Invalid credentials", status: 401 });
    });
  });

  describe("isAuthenticated", () => {
    it("returns true when token exists", async () => {
      await AsyncStorage.setItem("@alphamind/auth_token", "some-token");
      const result = await authService.isAuthenticated();
      expect(result).toBe(true);
    });

    it("returns false when no token", async () => {
      const result = await authService.isAuthenticated();
      expect(result).toBe(false);
    });
  });

  describe("getUserData", () => {
    it("returns parsed user data when stored", async () => {
      const user = { id: 1, email: "test@example.com" };
      await AsyncStorage.setItem("@alphamind/user_data", JSON.stringify(user));
      const result = await authService.getUserData();
      expect(result).toEqual(user);
    });

    it("returns null when no user data stored", async () => {
      const result = await authService.getUserData();
      expect(result).toBeNull();
    });
  });

  describe("logout", () => {
    it("clears storage on logout", async () => {
      await AsyncStorage.setItem("@alphamind/auth_token", "token");
      await AsyncStorage.setItem("@alphamind/user_data", "{}");
      api.post.mockResolvedValueOnce({});

      await authService.logout();

      expect(AsyncStorage.multiRemove).toHaveBeenCalledWith([
        "@alphamind/auth_token",
        "@alphamind/user_data",
      ]);
    });

    it("clears storage even when API call fails", async () => {
      api.post.mockRejectedValueOnce(new Error("Network error"));

      await authService.logout();

      expect(AsyncStorage.multiRemove).toHaveBeenCalled();
    });
  });
});
