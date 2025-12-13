import { API_BASE_URL, API_ENDPOINTS, STORAGE_KEYS } from "../../constants/config";

describe("Configuration Constants", () => {
  it("should have API_BASE_URL defined", () => {
    expect(API_BASE_URL).toBeDefined();
    expect(typeof API_BASE_URL).toBe("string");
  });

  it("should have API_ENDPOINTS defined", () => {
    expect(API_ENDPOINTS).toBeDefined();
    expect(API_ENDPOINTS.AUTH).toBeDefined();
    expect(API_ENDPOINTS.PORTFOLIO).toBeDefined();
  });

  it("should have correct AUTH endpoints", () => {
    expect(API_ENDPOINTS.AUTH.LOGIN).toBe("/api/auth/login");
    expect(API_ENDPOINTS.AUTH.REGISTER).toBe("/api/auth/register");
    expect(API_ENDPOINTS.AUTH.LOGOUT).toBe("/api/auth/logout");
  });

  it("should have STORAGE_KEYS defined", () => {
    expect(STORAGE_KEYS).toBeDefined();
    expect(STORAGE_KEYS.AUTH_TOKEN).toBe("@alphamind/auth_token");
    expect(STORAGE_KEYS.USER_DATA).toBe("@alphamind/user_data");
  });

  it("should have dynamic endpoint functions", () => {
    expect(typeof API_ENDPOINTS.PORTFOLIO.DETAILS).toBe("function");
    expect(API_ENDPOINTS.PORTFOLIO.DETAILS("123")).toBe("/api/portfolio/123");
  });
});
