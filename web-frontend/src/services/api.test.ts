import { beforeEach, describe, expect, it, vi } from "vitest";

// Capture the axios instance the service creates so we can assert the exact
// URLs each method calls. This pins the client side of the API contract and
// would catch endpoint drift (for example, market-data pointing at the bare
// root instead of /quotes).
const { mockInstance } = vi.hoisted(() => {
  return {
    mockInstance: {
      get: vi.fn(() => Promise.resolve({ data: [] })),
      post: vi.fn(() => Promise.resolve({ data: {} })),
      delete: vi.fn(() => Promise.resolve({ data: {} })),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    },
  };
});

vi.mock("axios", () => ({
  default: { create: () => mockInstance },
}));

import { apiService } from "./api";

describe("API client contract", () => {
  beforeEach(() => {
    mockInstance.get.mockClear();
    mockInstance.post.mockClear();
    mockInstance.delete.mockClear();
  });

  it("auth endpoints", async () => {
    await apiService.login({ email: "a@b.io", password: "x" });
    expect(mockInstance.post).toHaveBeenCalledWith("/api/auth/login", {
      email: "a@b.io",
      password: "x",
    });

    await apiService.register({ name: "A", email: "a@b.io", password: "x" });
    expect(mockInstance.post).toHaveBeenCalledWith(
      "/api/auth/register",
      expect.objectContaining({ email: "a@b.io" }),
    );
  });

  it("portfolio endpoints", async () => {
    await apiService.getPortfolio();
    expect(mockInstance.get).toHaveBeenCalledWith("/api/v1/portfolio/");
    await apiService.getPositions();
    expect(mockInstance.get).toHaveBeenCalledWith(
      "/api/v1/portfolio/positions",
    );
  });

  it("market data endpoints use /quotes (not the bare root)", async () => {
    await apiService.getMarketData();
    expect(mockInstance.get).toHaveBeenCalledWith("/api/v1/market-data/quotes");

    await apiService.getQuote("AAPL");
    expect(mockInstance.get).toHaveBeenCalledWith(
      "/api/v1/market-data/quote/AAPL",
    );

    await apiService.getHistoricalData("AAPL", 30);
    expect(mockInstance.get).toHaveBeenCalledWith(
      "/api/v1/market-data/historical/AAPL",
      expect.objectContaining({
        params: expect.objectContaining({ days: 30 }),
      }),
    );
  });

  it("trading endpoints", async () => {
    await apiService.getOrders();
    expect(mockInstance.get).toHaveBeenCalledWith("/api/v1/trading/orders");

    await apiService.createOrder({
      ticker: "AAPL",
      side: "BUY",
      quantity: 1,
      orderType: "MARKET",
    });
    expect(mockInstance.post).toHaveBeenCalledWith(
      "/api/v1/trading/orders",
      expect.objectContaining({ ticker: "AAPL" }),
    );

    await apiService.cancelOrder("order-1");
    expect(mockInstance.delete).toHaveBeenCalledWith(
      "/api/v1/trading/orders/order-1",
    );
  });

  it("research and alternative-data endpoints", async () => {
    await apiService.getResearchPapers();
    expect(mockInstance.get).toHaveBeenCalledWith(
      "/api/v1/research/papers",
      expect.anything(),
    );

    await apiService.getAlternativeDataSources();
    expect(mockInstance.get).toHaveBeenCalledWith(
      "/api/v1/alternative-data/sources",
    );
  });
});
