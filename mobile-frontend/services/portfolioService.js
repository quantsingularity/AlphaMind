import { API_ENDPOINTS, ENABLE_MOCK_DATA } from "../constants/config";
import api from "./api";

// Mock data for development
const mockPortfolioData = {
  value: 1250345.67,
  dailyPnL: 15678.9,
  dailyPnLPercent: 1.27,
  sharpeRatio: 2.35,
  activeStrategies: 12,
  performance: [
    { date: "2025-12-01", value: 1200000 },
    { date: "2025-12-05", value: 1220000 },
    { date: "2025-12-08", value: 1235000 },
    { date: "2025-12-13", value: 1250345.67 },
  ],
  holdings: [
    { symbol: "AAPL", shares: 1000, value: 180000, weight: 14.4 },
    { symbol: "GOOGL", shares: 500, value: 150000, weight: 12.0 },
    { symbol: "MSFT", shares: 800, value: 240000, weight: 19.2 },
    { symbol: "TSLA", shares: 400, value: 100000, weight: 8.0 },
  ],
};

export const portfolioService = {
  /**
   * Get portfolio overview
   */
  getPortfolio: async () => {
    if (ENABLE_MOCK_DATA) {
      // Return mock data for development
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockPortfolioData), 500);
      });
    }
    const response = await api.get(API_ENDPOINTS.PORTFOLIO.LIST);
    return response.data;
  },

  /**
   * Get portfolio performance data
   */
  getPerformance: async (timeframe = "1M") => {
    if (ENABLE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockPortfolioData.performance), 500);
      });
    }
    const response = await api.get(API_ENDPOINTS.PORTFOLIO.PERFORMANCE, {
      params: { timeframe },
    });
    return response.data;
  },

  /**
   * Get portfolio holdings
   */
  getHoldings: async () => {
    if (ENABLE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockPortfolioData.holdings), 500);
      });
    }
    const response = await api.get(API_ENDPOINTS.PORTFOLIO.HOLDINGS);
    return response.data;
  },

  /**
   * Get portfolio details by ID
   */
  getPortfolioDetails: async (id) => {
    const response = await api.get(API_ENDPOINTS.PORTFOLIO.DETAILS(id));
    return response.data;
  },
};
