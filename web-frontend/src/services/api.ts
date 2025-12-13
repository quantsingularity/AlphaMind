import axios, { type AxiosInstance, type AxiosError } from "axios";
import type {
  Strategy,
  MarketData,
  Portfolio,
  Position,
  Order,
  BacktestResult,
  RiskMetrics,
  AlternativeDataSource,
  ApiResponse,
  ApiError,
} from "../types";

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:5000",
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Request interceptor for adding auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem("authToken");
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error),
    );

    // Response interceptor for handling errors
    this.api.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiError>) => {
        const apiError: ApiError = {
          message:
            error.response?.data?.message ||
            error.message ||
            "An error occurred",
          code: error.response?.data?.code || error.code || "UNKNOWN_ERROR",
          details: error.response?.data?.details,
        };
        return Promise.reject(apiError);
      },
    );
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string }>> {
    const response = await this.api.get("/health");
    return response.data;
  }

  // Strategy endpoints
  async getStrategies(): Promise<Strategy[]> {
    const response =
      await this.api.get<ApiResponse<Strategy[]>>("/api/strategies");
    return response.data.data;
  }

  async getStrategy(id: string): Promise<Strategy> {
    const response = await this.api.get<ApiResponse<Strategy>>(
      `/api/strategies/${id}`,
    );
    return response.data.data;
  }

  async createStrategy(strategy: Partial<Strategy>): Promise<Strategy> {
    const response = await this.api.post<ApiResponse<Strategy>>(
      "/api/strategies",
      strategy,
    );
    return response.data.data;
  }

  async updateStrategy(
    id: string,
    strategy: Partial<Strategy>,
  ): Promise<Strategy> {
    const response = await this.api.put<ApiResponse<Strategy>>(
      `/api/strategies/${id}`,
      strategy,
    );
    return response.data.data;
  }

  async deleteStrategy(id: string): Promise<void> {
    await this.api.delete(`/api/strategies/${id}`);
  }

  async activateStrategy(id: string): Promise<Strategy> {
    const response = await this.api.post<ApiResponse<Strategy>>(
      `/api/strategies/${id}/activate`,
    );
    return response.data.data;
  }

  async deactivateStrategy(id: string): Promise<Strategy> {
    const response = await this.api.post<ApiResponse<Strategy>>(
      `/api/strategies/${id}/deactivate`,
    );
    return response.data.data;
  }

  // Market data endpoints
  async getMarketData(
    ticker: string,
    interval?: string,
  ): Promise<MarketData[]> {
    const response = await this.api.get<ApiResponse<MarketData[]>>(
      "/api/market-data",
      {
        params: { ticker, interval },
      },
    );
    return response.data.data;
  }

  async subscribeToMarketData(
    tickers: string[],
    callback: (data: MarketData) => void,
  ): Promise<() => void> {
    // WebSocket implementation would go here
    // For now, use polling
    const interval = setInterval(async () => {
      for (const ticker of tickers) {
        try {
          const data = await this.getMarketData(ticker);
          if (data.length > 0) {
            callback(data[0]);
          }
        } catch (error) {
          console.error(`Error fetching market data for ${ticker}:`, error);
        }
      }
    }, 5000);

    // Return a cleanup function
    return () => clearInterval(interval);
  }

  // Portfolio endpoints
  async getPortfolio(): Promise<Portfolio> {
    const response =
      await this.api.get<ApiResponse<Portfolio>>("/api/portfolio");
    return response.data.data;
  }

  async getPositions(): Promise<Position[]> {
    const response =
      await this.api.get<ApiResponse<Position[]>>("/api/positions");
    return response.data.data;
  }

  async getPosition(id: string): Promise<Position> {
    const response = await this.api.get<ApiResponse<Position>>(
      `/api/positions/${id}`,
    );
    return response.data.data;
  }

  async closePosition(id: string): Promise<void> {
    await this.api.post(`/api/positions/${id}/close`);
  }

  // Order endpoints
  async getOrders(): Promise<Order[]> {
    const response = await this.api.get<ApiResponse<Order[]>>("/api/orders");
    return response.data.data;
  }

  async createOrder(order: Partial<Order>): Promise<Order> {
    const response = await this.api.post<ApiResponse<Order>>(
      "/api/orders",
      order,
    );
    return response.data.data;
  }

  async cancelOrder(id: string): Promise<void> {
    await this.api.delete(`/api/orders/${id}`);
  }

  // Backtesting endpoints
  async runBacktest(
    strategyId: string,
    startDate: string,
    endDate: string,
    initialCapital: number,
  ): Promise<BacktestResult> {
    const response = await this.api.post<ApiResponse<BacktestResult>>(
      "/api/backtest",
      {
        strategyId,
        startDate,
        endDate,
        initialCapital,
      },
    );
    return response.data.data;
  }

  async getBacktestResults(strategyId: string): Promise<BacktestResult[]> {
    const response = await this.api.get<ApiResponse<BacktestResult[]>>(
      `/api/backtest/${strategyId}`,
    );
    return response.data.data;
  }

  // Risk metrics endpoints
  async getRiskMetrics(portfolioId?: string): Promise<RiskMetrics> {
    const response = await this.api.get<ApiResponse<RiskMetrics>>(
      "/api/risk/metrics",
      {
        params: { portfolioId },
      },
    );
    return response.data.data;
  }

  // Alternative data endpoints
  async getAlternativeDataSources(): Promise<AlternativeDataSource[]> {
    const response = await this.api.get<ApiResponse<AlternativeDataSource[]>>(
      "/api/alternative-data/sources",
    );
    return response.data.data;
  }

  async getAlternativeData(
    sourceId: string,
    params?: Record<string, any>,
  ): Promise<any> {
    const response = await this.api.get<ApiResponse<any>>(
      `/api/alternative-data/${sourceId}`,
      { params },
    );
    return response.data.data;
  }
}

export const apiService = new ApiService();
export default apiService;
