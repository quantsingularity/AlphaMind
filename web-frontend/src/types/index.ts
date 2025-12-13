// Core types for AlphaMind application

export interface Strategy {
  id: string;
  name: string;
  description: string;
  type: "TFT" | "RL" | "HYBRID" | "SENTIMENT";
  status: "active" | "inactive" | "backtesting";
  performance: StrategyPerformance;
  parameters: StrategyParameters;
  createdAt: string;
  updatedAt: string;
}

export interface StrategyPerformance {
  sharpeRatio: number;
  maxDrawdown: number;
  profitFactor: number;
  winRate: number;
  totalReturn: number;
  volatility: number;
  alpha: number;
  beta: number;
}

export interface StrategyParameters {
  [key: string]: string | number | boolean;
}

export interface MarketData {
  ticker: string;
  timestamp: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
}

export interface Position {
  id: string;
  ticker: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  timestamp: string;
}

export interface Order {
  id: string;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  orderType: "MARKET" | "LIMIT" | "STOP";
  price?: number;
  status: "pending" | "filled" | "cancelled" | "rejected";
  timestamp: string;
  filledAt?: string;
}

export interface Portfolio {
  id: string;
  name: string;
  totalValue: number;
  cash: number;
  positions: Position[];
  dailyPnL: number;
  totalPnL: number;
  allocation: AssetAllocation[];
}

export interface AssetAllocation {
  ticker: string;
  value: number;
  percentage: number;
}

export interface RiskMetrics {
  var: number; // Value at Risk
  cvar: number; // Conditional Value at Risk
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  beta: number;
  correlation: number;
  volatility: number;
}

export interface BacktestResult {
  id: string;
  strategyId: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  performance: StrategyPerformance;
  trades: Trade[];
  equityCurve: EquityPoint[];
  metrics: RiskMetrics;
}

export interface Trade {
  id: string;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  timestamp: string;
  pnl: number;
}

export interface EquityPoint {
  timestamp: string;
  value: number;
}

export interface AlternativeDataSource {
  id: string;
  name: string;
  type: "satellite" | "sentiment" | "sec" | "social";
  status: "active" | "inactive";
  lastUpdate: string;
  dataPoints: number;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface ApiError {
  message: string;
  code: string;
  details?: any;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: "admin" | "trader" | "analyst";
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: "light" | "dark";
  notifications: boolean;
  defaultStrategy?: string;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  borderColor?: string;
  backgroundColor?: string;
}
