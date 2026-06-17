import { useQuery } from "@tanstack/react-query";
import { apiService } from "../services/api";
import type { MarketData, OHLCV } from "../types";

export function useMarketQuotes() {
  return useQuery<MarketData[]>({
    queryKey: ["market-quotes"],
    queryFn: () => apiService.getMarketData(),
    refetchInterval: 15000,
  });
}

export function useMarketHistory(symbol: string | null, days = 90) {
  return useQuery<OHLCV[]>({
    queryKey: ["market-history", symbol, days],
    queryFn: () => apiService.getHistoricalData(symbol as string, days),
    enabled: Boolean(symbol),
  });
}
