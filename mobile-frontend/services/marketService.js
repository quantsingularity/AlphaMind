import api from "./api";
import { API_ENDPOINTS } from "../constants/config";

export const marketService = {
  getQuotes: async () => {
    const response = await api.get(API_ENDPOINTS.MARKET.QUOTES);
    return response.data;
  },

  getQuote: async (symbol) => {
    const response = await api.get(API_ENDPOINTS.MARKET.QUOTE(symbol));
    return response.data;
  },

  getHistory: async (symbol, days = 90) => {
    const response = await api.get(API_ENDPOINTS.MARKET.HISTORICAL(symbol), {
      params: { days },
    });
    return response.data;
  },
};

export default marketService;
