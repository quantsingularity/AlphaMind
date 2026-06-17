import api from "./api";
import { API_ENDPOINTS } from "../constants/config";

export const tradingService = {
  getOrders: async (status) => {
    const response = await api.get(API_ENDPOINTS.TRADING.ORDERS, {
      params: status ? { status } : undefined,
    });
    return response.data;
  },

  createOrder: async ({ ticker, side, quantity, orderType, price }) => {
    const body = { ticker, side, quantity, orderType };
    if (price != null) body.price = price;
    const response = await api.post(API_ENDPOINTS.TRADING.ORDERS, body);
    return response.data;
  },

  cancelOrder: async (id) => {
    await api.delete(API_ENDPOINTS.TRADING.CANCEL(id));
    return id;
  },
};

export default tradingService;
