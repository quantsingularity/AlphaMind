import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import apiService from "../services/api";

export const usePortfolio = () => {
  return useQuery({
    queryKey: ["portfolio"],
    queryFn: () => apiService.getPortfolio(),
    staleTime: 5000, // 5 seconds
    refetchInterval: 10000, // Refetch every 10 seconds
  });
};

export const usePositions = () => {
  return useQuery({
    queryKey: ["positions"],
    queryFn: () => apiService.getPositions(),
    staleTime: 5000,
    refetchInterval: 10000,
  });
};

export const usePosition = (id: string) => {
  return useQuery({
    queryKey: ["position", id],
    queryFn: () => apiService.getPosition(id),
    enabled: !!id,
  });
};

export const useClosePosition = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => apiService.closePosition(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["positions"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
    },
  });
};
