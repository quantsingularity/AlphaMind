import { useQuery } from "@tanstack/react-query";
import { apiService } from "../services/api";
import type { AlternativeDataSource } from "../types";

export function useAlternativeDataSources() {
  return useQuery<AlternativeDataSource[]>({
    queryKey: ["alternative-data-sources"],
    queryFn: () => apiService.getAlternativeDataSources(),
  });
}
