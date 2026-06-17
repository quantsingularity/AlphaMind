import { useQuery } from "@tanstack/react-query";
import { apiService } from "../services/api";
import type { ResearchPaper } from "../types";

export function useResearchPapers(category?: string) {
  return useQuery<ResearchPaper[]>({
    queryKey: ["research-papers", category ?? "all"],
    queryFn: () => apiService.getResearchPapers(category),
  });
}
