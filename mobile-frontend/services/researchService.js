import api from "./api";
import { API_ENDPOINTS, ENABLE_MOCK_DATA } from "../constants/config";

// Mock research data
const mockResearchPapers = [
  {
    id: "1",
    title: "Deep Learning for Market Prediction",
    summary: "Exploring the efficacy of LSTM networks in forecasting short-term market movements.",
    authors: ["Dr. John Smith", "Dr. Jane Doe"],
    date: "2025-11-15",
    category: "Machine Learning",
    url: "https://example.com/paper1.pdf",
  },
  {
    id: "2",
    title: "Factor Investing with Alternative Data",
    summary:
      "A study on integrating satellite imagery data into quantitative investment strategies.",
    authors: ["Dr. Alice Johnson"],
    date: "2025-10-20",
    category: "Alternative Data",
    url: "https://example.com/paper2.pdf",
  },
  {
    id: "3",
    title: "High-Frequency Trading Algorithms",
    summary: "Analysis of optimal execution strategies in volatile market conditions.",
    authors: ["Dr. Bob Wilson", "Dr. Carol Brown"],
    date: "2025-09-10",
    category: "Execution",
    url: "https://example.com/paper3.pdf",
  },
];

export const researchService = {
  /**
   * Get list of research papers
   */
  getPapers: async (filters = {}) => {
    if (ENABLE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => resolve(mockResearchPapers), 500);
      });
    }

    try {
      const response = await api.get(API_ENDPOINTS.RESEARCH.PAPERS, {
        params: filters,
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  /**
   * Get research paper details
   */
  getPaperDetails: async (id) => {
    if (ENABLE_MOCK_DATA) {
      return new Promise((resolve) => {
        const paper = mockResearchPapers.find((p) => p.id === id);
        setTimeout(() => resolve(paper || null), 500);
      });
    }

    try {
      const response = await api.get(API_ENDPOINTS.RESEARCH.DETAILS(id));
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};
