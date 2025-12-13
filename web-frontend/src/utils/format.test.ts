import { describe, it, expect } from "vitest";
import {
  formatCurrency,
  formatPercentage,
  formatNumber,
  formatDate,
} from "./format";

describe("Format Utilities", () => {
  describe("formatCurrency", () => {
    it("formats positive numbers correctly", () => {
      expect(formatCurrency(1234.56)).toBe("$1,234.56");
    });

    it("formats negative numbers correctly", () => {
      expect(formatCurrency(-1234.56)).toBe("-$1,234.56");
    });

    it("formats zero correctly", () => {
      expect(formatCurrency(0)).toBe("$0.00");
    });
  });

  describe("formatPercentage", () => {
    it("formats decimal to percentage", () => {
      expect(formatPercentage(0.1234)).toBe("12.34%");
    });

    it("handles custom decimal places", () => {
      expect(formatPercentage(0.12345, 3)).toBe("12.345%");
    });
  });

  describe("formatNumber", () => {
    it("formats numbers with thousand separators", () => {
      expect(formatNumber(1234567.89)).toBe("1,234,567.89");
    });
  });

  describe("formatDate", () => {
    it("formats date string correctly", () => {
      const result = formatDate("2025-01-15");
      expect(result).toMatch(/Jan/);
    });
  });
});
