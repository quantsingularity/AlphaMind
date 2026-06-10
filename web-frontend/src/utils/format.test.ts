import { describe, expect, it } from "vitest";
import {
  formatCurrency,
  formatDate,
  formatLargeNumber,
  formatNumber,
  formatPercentage,
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

  describe("formatLargeNumber", () => {
    it("formats positive billions/millions/thousands", () => {
      expect(formatLargeNumber(1_500_000_000)).toBe("1.50B");
      expect(formatLargeNumber(2_300_000)).toBe("2.30M");
      expect(formatLargeNumber(4_500)).toBe("4.50K");
    });

    it("formats negative large numbers with the correct suffix", () => {
      expect(formatLargeNumber(-1_500_000_000)).toBe("-1.50B");
      expect(formatLargeNumber(-2_300_000)).toBe("-2.30M");
      expect(formatLargeNumber(-4_500)).toBe("-4.50K");
    });

    it("formats small numbers without a suffix", () => {
      expect(formatLargeNumber(42)).toBe("42.00");
      expect(formatLargeNumber(-42)).toBe("-42.00");
    });
  });
});
