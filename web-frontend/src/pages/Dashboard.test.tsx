import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Dashboard } from "./Dashboard";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe("Dashboard Page", () => {
  it("renders dashboard header", async () => {
    render(<Dashboard />, { wrapper });
    expect(await screen.findByText("Trading Dashboard")).toBeInTheDocument();
  });

  it("displays portfolio metrics cards", async () => {
    render(<Dashboard />, { wrapper });
    expect(await screen.findByText("Total Value")).toBeInTheDocument();
    expect(await screen.findByText("Daily P&L")).toBeInTheDocument();
    expect(await screen.findByText("Cash")).toBeInTheDocument();
  });

  it("displays positions table", async () => {
    render(<Dashboard />, { wrapper });
    expect(await screen.findByText("Current Positions")).toBeInTheDocument();
  });
});
