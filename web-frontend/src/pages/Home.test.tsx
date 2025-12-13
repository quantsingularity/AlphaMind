import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { Home } from "./Home";

describe("Home Page", () => {
  it("renders hero section", () => {
    render(
      <BrowserRouter>
        <Home />
      </BrowserRouter>,
    );

    expect(screen.getByText(/Next-Gen/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Quantitative AI Trading System/i),
    ).toBeInTheDocument();
  });

  it("renders features section", () => {
    render(
      <BrowserRouter>
        <Home />
      </BrowserRouter>,
    );

    expect(screen.getByText("Key Features")).toBeInTheDocument();
    expect(
      screen.getByText("Alternative Data Integration"),
    ).toBeInTheDocument();
    expect(screen.getByText("Quantitative Research")).toBeInTheDocument();
  });

  it("renders performance metrics table", () => {
    render(
      <BrowserRouter>
        <Home />
      </BrowserRouter>,
    );

    expect(screen.getByText("Performance Metrics")).toBeInTheDocument();
    expect(screen.getByText("Sharpe Ratio")).toBeInTheDocument();
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
  });

  it("renders getting started section", () => {
    render(
      <BrowserRouter>
        <Home />
      </BrowserRouter>,
    );

    expect(screen.getByText("Getting Started")).toBeInTheDocument();
  });
});
