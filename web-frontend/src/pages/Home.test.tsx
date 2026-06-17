import { render, screen } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { Home } from "./Home";

const renderHome = () =>
  render(
    <BrowserRouter>
      <Home />
    </BrowserRouter>,
  );

describe("Home Page", () => {
  it("renders hero section", () => {
    renderHome();
    expect(screen.getByText(/Where alternative data/i)).toBeInTheDocument();
    expect(screen.getByText(/durable alpha/i)).toBeInTheDocument();
  });

  it("renders capabilities section", () => {
    renderHome();
    expect(screen.getByText("Four engines, one edge")).toBeInTheDocument();
    expect(screen.getByText("Alternative Data")).toBeInTheDocument();
    expect(screen.getByText("Quant Research")).toBeInTheDocument();
    expect(screen.getByText("Execution Engine")).toBeInTheDocument();
  });

  it("renders backtested performance table", () => {
    renderHome();
    expect(screen.getByText("Backtested performance")).toBeInTheDocument();
    expect(screen.getByText("Sharpe")).toBeInTheDocument();
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
  });

  it("renders a primary call to action", () => {
    renderHome();
    expect(screen.getByText(/Get started free/i)).toBeInTheDocument();
  });
});
