import React from "react";
import { render, screen } from "@testing-library/react-native";
import { Provider as PaperProvider } from "react-native-paper";
import KPICard from "../../components/KPICard";

const renderWithProvider = (component) => {
  return render(<PaperProvider>{component}</PaperProvider>);
};

describe("KPICard Component", () => {
  const defaultProps = {
    title: "Portfolio Value",
    value: "$1,250,345.67",
    change: "+1.2%",
    changeColor: "green",
    icon: "chart-line",
  };

  it("renders all props correctly", () => {
    renderWithProvider(<KPICard {...defaultProps} />);
    
    expect(screen.getByText("Portfolio Value")).toBeTruthy();
    expect(screen.getByText("$1,250,345.67")).toBeTruthy();
    expect(screen.getByText("+1.2%")).toBeTruthy();
  });

  it("shows loading state", () => {
    renderWithProvider(<KPICard {...defaultProps} isLoading={true} />);
    
    expect(screen.getByText("Portfolio Value")).toBeTruthy();
    expect(screen.getByText("...")).toBeTruthy();
  });

  it("renders without change value", () => {
    const propsWithoutChange = { ...defaultProps, change: undefined };
    renderWithProvider(<KPICard {...propsWithoutChange} />);
    
    expect(screen.getByText("Portfolio Value")).toBeTruthy();
    expect(screen.getByText("$1,250,345.67")).toBeTruthy();
    expect(screen.queryByText("+1.2%")).toBeNull();
  });

  it("renders without icon", () => {
    const propsWithoutIcon = { ...defaultProps, icon: undefined };
    renderWithProvider(<KPICard {...propsWithoutIcon} />);
    
    expect(screen.getByText("Portfolio Value")).toBeTruthy();
  });
});
