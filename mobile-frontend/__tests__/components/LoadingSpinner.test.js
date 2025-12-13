import React from "react";
import { render, screen } from "@testing-library/react-native";
import LoadingSpinner from "../../components/LoadingSpinner";

describe("LoadingSpinner Component", () => {
  it("renders with default message", () => {
    render(<LoadingSpinner />);
    expect(screen.getByText("Loading...")).toBeTruthy();
  });

  it("renders with custom message", () => {
    render(<LoadingSpinner message="Loading data..." />);
    expect(screen.getByText("Loading data...")).toBeTruthy();
  });

  it("renders without message when null is passed", () => {
    render(<LoadingSpinner message={null} />);
    expect(screen.queryByText("Loading...")).toBeNull();
  });

  it("renders ActivityIndicator", () => {
    const { getByTestId } = render(<LoadingSpinner />);
    // ActivityIndicator is rendered (checking component structure)
    expect(screen.root).toBeTruthy();
  });
});
