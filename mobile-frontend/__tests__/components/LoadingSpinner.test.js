import { render, screen } from "@testing-library/react-native";
import { Provider as PaperProvider } from "react-native-paper";
import LoadingSpinner from "../../components/LoadingSpinner";

const renderWithPaper = (component) =>
  render(<PaperProvider>{component}</PaperProvider>);

describe("LoadingSpinner", () => {
  it("renders with default message", () => {
    renderWithPaper(<LoadingSpinner />);
    expect(screen.getByText("Loading...")).toBeTruthy();
  });

  it("renders with custom message", () => {
    renderWithPaper(<LoadingSpinner message="Fetching data..." />);
    expect(screen.getByText("Fetching data...")).toBeTruthy();
  });

  it("does not render message when message is empty string", () => {
    renderWithPaper(<LoadingSpinner message="" />);
    expect(screen.queryByText("Loading...")).toBeNull();
  });

  it("has correct accessibility role", () => {
    renderWithPaper(<LoadingSpinner message="Loading..." />);
    expect(screen.getByRole("progressbar")).toBeTruthy();
  });
});
