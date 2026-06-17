import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Navigate,
  Route,
  BrowserRouter as Router,
  Routes,
} from "react-router-dom";
import "./App.css";
import { AppLayout } from "./components/AppLayout";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { PublicLayout } from "./components/PublicLayout";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { About } from "./pages/About";
import { AlternativeData } from "./pages/AlternativeData";
import { Backtest } from "./pages/Backtest";
import { Dashboard } from "./pages/Dashboard";
import { Documentation } from "./pages/Documentation";
import { Home } from "./pages/Home";
import { NotFound } from "./pages/NotFound";
import { Portfolio } from "./pages/Portfolio";
import { Research } from "./pages/Research";
import { RiskManagement } from "./pages/RiskManagement";
import { Settings } from "./pages/Settings";
import { SignIn } from "./pages/SignIn";
import { SignUp } from "./pages/SignUp";
import { Strategies } from "./pages/Strategies";
import { Trading } from "./pages/Trading";
import { MarketData } from "./pages/MarketData";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5000,
    },
  },
});

function GuestRoute({ children }: { children: React.ReactElement }) {
  const { isAuthenticated, isInitializing } = useAuth();
  if (isInitializing) return null;
  if (isAuthenticated) return <Navigate to="/dashboard" replace />;
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      <Route element={<PublicLayout />}>
        <Route index element={<Home />} />
        <Route path="/documentation" element={<Documentation />} />
        <Route path="/research" element={<Research />} />
        <Route path="/about" element={<About />} />
      </Route>

      <Route
        path="/signin"
        element={
          <GuestRoute>
            <SignIn />
          </GuestRoute>
        }
      />
      <Route
        path="/signup"
        element={
          <GuestRoute>
            <SignUp />
          </GuestRoute>
        }
      />

      <Route element={<ProtectedRoute />}>
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/risk" element={<RiskManagement />} />
          <Route path="/alternative-data" element={<AlternativeData />} />
          <Route path="/market-data" element={<MarketData />} />
          <Route path="/trading" element={<Trading />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <Router>
            <AppRoutes />
          </Router>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
