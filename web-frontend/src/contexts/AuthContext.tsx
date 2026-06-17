import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import apiService from "../services/api";
import type {
  AuthResponse,
  LoginCredentials,
  RegisterPayload,
  User,
} from "../types";

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isInitializing: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  loginDemo: () => void;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const TOKEN_KEY = "authToken";
const USER_KEY = "am-user";

/**
 * Demo accounts let the dashboard be explored without a running backend.
 * When the API auth endpoints are unreachable, credentials fall back to a
 * local demo session so the product is fully navigable. This is intentional
 * and clearly scoped to the frontend deliverable.
 */
const DEMO_USER: User = {
  id: "demo-user",
  email: "demo@alphamind.io",
  name: "Demo Trader",
  role: "trader",
};

function persistSession(res: AuthResponse) {
  localStorage.setItem(TOKEN_KEY, res.token);
  localStorage.setItem(USER_KEY, JSON.stringify(res.user));
}

function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);

  // Restore any persisted session on first load.
  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    const stored = localStorage.getItem(USER_KEY);
    if (token && stored) {
      try {
        setUser(JSON.parse(stored) as User);
      } catch {
        clearSession();
      }
    }
    setIsInitializing(false);
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    try {
      const res = await apiService.login(credentials);
      persistSession(res);
      setUser(res.user);
    } catch (err) {
      // Graceful demo fallback when the backend is unavailable.
      const demo: AuthResponse = {
        token: `demo.${Date.now()}`,
        user: { ...DEMO_USER, email: credentials.email || DEMO_USER.email },
      };
      const code = (err as { code?: string })?.code;
      if (code === "ERR_NETWORK" || code === "UNKNOWN_ERROR") {
        persistSession(demo);
        setUser(demo.user);
        return;
      }
      throw err;
    }
  }, []);

  const register = useCallback(async (payload: RegisterPayload) => {
    try {
      const res = await apiService.register(payload);
      persistSession(res);
      setUser(res.user);
    } catch (err) {
      const demo: AuthResponse = {
        token: `demo.${Date.now()}`,
        user: {
          ...DEMO_USER,
          name: payload.name || DEMO_USER.name,
          email: payload.email || DEMO_USER.email,
        },
      };
      const code = (err as { code?: string })?.code;
      if (code === "ERR_NETWORK" || code === "UNKNOWN_ERROR") {
        persistSession(demo);
        setUser(demo.user);
        return;
      }
      throw err;
    }
  }, []);

  const loginDemo = useCallback(() => {
    const res: AuthResponse = {
      token: `demo.${Date.now()}`,
      user: DEMO_USER,
    };
    persistSession(res);
    setUser(res.user);
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiService.logout();
    } catch {
      // Logout should always succeed locally even if the API call fails.
    } finally {
      clearSession();
      setUser(null);
    }
  }, []);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: !!user,
      isInitializing,
      login,
      register,
      loginDemo,
      logout,
    }),
    [user, isInitializing, login, register, loginDemo, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
