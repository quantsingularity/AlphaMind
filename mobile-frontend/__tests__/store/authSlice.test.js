import authReducer, {
  clearError,
  loginUser,
  logoutUser,
  setUser,
} from "../../store/slices/authSlice";

describe("authSlice", () => {
  const initialState = {
    user: null,
    isAuthenticated: false,
    loading: false,
    error: null,
  };

  it("should return the initial state", () => {
    expect(authReducer(undefined, { type: "unknown" })).toEqual(initialState);
  });

  it("should handle clearError", () => {
    const previousState = { ...initialState, error: "Some error" };
    expect(authReducer(previousState, clearError())).toEqual(initialState);
  });

  it("should handle setUser", () => {
    const user = { id: 1, email: "test@example.com", name: "Test User" };
    const nextState = authReducer(initialState, setUser(user));
    expect(nextState.user).toEqual(user);
    expect(nextState.isAuthenticated).toBe(true);
  });

  it("should handle setUser with null (logout)", () => {
    const loggedInState = {
      ...initialState,
      user: { id: 1, email: "test@example.com" },
      isAuthenticated: true,
    };
    const nextState = authReducer(loggedInState, setUser(null));
    expect(nextState.user).toBeNull();
    expect(nextState.isAuthenticated).toBe(false);
  });

  it("should handle loginUser.pending", () => {
    const nextState = authReducer(initialState, {
      type: loginUser.pending.type,
    });
    expect(nextState.loading).toBe(true);
    expect(nextState.error).toBeNull();
  });

  it("should handle loginUser.fulfilled", () => {
    const user = { id: 1, email: "test@example.com", name: "Test User" };
    const nextState = authReducer(initialState, {
      type: loginUser.fulfilled.type,
      payload: { user },
    });
    expect(nextState.loading).toBe(false);
    expect(nextState.isAuthenticated).toBe(true);
    expect(nextState.user).toEqual(user);
    expect(nextState.error).toBeNull();
  });

  it("should handle loginUser.rejected", () => {
    const nextState = authReducer(initialState, {
      type: loginUser.rejected.type,
      payload: "Login failed",
    });
    expect(nextState.loading).toBe(false);
    expect(nextState.error).toBe("Login failed");
    expect(nextState.isAuthenticated).toBe(false);
  });

  it("should handle logoutUser.pending", () => {
    const loggedInState = {
      ...initialState,
      isAuthenticated: true,
      user: { id: 1 },
    };
    const nextState = authReducer(loggedInState, {
      type: logoutUser.pending.type,
    });
    expect(nextState.loading).toBe(true);
  });

  it("should handle logoutUser.fulfilled", () => {
    const loggedInState = {
      ...initialState,
      isAuthenticated: true,
      user: { id: 1 },
    };
    const nextState = authReducer(loggedInState, {
      type: logoutUser.fulfilled.type,
    });
    expect(nextState.user).toBeNull();
    expect(nextState.isAuthenticated).toBe(false);
    expect(nextState.loading).toBe(false);
    expect(nextState.error).toBeNull();
  });

  it("should handle logoutUser.rejected and still clear auth", () => {
    const loggedInState = {
      ...initialState,
      isAuthenticated: true,
      user: { id: 1 },
    };
    const nextState = authReducer(loggedInState, {
      type: logoutUser.rejected.type,
      payload: "Logout API error",
    });
    expect(nextState.user).toBeNull();
    expect(nextState.isAuthenticated).toBe(false);
    expect(nextState.loading).toBe(false);
  });
});
