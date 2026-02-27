import authReducer, {
  loginUser,
  registerUser,
  logoutUser,
  checkAuth,
  clearError,
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
  });

  it("should handle loginUser.rejected", () => {
    const nextState = authReducer(initialState, {
      type: loginUser.rejected.type,
      payload: "Login failed",
    });
    expect(nextState.loading).toBe(false);
    expect(nextState.error).toBe("Login failed");
  });
});
