import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { authService } from "../../services/authService";

// Async thunks
export const loginUser = createAsyncThunk("auth/login", async ({ email, password }, thunkAPI) => {
  try {
    const response = await authService.login(email, password);
    return response;
  } catch (error) {
    return thunkAPI.rejectWithValue(error.message || "Login failed");
  }
});

export const registerUser = createAsyncThunk("auth/register", async (userData, thunkAPI) => {
  try {
    const response = await authService.register(userData);
    return response;
  } catch (error) {
    return thunkAPI.rejectWithValue(error.message || "Registration failed");
  }
});

export const logoutUser = createAsyncThunk("auth/logout", async (_, thunkAPI) => {
  try {
    await authService.logout();
    return null;
  } catch (error) {
    return thunkAPI.rejectWithValue(error.message || "Logout failed");
  }
});

export const checkAuth = createAsyncThunk("auth/checkAuth", async (_, thunkAPI) => {
  try {
    const isAuthenticated = await authService.isAuthenticated();
    if (isAuthenticated) {
      const userData = await authService.getUserData();
      return { isAuthenticated: true, user: userData };
    }
    return { isAuthenticated: false, user: null };
  } catch (error) {
    return thunkAPI.rejectWithValue(error.message || "Auth check failed");
  }
});

const initialState = {
  user: null,
  isAuthenticated: false,
  loading: false,
  error: null,
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setUser: (state, action) => {
      state.user = action.payload;
      state.isAuthenticated = !!action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(loginUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.error = null;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Register
      .addCase(registerUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.error = null;
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Logout
      .addCase(logoutUser.fulfilled, (state) => {
        state.user = null;
        state.isAuthenticated = false;
        state.loading = false;
        state.error = null;
      })
      // Check Auth
      .addCase(checkAuth.pending, (state) => {
        state.loading = true;
      })
      .addCase(checkAuth.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = action.payload.isAuthenticated;
        state.user = action.payload.user;
      })
      .addCase(checkAuth.rejected, (state) => {
        state.loading = false;
        state.isAuthenticated = false;
        state.user = null;
      });
  },
});

export const { clearError, setUser } = authSlice.actions;
export default authSlice.reducer;
