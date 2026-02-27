import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { portfolioService } from "../../services/portfolioService";

// Async thunks
export const fetchPortfolio = createAsyncThunk(
  "portfolio/fetch",
  async (_, thunkAPI) => {
    try {
      const data = await portfolioService.getPortfolio();
      return data;
    } catch (error) {
      return thunkAPI.rejectWithValue(
        error.message || "Failed to fetch portfolio",
      );
    }
  },
);

export const fetchPerformance = createAsyncThunk(
  "portfolio/fetchPerformance",
  async (timeframe, thunkAPI) => {
    try {
      const data = await portfolioService.getPerformance(timeframe);
      return data;
    } catch (error) {
      return thunkAPI.rejectWithValue(
        error.message || "Failed to fetch performance",
      );
    }
  },
);

export const fetchHoldings = createAsyncThunk(
  "portfolio/fetchHoldings",
  async (_, thunkAPI) => {
    try {
      const data = await portfolioService.getHoldings();
      return data;
    } catch (error) {
      return thunkAPI.rejectWithValue(
        error.message || "Failed to fetch holdings",
      );
    }
  },
);

const initialState = {
  data: null,
  performance: [],
  holdings: [],
  loading: false,
  performanceLoading: false,
  holdingsLoading: false,
  error: null,
  lastUpdated: null,
};

const portfolioSlice = createSlice({
  name: "portfolio",
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    resetPortfolio: (state) => {
      return initialState;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch Portfolio
      .addCase(fetchPortfolio.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPortfolio.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
        state.lastUpdated = new Date().toISOString();
        state.error = null;
      })
      .addCase(fetchPortfolio.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      // Fetch Performance
      .addCase(fetchPerformance.pending, (state) => {
        state.performanceLoading = true;
      })
      .addCase(fetchPerformance.fulfilled, (state, action) => {
        state.performanceLoading = false;
        state.performance = action.payload;
      })
      .addCase(fetchPerformance.rejected, (state, action) => {
        state.performanceLoading = false;
        state.error = action.payload;
      })
      // Fetch Holdings
      .addCase(fetchHoldings.pending, (state) => {
        state.holdingsLoading = true;
      })
      .addCase(fetchHoldings.fulfilled, (state, action) => {
        state.holdingsLoading = false;
        state.holdings = action.payload;
      })
      .addCase(fetchHoldings.rejected, (state, action) => {
        state.holdingsLoading = false;
        state.error = action.payload;
      });
  },
});

export const { clearError, resetPortfolio } = portfolioSlice.actions;
export default portfolioSlice.reducer;
