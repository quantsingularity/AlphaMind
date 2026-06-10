import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy all /api/* and /health requests to the FastAPI backend in dev.
      // Uses VITE_DEV_PROXY_TARGET (separate from VITE_API_BASE_URL, which sets
      // the axios baseURL) so the two concerns don't collide.
      "/api": {
        target: process.env.VITE_DEV_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: process.env.VITE_DEV_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
