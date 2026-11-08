import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/knowledge-base": "http://127.0.0.1:8000",
      "/ingest": "http://127.0.0.1:8000",
      "/query": "http://127.0.0.1:8000",
      "/evaluation": "http://127.0.0.1:8000",
    },
  },
});
