import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev: run `python -m gem_studio --dev --no-open --port 8765` and `npm run dev`.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  build: {
    outDir: "../gem_studio/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
})
