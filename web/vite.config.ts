import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Порт зафиксирован: CORS в praxis/api.py разрешает только 5173.
    port: 5173,
    host: true,
    // Через прокси фронт и бэкенд выглядят одним origin — ни CORS, ни preflight
    // на PUT и POST. Прод собирается тем же кодом: там SPA отдаёт сам FastAPI.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
