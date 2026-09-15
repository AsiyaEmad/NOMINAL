# NOMINAL Dashboard

Run the FastAPI service on port `8000`, then start the dashboard:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000` during development. For a separately hosted API, set `VITE_API_BASE_URL` in a local `.env` file.

The dashboard reads only these existing endpoints:

- `GET /api/metrics/summary`
- `GET /api/traces`
- `GET /api/traces/{request_id}`
- `GET /api/benchmarks`
- `POST /api/benchmarks/run`
