# Troubleshooting

## "Network error" in the clients while the backend is running

This almost always means the client is calling a host or port it cannot reach, not that the backend is down.

1. Confirm the backend is up: open `http://localhost:8000/docs`. If that fails, the problem is connectivity to the backend host itself (common on WSL or behind a browser VPN).
2. Web dashboard: it proxies `/api` to `http://localhost:8000`. Make sure the backend is on 8000.
3. Mobile (Expo web) on WSL or behind a VPN: open the app at the LAN address Expo prints (for example `http://192.168.x.x:8081`) instead of `localhost:8081`. The app then calls the backend at the same host on port 8000, bypassing localhost forwarding and VPN interception. Or set `EXPO_PUBLIC_API_BASE_URL=http://<host>:8000`.
4. Cross-origin: the backend allows `localhost`, `127.0.0.1`, and private LAN IPs by default. If you serve a client from another origin, add it to `CORS_ORIGINS`.

If the backend is genuinely unreachable, the clients fall back to a demo session so you can still navigate; data screens then show empty or placeholder states.

## The app starts in the dashboard instead of the homepage

Offline demo sessions are not restored on startup, so a fresh launch opens on the homepage. A real backend session does persist. If you want to force the homepage, sign out (Settings → Sign Out).

## Sign Out does nothing (web)

This was a known issue caused by `Alert.alert` being a no-op on React Native Web and has been fixed; sign-out now uses the browser confirm dialog on web. If you still see it, make sure you are running the current `mobile-frontend/screens/SettingsScreen.js`.

## Backend refuses to start: SECRET_KEY error

In `production`, `prod`, or `staging`, the app requires a strong `SECRET_KEY` (at least 32 characters). Set one:

```bash
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

In development the placeholder is allowed.

## `ModuleNotFoundError: No module named 'aiosqlite'`

Install the async SQLite driver (it is in `requirements.txt`): `pip install aiosqlite`.

## Backend tests fail to collect (TensorFlow)

Some model tests import TensorFlow. If it is not installed, run the rest of the suite explicitly:

```bash
pytest tests/test_api.py tests/test_basic_api.py \
       tests/test_frontend_contracts.py tests/test_security.py \
       tests/test_market_data_service.py
```

## Market data shows `source: synthetic`

That means the live connectors were not reachable and the synthetic fallback served the data. Check network access; set `POLYGON_API_KEY` to enable Polygon. Yahoo Finance needs no key.

## Port already in use

The backend uses 8000, the web dashboard 3000, and Expo 8081. Stop the conflicting process or change the port (for example `npm run dev -- --port 3001`).
