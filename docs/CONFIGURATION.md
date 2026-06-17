# Configuration

AlphaMind is configured through environment variables. Defaults let the whole stack run locally with no setup.

## Backend

| Variable                  | Default                         | Purpose                                                                                                                                     |
| :------------------------ | :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------ |
| `SECRET_KEY`              | dev placeholder                 | JWT signing key. Required (>= 32 chars) when `ENVIRONMENT` is `production`, `prod`, or `staging`; the app refuses to start otherwise.       |
| `ENVIRONMENT` / `APP_ENV` | `development`                   | Environment name. Controls the secret hardening above.                                                                                      |
| `JWT_EXPIRATION_HOURS`    | `24`                            | Token lifetime.                                                                                                                             |
| `CORS_ORIGINS`            | `localhost:3000,3001,5173,8081` | Comma-separated allow-list. In addition, any `localhost`, `127.0.0.1`, or private LAN IP on any port is allowed in development via a regex. |
| `DATABASE_URL`            | SQLite (`alphamind.db`)         | SQLAlchemy URL. Use a MySQL or PostgreSQL URL for those backends.                                                                           |
| `POLYGON_API_KEY`         | unset                           | Enables the Polygon market-data connector. Yahoo Finance needs no key.                                                                      |

Set them inline or via a `.env` file in `code/backend`:

```bash
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export ENVIRONMENT=production
python -m main
```

## Web frontend

The dev server runs on port 3000 and proxies `/api` and `/health` to the backend. The proxy target defaults to `http://localhost:8000` and can be overridden with `VITE_DEV_PROXY_TARGET`:

```bash
VITE_DEV_PROXY_TARGET=http://192.168.82.76:8000 npm run dev
```

In production, serve the built app behind a reverse proxy that forwards `/api` to the backend.

## Mobile frontend

The API base URL is resolved in `constants/config.js` in this order:

1. `extra.apiBaseUrl` in `app.json` (leave unset unless you want to force a URL).
2. `EXPO_PUBLIC_API_BASE_URL` environment variable.
3. `API_BASE_URL` environment variable.
4. A platform default: on web, the host the app was loaded from on port 8000; on the Android emulator, `http://10.0.2.2:8000`; otherwise `http://localhost:8000`.

```bash
# Force a specific backend (handy on WSL or behind a VPN)
EXPO_PUBLIC_API_BASE_URL=http://192.168.82.76:8000 npm start
```

Other settings (theme, notifications, display currency) are stored on-device with AsyncStorage and persist across launches. Offline demo sessions are intentionally not restored on startup, so the app always opens on the homepage unless a real backend session exists.

## Notifications and theme

Theme (light, dark, system), notification toggles, and display currency live in the mobile Settings screen and are persisted automatically. A quick light/dark toggle is also available in the header and on the homepage and dashboard.
