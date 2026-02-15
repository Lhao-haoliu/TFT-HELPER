# TFT Helper Monorepo

This repository contains:

- `backend/`: FastAPI backend
- `miniprogram/`: WeChat Mini Program (native)
- `docs/`: API docs and project docs

## Backend

Run from `backend/`:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --app-dir src --reload
```

Or run from monorepo root:

```powershell
python -m uvicorn app.main:app --app-dir backend/src --reload
```

If you see `ModuleNotFoundError: No module named 'src'`, your startup path is wrong.
Use one of the two commands above.

Health check:

- `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`

### Cache prewarm and scheduler

- Startup prewarm: enabled by default (logs under `app.cache`)
- Daily refresh time: server local `00:00`
- Optional snapshot persistence:
  - `CACHE_PERSIST_ENABLED=1` (default)
  - `CACHE_PERSIST_PATH` (default `backend/.cache/cache_snapshot.json`)
- For scheduler testing (instead of waiting until midnight):
  - set `CACHE_REFRESH_INTERVAL_SECONDS=60`

API reference:

- `docs/API.md`
- Local asset sync scripts: `backend/tools/sync_champion_assets.py`, `backend/tools/sync_augment_assets.py`

## Mini Program

1. Open WeChat DevTools
2. Import `miniprogram/` as the project root
3. Build and run the home page (`pages/index/index`)

### Local backend during development

1. In DevTools, enable:
   - `Details` -> `Local Settings` -> `Do not verify valid domain names, web-view (business domain names), TLS version, and HTTPS certificates`
2. Configure backend base URL in:
   - `miniprogram/config.js`
3. Default value is:
   - `http://127.0.0.1:8000`

`pages/battle/battle` contains a simple `getChampions({limit: 3})` integration for API smoke testing.
