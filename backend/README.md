# TFT Helper - FastAPI Scaffold

Minimal runnable FastAPI scaffold with:
- `GET /health`
- `GET /api/champions/stats`
- `GET /api/champions/mapping`
- `GET /api/champions`
- `GET /api/champions/{championId}/augments`
- `GET /api/champions/{championId}/augment-combos`
- `GET /api/augments/mapping`
- `POST /api/battle/recommend`
- Swagger UI at `/docs`
- One-command dev startup scripts

## 1. Runtime

- Python 3.14 (validated on this machine)
- Python 3.11+ is also supported by this project setup

## 2. Create venv and install dependencies

### Windows (PowerShell, Python 3.14)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Start the service

Option A (direct):

```bash
uvicorn app.main:app --app-dir src --reload
```

From monorepo root, use:

```bash
uvicorn app.main:app --app-dir backend/src --reload
```

Option B (Make):

```bash
make dev
```

Option C (scripts):

Windows:

```bat
scripts\dev.bat
```

macOS / Linux:

```bash
bash scripts/dev.sh
```

## 4. Verify

- Health endpoint: `http://127.0.0.1:8000/health`
- Champion stats endpoint: `http://127.0.0.1:8000/api/champions/stats`
- Champion mapping endpoint: `http://127.0.0.1:8000/api/champions/mapping`
- Champion merged endpoint: `http://127.0.0.1:8000/api/champions`
- Champion augment ranking endpoint: `http://127.0.0.1:8000/api/champions/887/augments`
- Augment mapping endpoint: `http://127.0.0.1:8000/api/augments/mapping`
- Swagger UI: `http://127.0.0.1:8000/docs`

Query params for `/api/champions`:
- `q`: fuzzy match on `name_cn` and `fullname_cn`
- `sort`: `winRate` or `pickRate`
- `order`: `desc` or `asc`
- `limit`: default `50`, max `200`
- `offset`: default `0`

Expected health response:

```json
{"ok": true}
```

## 5. Cache prewarm and refresh

- Startup prewarm runs automatically and logs progress (`app.cache` logger).
- Scheduled refresh runs every day at local server `00:00`.
- Optional snapshot file:
  - `CACHE_PERSIST_ENABLED=1` (default)
  - `CACHE_PERSIST_PATH` (default `backend/.cache/cache_snapshot.json`)
- For quick cron testing:
  - set `CACHE_REFRESH_INTERVAL_SECONDS=60`

## 6. Sync local static assets

Run from `backend/`:

```bash
python tools/sync_champion_assets.py
python tools/sync_augment_assets.py
```

Generated output:

- `static/champions/{championId}.{ext}`
- `static/augments/{augmentId}.{ext}`
- `static/champions/manifest.json`
- `static/augments/manifest.json`

The API returns relative local static paths in:

- `/api/champions[*].iconUrl` -> `/static/champions/...`
- `/api/champions/{id}/augments[*].iconUrl` -> `/static/augments/...`

Static responses under `/static/*` include:

- `Cache-Control: public, max-age=31536000`

## 7. OCR endpoint (MVP)

New API:

- `POST /api/ocr/augments-names` (multipart file field: `image`)

Required base deps are already in `requirements.txt`:

- `opencv-python-headless`, `numpy`, `rapidfuzz`, `python-multipart`

OCR engine options (install at least one):

1. `paddleocr` (preferred)
2. `easyocr`
3. `pytesseract` + system Tesseract OCR

Debug utility:

```bash
python tools/debug_ocr_cards.py path/to/screenshot.jpg
```

This writes overlay/card/title debug images to `backend/.cache/ocr_debug/`.

## 8. OCR regression tests

Run from repo root:

```bash
python -m pytest backend/tests -q
```

Dataset location:

- `backend/tests/assets/ocr_samples/manifest.json`
- `backend/tests/assets/ocr_samples/*.jpg`

Current checks:

- OCR result order must be `left/middle/right`
- each sample must hit at least `2/3` expected names
- battle recommendation rules (Round2/3/4 and tie-breaks) are unit-tested
