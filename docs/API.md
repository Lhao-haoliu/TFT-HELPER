# TFT Helper API Docs

## Base URL

- Local: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

## Common Notes

- Response type: `application/json`
- Some endpoints may return `503`: `{"detail":"..."}`
- Pagination params are only used by `/api/champions`
- Static assets are served by backend under `/static/...`

## 1) Health

- Method: `GET`
- Path: `/health`

Example:

```json
{"ok": true}
```

## 2) Champion Raw Stats

- Method: `GET`
- Path: `/api/champions/stats`
- Description: normalized champion stats from dtodo

Response item fields:

- `championId` string
- `winRate` number (0~1)
- `pickRate` number (0~1)
- `matches` number|null

## 3) Champion CN Mapping

- Method: `GET`
- Path: `/api/champions/mapping`
- Description: `championId -> Chinese name`

Response:

```json
{
  "887": {
    "name_cn": "格温",
    "fullname_cn": "灵罗娃娃 格温"
  }
}
```

## 4) Champion Merged List

- Method: `GET`
- Path: `/api/champions`
- Description: merged stats + Chinese mapping + local icon path

Query params:

- `q` string, fuzzy match by `name_cn/fullname_cn`
- `sort` string, `winRate|pickRate`, default `winRate`
- `order` string, `desc|asc`, default `desc`
- `limit` int, default `50`, range `1~200`
- `offset` int, default `0`, min `0`

Response item fields:

- `championId` string
- `name_cn` string
- `fullname_cn` string
- `winRate` number
- `pickRate` number
- `iconUrl` string, example `/static/champions/887.png`

Example:

```json
[
  {
    "championId": "887",
    "name_cn": "格温",
    "fullname_cn": "灵罗娃娃 格温",
    "winRate": 0.5851,
    "pickRate": 0.0033,
    "iconUrl": "/static/champions/887.png"
  }
]
```

## 5) Champion Augment Ranking

- Method: `GET`
- Path: `/api/champions/{championId}/augments`
- Description: augment ranking for one champion

Path params:

- `championId` string|number

Query params:

- `sort` string, `winRate|pickRate`, default `winRate`
- `order` string, `desc|asc`, default `desc`

Response item fields:

- `augmentId` number
- `name_cn` string
- `tier` string
- `winRate` number (0~1)
- `pickRate` number (0~1)
- `games` number|null
- `iconUrl` string, example `/static/augments/1420.png`
- `rarity` string, `prismatic|gold|silver|unknown`

Example:

```json
[
  {
    "augmentId": 1420,
    "name_cn": "咏叹奏鸣",
    "tier": "T5",
    "winRate": 0.6737,
    "pickRate": 0.0036,
    "games": 610,
    "iconUrl": "/static/augments/1420.png",
    "rarity": "gold"
  }
]
```

## 6) Champion Augment Combos

- Method: `GET`
- Path: `/api/champions/{championId}/augment-combos`
- Description: recommended augment combos

Path params:

- `championId` string|number

Response fields:

- `championId` number|string
- `version` string|null
- `combos` array

`combos` item fields:

- `rank` number
- `augments` array

`augments` item fields:

- `augmentId` number
- `name_cn` string (fallback format `#ID` if missing)

Example:

```json
{
  "championId": 887,
  "version": "16.3",
  "combos": [
    {
      "rank": 1,
      "augments": [
        {"augmentId": 1071, "name_cn": "xxx"},
        {"augmentId": 1129, "name_cn": "xxx"},
        {"augmentId": 1346, "name_cn": "xxx"}
      ]
    }
  ]
}
```

## 7) Augment Mapping

- Method: `GET`
- Path: `/api/augments/mapping`
- Description: augment dictionary from CommunityDragon latest Chinese source

Response map value fields:

- `augmentId` string
- `name_cn` string
- `desc_cn` string|null
- `icon` string|null (remote source icon URL)
- `rarity` string, `prismatic|gold|silver|unknown`

## 8) Static Assets

Mounted routes:

- Champion icons: `/static/champions/{championId}.{ext}`
- Augment icons: `/static/augments/{augmentId}.{ext}`

## 9) Asset Sync Scripts

Run from `backend/`:

```bash
python tools/sync_champion_assets.py
python tools/sync_augment_assets.py
```

Generated files:

- `backend/static/champions/*`
- `backend/static/augments/*`
- `backend/static/champions/manifest.json`
- `backend/static/augments/manifest.json`

## 10) OCR Augment Names (MVP)

- Method: `POST`
- Path: `/api/ocr/augments-names`
- Content-Type: `multipart/form-data`
- Field: `image` (screenshot image file)

Response:

- `names`: fixed length 3, ordered `left/middle/right`
  - `position`: `left|middle|right`
  - `name_cn`: recognized + dictionary-corrected Chinese name
  - `confidence`: `0~1`
- `debug.cards`: detected card rectangles after resize

Example:

```json
{
  "names": [
    {"position": "left", "name_cn": "尤里卡", "confidence": 0.86},
    {"position": "middle", "name_cn": "珠光护手", "confidence": 0.81},
    {"position": "right", "name_cn": "炼狱导管", "confidence": 0.79}
  ],
  "debug": {
    "cards": [
      {"x": 123, "y": 220, "w": 360, "h": 740},
      {"x": 520, "y": 220, "w": 360, "h": 740},
      {"x": 920, "y": 220, "w": 360, "h": 740}
    ]
  }
}
```

Notes:

- Card detection: Canny + contour rectangle filtering + perspective warp.
- Title OCR output is fuzzy-matched against `/api/augments/mapping` `name_cn` dictionary.
- If no OCR engine is available, API returns `503` with install guidance.

### OCR Debug Tool

Run from `backend/`:

```bash
python tools/debug_ocr_cards.py path/to/screenshot.jpg
```

Output files (default `backend/.cache/ocr_debug/`):

- `debug_overlay.jpg`
- `card_left.jpg`, `card_middle.jpg`, `card_right.jpg`
- `title_left.jpg`, `title_middle.jpg`, `title_right.jpg`

## 11) Battle Recommend

- Method: `POST`
- Path: `/api/battle/recommend`
- Description: server-side recommendation for battle round candidates

Request body:

```json
{
  "championId": 887,
  "round": 2,
  "pickedAugmentIds": [1234],
  "candidateAugmentIds": [5678, 9101, 1112]
}
```

Response fields:

- `candidates` array
  - `augmentId` number
  - `name_cn` string
  - `winRate` number|null
  - `pickRate` number|null
  - `tier` string
  - `rarity` string
  - `iconUrl` string
- `recommendSingle` object|null
  - `augmentId` number
  - `name_cn` string
  - `reason` string
- `recommendCombo` object|null
  - `augmentId` number
  - `name_cn` string
  - `comboRank` number
  - `reason` string
- `comboSource` string (currently `TOP10`)
