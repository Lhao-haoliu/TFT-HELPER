from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

BASE_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
CHAMPION_DIR = BASE_STATIC_DIR / "champions"
AUGMENT_DIR = BASE_STATIC_DIR / "augments"
CHAMPION_MANIFEST_PATH = CHAMPION_DIR / "manifest.json"
AUGMENT_MANIFEST_PATH = AUGMENT_DIR / "manifest.json"

_cache_lock = Lock()
_manifest_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _load_manifest(path: Path) -> dict[str, Any]:
    cache_key = str(path)
    mtime = path.stat().st_mtime if path.exists() else -1.0

    with _cache_lock:
        cached = _manifest_cache.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    if not path.exists():
        data: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, dict) else {}
        except Exception:
            data = {}

    with _cache_lock:
        _manifest_cache[cache_key] = (mtime, data)
    return data


def _pick_filename(manifest: dict[str, Any], item_id: str) -> str | None:
    items = manifest.get("items")
    if isinstance(items, dict):
        filename = items.get(item_id)
        if isinstance(filename, str) and filename.strip():
            return filename.strip()
    return None


def get_champion_icon_url(champion_id: str) -> str:
    cid = str(champion_id).strip()
    manifest = _load_manifest(CHAMPION_MANIFEST_PATH)
    filename = _pick_filename(manifest, cid)
    if filename:
        return f"/static/champions/{filename}"
    return f"/static/champions/{cid}.png"


def get_augment_icon_url(augment_id: int | str) -> str:
    aid = str(augment_id).strip()
    manifest = _load_manifest(AUGMENT_MANIFEST_PATH)
    filename = _pick_filename(manifest, aid)
    if filename:
        return f"/static/augments/{filename}"
    return f"/static/augments/{aid}.png"
