from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final
from urllib.parse import urlparse
from urllib.request import Request, urlopen

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.services.augment_mapping import get_augment_mapping  # noqa: E402

USER_AGENT: Final[str] = "Mozilla/5.0 (tft-helper-assets)"
VALID_EXTS: Final[set[str]] = {".png", ".webp", ".jpg", ".jpeg", ".gif"}
CONTENT_TYPE_TO_EXT: Final[dict[str, str]] = {
    "image/png": ".png",
    "image/webp": ".webp",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
}

OUTPUT_DIR = BACKEND_DIR / "static" / "augments"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
DEFAULT_WORKERS = 16


def resolve_extension(image_url: str, content_type: str) -> str:
    path_ext = Path(urlparse(image_url).path).suffix.lower()
    if path_ext in VALID_EXTS:
        return path_ext
    normalized_type = content_type.split(";")[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(normalized_type, ".png")


def download_image(image_url: str, augment_id: str) -> str:
    req = Request(image_url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
        content_type = resp.headers.get("Content-Type", "")
    ext = resolve_extension(image_url, content_type)
    filename = f"{augment_id}{ext}"
    (OUTPUT_DIR / filename).write_bytes(data)
    return filename


def find_existing_file(augment_id: str) -> str | None:
    for ext in VALID_EXTS:
        path = OUTPUT_DIR / f"{augment_id}{ext}"
        if path.exists():
            return path.name
    return None


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = get_augment_mapping(force_refresh=True)
    if not mapping:
        raise RuntimeError("augment mapping is empty")

    saved: dict[str, str] = {}
    failed = 0
    skipped = 0
    to_download: list[tuple[str, str]] = []

    for augment_id, item in sorted(mapping.items(), key=lambda x: int(x[0])):
        icon_url = item.get("icon") if isinstance(item, dict) else None
        if not isinstance(icon_url, str) or not icon_url.strip():
            skipped += 1
            continue
        existing = find_existing_file(augment_id)
        if existing:
            saved[augment_id] = existing
            continue
        to_download.append((augment_id, icon_url))

    workers = max(1, int(os.getenv("ASSET_SYNC_WORKERS", str(DEFAULT_WORKERS))))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(download_image, icon_url, augment_id): augment_id
            for augment_id, icon_url in to_download
        }
        for future in as_completed(future_map):
            augment_id = future_map[future]
            try:
                saved[augment_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[WARN] failed to download augment {augment_id}: {exc}")

    manifest = {
        "updatedAt": int(time.time()),
        "source": "augment_mapping",
        "items": saved,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    print(
        f"[OK] augment assets synced: {len(saved)} files, "
        f"failed={failed}, skipped={skipped}, manifest={MANIFEST_PATH}, "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
