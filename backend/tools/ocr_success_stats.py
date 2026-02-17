from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.services.augment_mapping import AugmentMappingError, get_augment_mapping  # noqa: E402
from app.services.ocr_augments import (  # noqa: E402
    OcrDependencyError,
    OcrEngineUnavailableError,
    recognize_augment_names,
)


DEFAULT_MANIFEST = BACKEND_DIR / "tests" / "assets" / "ocr_samples" / "manifest.json"
DEFAULT_ASSETS_DIR = BACKEND_DIR / "tests" / "assets" / "ocr_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute OCR success stats for screenshot dataset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--min-hits", type=int, default=2, help="minimum correct positions per image")
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        return []
    return [item for item in samples if isinstance(item, dict)]


def _mapping_from_manifest(samples: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    names: dict[str, int] = {}
    next_id = 100000
    for sample in samples:
        expected = sample.get("expected")
        if not isinstance(expected, dict):
            continue
        for key in ("left", "middle", "right"):
            value = expected.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in names:
                names[value.strip()] = next_id
                next_id += 1

    return {
        str(aid): {
            "augmentId": str(aid),
            "name_cn": name,
            "rarity": "unknown",
        }
        for name, aid in names.items()
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    assets_dir = args.assets_dir.resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    samples = _load_manifest(manifest_path)
    if not samples:
        raise RuntimeError("no samples found in manifest")

    try:
        mapping = get_augment_mapping(force_refresh=False)
    except AugmentMappingError:
        mapping = _mapping_from_manifest(samples)

    ok_images = 0
    total_images = 0
    total_hits = 0
    total_slots = 0

    for sample in samples:
        rel = sample.get("file")
        expected = sample.get("expected")
        if not isinstance(rel, str) or not isinstance(expected, dict):
            continue

        image_path = assets_dir / rel
        if not image_path.exists():
            print(f"[SKIP] missing file: {image_path}")
            continue

        total_images += 1
        payload = image_path.read_bytes()

        try:
            result, _ = recognize_augment_names(
                image_bytes=payload,
                augment_mapping=mapping,
                keep_debug_images=False,
            )
        except (OcrDependencyError, OcrEngineUnavailableError) as exc:
            print(f"[ERROR] OCR backend unavailable: {exc}")
            return

        got = {
            str(row.get("position")): str(row.get("name_cn") or "")
            for row in result.get("names", [])
            if isinstance(row, dict)
        }

        hits = 0
        for pos in ("left", "middle", "right"):
            exp = str(expected.get(pos) or "").strip()
            if not exp:
                continue
            total_slots += 1
            if got.get(pos, "").strip() == exp:
                hits += 1
                total_hits += 1

        image_ok = hits >= args.min_hits
        if image_ok:
            ok_images += 1
        print(f"[{ 'PASS' if image_ok else 'FAIL' }] {rel} hits={hits}/3 got={got}")

    image_rate = (ok_images / total_images) if total_images else 0.0
    slot_rate = (total_hits / total_slots) if total_slots else 0.0
    print("\n=== OCR Success Stats ===")
    print(f"images_passed: {ok_images}/{total_images} ({image_rate:.2%})")
    print(f"slots_correct: {total_hits}/{total_slots} ({slot_rate:.2%})")


if __name__ == "__main__":
    main()
