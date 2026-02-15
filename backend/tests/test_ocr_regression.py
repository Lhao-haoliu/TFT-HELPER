from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.ocr_augments import (  # type: ignore[import-not-found]
    OcrDependencyError,
    OcrEngineUnavailableError,
    recognize_augment_names,
)


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "ocr_samples"
MANIFEST_PATH = ASSET_DIR / "manifest.json"

SIMPLE_MAPPING = {
    "101": {"augmentId": "101", "name_cn": "尤里卡"},
    "102": {"augmentId": "102", "name_cn": "珠光护手"},
    "103": {"augmentId": "103", "name_cn": "炼狱导管"},
}


def _load_manifest() -> list[dict[str, object]]:
    if not MANIFEST_PATH.exists():
        pytest.skip(f"missing OCR sample manifest: {MANIFEST_PATH}")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        pytest.skip("OCR sample manifest has no samples")
    return [item for item in samples if isinstance(item, dict)]


def _index_names(resp: dict[str, object]) -> dict[str, str]:
    out: dict[str, str] = {}
    names = resp.get("names")
    if not isinstance(names, list):
        return out
    for row in names:
        if not isinstance(row, dict):
            continue
        pos = str(row.get("position") or "")
        name_cn = str(row.get("name_cn") or "")
        if pos:
            out[pos] = name_cn
    return out


@pytest.mark.parametrize("sample", _load_manifest(), ids=lambda x: str(x.get("file")))
def test_ocr_sample_has_minimum_hits(sample: dict[str, object]) -> None:
    rel_path = sample.get("file")
    expected = sample.get("expected")
    if not isinstance(rel_path, str) or not rel_path.strip():
        pytest.fail("invalid sample entry: missing file")
    if not isinstance(expected, dict):
        pytest.fail("invalid sample entry: missing expected")

    image_path = ASSET_DIR / rel_path
    if not image_path.exists():
        pytest.fail(f"missing sample image: {image_path}")

    image_bytes = image_path.read_bytes()
    try:
        resp, _ = recognize_augment_names(
            image_bytes=image_bytes,
            augment_mapping=SIMPLE_MAPPING,
            keep_debug_images=False,
        )
    except (OcrDependencyError, OcrEngineUnavailableError) as exc:
        pytest.skip(f"OCR backend unavailable: {exc}")

    names_by_pos = _index_names(resp)

    # Enforce stable position order and per-image quality baseline.
    assert [row.get("position") for row in resp.get("names", [])] == [
        "left",
        "middle",
        "right",
    ]

    hits = 0
    for pos in ("left", "middle", "right"):
        exp = str(expected.get(pos, "")).strip()
        got = names_by_pos.get(pos, "").strip()
        if exp and got == exp:
            hits += 1

    assert hits >= 2, f"{rel_path} expected >=2/3 hits, got {hits}/3: {names_by_pos}"

