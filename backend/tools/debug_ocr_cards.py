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
    OcrAugmentError,
    OcrDependencyError,
    recognize_augment_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug OCR card detection and title extraction for augment screenshots."
    )
    parser.add_argument("image", type=Path, help="Path to screenshot image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / ".cache" / "ocr_debug",
        help="Directory for debug outputs",
    )
    return parser.parse_args()


def _save_image(path: Path, image: object) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise OcrDependencyError("opencv-python-headless is required to save debug images") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    payload = image_path.read_bytes()

    try:
        mapping = get_augment_mapping(force_refresh=False)
    except AugmentMappingError:
        mapping = {}

    result, artifacts = recognize_augment_names(
        image_bytes=payload,
        augment_mapping=mapping,
        keep_debug_images=True,
    )

    if artifacts is None:
        raise RuntimeError("no debug artifacts returned")

    _save_image(output_dir / "debug_overlay.jpg", artifacts["overlay"])

    positions = ["left", "middle", "right"]
    for idx, image in enumerate(artifacts.get("cards", [])):
        position = positions[idx] if idx < len(positions) else f"{idx + 1}"
        _save_image(output_dir / f"card_{position}.jpg", image)

    for idx, image in enumerate(artifacts.get("titles", [])):
        position = positions[idx] if idx < len(positions) else f"{idx + 1}"
        _save_image(output_dir / f"title_{position}.jpg", image)
    for idx, image in enumerate(artifacts.get("titles_processed", [])):
        position = positions[idx] if idx < len(positions) else f"{idx + 1}"
        _save_image(output_dir / f"title_{position}_processed.jpg", image)

    print("[OCR backend]", result.get("debug", {}).get("engine"))
    print("[Detected names]")
    for row in result.get("names", []):
        print(
            f"- {row.get('position')}: {row.get('name_cn')} "
            f"(confidence={row.get('confidence')})"
        )

    print("[Raw OCR debug]")
    for row in result.get("debug", {}).get("ocr", []):
        top5 = row.get("candidates_top5")
        print(
            f"- {row.get('position')}: best_text={row.get('best_text')} "
            f"best_conf={row.get('best_text_conf')} final={row.get('final')}"
        )
        if isinstance(top5, list):
            print(f"  top5={top5}")

    print("[Output files]", output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except OcrAugmentError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
