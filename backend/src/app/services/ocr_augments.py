from __future__ import annotations

import json
import shutil
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process


class OcrAugmentError(Exception):
    """Base exception for OCR augment pipeline."""


class OcrInputError(OcrAugmentError):
    """Raised when OCR input payload is invalid."""


class OcrDependencyError(OcrAugmentError):
    """Raised when required OCR dependency is missing."""


class OcrEngineUnavailableError(OcrAugmentError):
    """Raised when OCR engine is installed but unavailable at runtime."""


SUPPORTED_RARITIES = {"prismatic", "gold", "silver", "unknown"}
POSITIONS = ("left", "middle", "right")
MATCH_PRIMARY_THRESHOLD = 72.0
FINAL_CONFIDENCE_ICON_FALLBACK = 0.72

_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
_AUGMENT_STATIC_DIR = _STATIC_DIR / "augments"
_AUGMENT_MANIFEST_PATH = _AUGMENT_STATIC_DIR / "manifest.json"

_icon_lock = Lock()
_icon_db_cache: list[dict[str, Any]] | None = None


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise OcrDependencyError("opencv-python-headless is required") from exc
    return cv2


def _normalize_rarity(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in SUPPORTED_RARITIES:
        return text
    if text in ("妫卞僵", "prism", "prismatic"):
        return "prismatic"
    if text in ("榛勯噾", "gold"):
        return "gold"
    if text in ("鐧介摱", "silver"):
        return "silver"
    return "unknown"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clean_text(text: str) -> str:
    out = text.strip()
    out = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", out)
    return out


def _extract_mapping_name(item: dict[str, object]) -> str | None:
    for key in ("name_cn", "name", "title", "displayName"):
        raw = item.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _build_name_pool(
    augment_mapping: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    by_name: dict[str, dict[str, object]] = {}
    by_rarity: dict[str, list[dict[str, object]]] = {
        "prismatic": [],
        "gold": [],
        "silver": [],
        "unknown": [],
    }
    all_items: list[dict[str, object]] = []

    for augment_id, raw in augment_mapping.items():
        if not isinstance(raw, dict):
            continue
        name_cn = _extract_mapping_name(raw)
        if not name_cn:
            continue
        rarity = _normalize_rarity(raw.get("rarity"))
        item = {
            "augmentId": str(augment_id),
            "name_cn": name_cn,
            "rarity": rarity,
        }
        by_name[name_cn] = item
        by_rarity.setdefault(rarity, []).append(item)
        all_items.append(item)

    return by_name, by_rarity, all_items


def _decode_image(image_bytes: bytes) -> tuple[Any, np.ndarray]:
    if not image_bytes:
        raise OcrInputError("empty image bytes")
    cv2 = _require_cv2()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise OcrInputError("failed to decode image bytes")
    return cv2, image


def _resize_to_width(cv2: Any, image: np.ndarray, width: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        raise OcrInputError("invalid image size")
    if w == width:
        return image
    scale = width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _detect_cards(cv2: Any, image: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image.shape[:2]
    img_area = float(img_w * img_h)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.03:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        ratio = h / float(w)
        if ratio < 1.15 or ratio > 2.8:
            continue

        card: dict[str, Any] = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": float(area),
            "quad": None,
        }
        if len(approx) == 4:
            card["quad"] = approx.reshape(4, 2).astype(np.float32)
        candidates.append(card)

    candidates.sort(key=lambda item: item["area"], reverse=True)

    picked: list[dict[str, Any]] = []
    for cand in candidates:
        overlap = False
        for ex in picked:
            dx = abs(cand["x"] - ex["x"])
            if dx < min(cand["w"], ex["w"]) * 0.5:
                overlap = True
                break
        if not overlap:
            picked.append(cand)
        if len(picked) >= 3:
            break

    if len(picked) < 3:
        # Fallback to fixed 3 vertical windows.
        section_w = img_w // 3
        margin_x = max(8, int(section_w * 0.06))
        margin_y = max(8, int(img_h * 0.14))
        height = int(img_h * 0.68)
        top = int(img_h * 0.15)
        picked = []
        for idx in range(3):
            x0 = idx * section_w + margin_x
            w0 = section_w - 2 * margin_x
            picked.append(
                {
                    "x": int(x0),
                    "y": int(top + margin_y // 2),
                    "w": int(max(40, w0)),
                    "h": int(max(120, height - margin_y)),
                    "area": float(max(1, w0 * height)),
                    "quad": None,
                }
            )

    picked.sort(key=lambda item: item["x"])
    return picked[:3]


def _warp_card(cv2: Any, image: np.ndarray, card: dict[str, Any]) -> np.ndarray:
    x = int(card["x"])
    y = int(card["y"])
    w = int(card["w"])
    h = int(card["h"])
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    quad = card.get("quad")
    if isinstance(quad, np.ndarray) and quad.shape == (4, 2):
        src = _order_quad_points(quad)
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (w, h))

    return image[y : y + h, x : x + w].copy()


def _detect_rarity_from_card(cv2: Any, card_image: np.ndarray) -> tuple[str, dict[str, float]]:
    h, w = card_image.shape[:2]
    bw = max(4, int(w * 0.08))
    bh = max(4, int(h * 0.08))

    strips = [
        card_image[:bh, :],
        card_image[h - bh :, :],
        card_image[:, :bw],
        card_image[:, w - bw :],
    ]
    border = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)

    hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_mean = float(np.mean(hsv[:, 0]))
    s_mean = float(np.mean(hsv[:, 1]))
    v_mean = float(np.mean(hsv[:, 2]))

    if s_mean < 45 and v_mean > 110:
        rarity = "silver"
    elif (18 <= h_mean <= 40 and s_mean > 75) or (s_mean > 110 and 12 <= h_mean <= 50):
        rarity = "gold"
    elif (h_mean >= 120 or h_mean <= 8) and s_mean > 55:
        rarity = "prismatic"
    else:
        rarity = "unknown"

    return rarity, {"h_mean": h_mean, "s_mean": s_mean, "v_mean": v_mean}


def _crop_title_line(card: np.ndarray) -> np.ndarray:
    h, w = card.shape[:2]
    y0 = max(0, int(h * 0.38))
    y1 = min(h, int(h * 0.55))
    x0 = max(0, int(w * 0.12))
    x1 = min(w, int(w * 0.88))
    if y1 <= y0 or x1 <= x0:
        return card.copy()
    return card[y0:y1, x0:x1].copy()


def _preprocess_title(cv2: Any, title_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(title_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    # White background + black text for Tesseract readability.
    if np.mean(binary) < 127:
        binary = 255 - binary
    return binary


def _ocr_with_tesseract(image: np.ndarray) -> list[tuple[str, float]]:
    try:
        import pytesseract  # type: ignore
        from pytesseract import TesseractNotFoundError  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise OcrDependencyError("pytesseract is required") from exc

    config = "--oem 3 --psm 7"
    try:
        data = pytesseract.image_to_data(
            image,
            lang="chi_sim",
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except TesseractNotFoundError as exc:
        raise OcrEngineUnavailableError(
            "pytesseract failed: tesseract is not installed or not in PATH"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise OcrEngineUnavailableError(f"tesseract OCR failed: {exc}") from exc

    texts = data.get("text", [])
    confs = data.get("conf", [])
    out: list[tuple[str, float]] = []
    for idx, raw in enumerate(texts):
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            conf = float(confs[idx]) / 100.0
        except Exception:
            conf = 0.0
        cleaned = _clean_text(text)
        if cleaned:
            out.append((cleaned, max(0.0, min(1.0, conf))))
    return out


def _ocr_with_paddle(image: np.ndarray) -> list[tuple[str, float]]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return []

    global _paddle_ocr_singleton
    if "_paddle_ocr_singleton" not in globals():
        _paddle_ocr_singleton = None

    if _paddle_ocr_singleton is None:
        try:
            _paddle_ocr_singleton = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
        except Exception:
            return []

    try:
        result = _paddle_ocr_singleton.ocr(image, cls=False)
    except Exception:
        return []

    out: list[tuple[str, float]] = []
    for block in result or []:
        for line in block or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            text_conf = line[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue
            text = _clean_text(str(text_conf[0] or ""))
            conf = _safe_float(text_conf[1], 0.0)
            if text:
                out.append((text, max(0.0, min(1.0, conf))))
    return out


def _run_ocr(image: np.ndarray) -> tuple[list[tuple[str, float]], str]:
    paddle_lines = _ocr_with_paddle(image)
    if paddle_lines:
        return paddle_lines, "paddleocr"
    tess_lines = _ocr_with_tesseract(image)
    return tess_lines, "tesseract"


def inspect_ocr_runtime() -> dict[str, object]:
    status: dict[str, object] = {
        "opencv": False,
        "pytesseract": False,
        "paddleocr": False,
        "tesseract_cmd": "",
        "tesseract_version": "",
        "tesseract_langs": [],
        "errors": [],
    }

    errors: list[str] = []
    try:
        _require_cv2()
        status["opencv"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"opencv unavailable: {exc}")

    try:
        import pytesseract  # type: ignore

        status["pytesseract"] = True
        tesseract_cmd = shutil.which("tesseract") or ""
        status["tesseract_cmd"] = tesseract_cmd
        if not tesseract_cmd:
            errors.append("tesseract binary not found in PATH")
        else:
            try:
                proc = subprocess.run(
                    [tesseract_cmd, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version_line = (proc.stdout or proc.stderr or "").splitlines()
                status["tesseract_version"] = version_line[0] if version_line else ""
            except Exception as exc:  # noqa: BLE001
                errors.append(f"failed to read tesseract version: {exc}")

            try:
                langs = pytesseract.get_languages(config="")
                status["tesseract_langs"] = list(langs)
                if "chi_sim" not in langs:
                    errors.append("chi_sim language pack missing")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"failed to query tesseract languages: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pytesseract unavailable: {exc}")

    try:
        from paddleocr import PaddleOCR  # type: ignore # noqa: F401

        status["paddleocr"] = True
    except Exception:
        status["paddleocr"] = False

    status["errors"] = errors
    return status


def _match_from_pool(
    text: str,
    pool: list[dict[str, object]],
    top_k: int = 5,
) -> list[dict[str, object]]:
    if not text or not pool:
        return []

    name_to_item: dict[str, dict[str, object]] = {}
    for item in pool:
        name = str(item.get("name_cn") or "").strip()
        if name:
            name_to_item[name] = item

    if not name_to_item:
        return []

    matches = process.extract(
        text,
        list(name_to_item.keys()),
        scorer=fuzz.WRatio,
        limit=top_k,
    )

    out: list[dict[str, object]] = []
    for name, score, _ in matches:
        item = name_to_item.get(str(name))
        if not item:
            continue
        out.append(
            {
                "augmentId": int(str(item.get("augmentId") or "0") or "0"),
                "name_cn": str(item.get("name_cn") or ""),
                "rarity": str(item.get("rarity") or "unknown"),
                "score": float(score) / 100.0,
            }
        )
    return out


def _phash64(cv2: Any, image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8]
    flat = low.flatten()
    median = np.median(flat[1:]) if flat.size > 1 else 0.0
    bits = 0
    for value in flat:
        bits = (bits << 1) | (1 if value > median else 0)
    return int(bits)


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _build_icon_db() -> list[dict[str, Any]]:
    cv2 = _require_cv2()
    manifest_map: dict[str, str] = {}
    if _AUGMENT_MANIFEST_PATH.exists():
        try:
            payload = json.loads(_AUGMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
            items = payload.get("items") if isinstance(payload, dict) else None
            if isinstance(items, dict):
                for k, v in items.items():
                    if isinstance(k, str) and isinstance(v, str):
                        manifest_map[k.strip()] = v.strip()
        except Exception:
            manifest_map = {}

    candidates: list[tuple[str, Path]] = []
    for augment_id, filename in manifest_map.items():
        path = _AUGMENT_STATIC_DIR / filename
        if path.exists():
            candidates.append((augment_id, path))

    if not candidates and _AUGMENT_STATIC_DIR.exists():
        for path in _AUGMENT_STATIC_DIR.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            stem = path.stem
            if stem.isdigit():
                candidates.append((stem, path))

    orb = cv2.ORB_create(nfeatures=256)
    db: list[dict[str, Any]] = []
    for augment_id, path in candidates:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ph = _phash64(cv2, img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        db.append(
            {
                "augmentId": int(augment_id),
                "phash": ph,
                "des": des,
            }
        )
    return db


def _get_icon_db() -> list[dict[str, Any]]:
    global _icon_db_cache
    with _icon_lock:
        if _icon_db_cache is None:
            _icon_db_cache = _build_icon_db()
        return _icon_db_cache


def _icon_region(card: np.ndarray) -> np.ndarray:
    h, w = card.shape[:2]
    y0 = max(0, int(h * 0.08))
    y1 = min(h, int(h * 0.36))
    x0 = max(0, int(w * 0.22))
    x1 = min(w, int(w * 0.78))
    if y1 <= y0 or x1 <= x0:
        return card.copy()
    return card[y0:y1, x0:x1].copy()


def _icon_match(
    card: np.ndarray,
    rarity: str,
    augment_mapping: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    cv2 = _require_cv2()
    icon = _icon_region(card)
    db = _get_icon_db()
    if not db:
        return None

    ph = _phash64(cv2, icon)
    orb = cv2.ORB_create(nfeatures=256)
    gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    _, des = orb.detectAndCompute(gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best: dict[str, object] | None = None
    for item in db:
        augment_id = int(item["augmentId"])
        raw = augment_mapping.get(str(augment_id), {})
        if rarity in ("prismatic", "gold", "silver"):
            item_rarity = _normalize_rarity(raw.get("rarity")) if isinstance(raw, dict) else "unknown"
            if item_rarity != rarity:
                continue

        ph_sim = 1.0 - (_hamming64(ph, int(item["phash"])) / 64.0)
        orb_score = 0.0
        if des is not None and item.get("des") is not None:
            try:
                matches = bf.match(des, item["des"])
                if matches:
                    good = [m for m in matches if m.distance < 45]
                    orb_score = min(1.0, len(good) / 30.0)
            except Exception:
                orb_score = 0.0

        score = 0.65 * ph_sim + 0.35 * orb_score
        if best is None or score > float(best["score"]):
            mapped = augment_mapping.get(str(augment_id), {})
            best = {
                "augmentId": augment_id,
                "name_cn": str(mapped.get("name_cn") or f"#{augment_id}") if isinstance(mapped, dict) else f"#{augment_id}",
                "score": float(score),
                "method": "icon",
            }

    return best


def _resolve_name(
    *,
    text: str,
    rarity_hint: str,
    by_rarity: dict[str, list[dict[str, object]]],
    all_items: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], bool]:
    rarity_pool = by_rarity.get(rarity_hint, []) if rarity_hint in SUPPORTED_RARITIES else []
    primary_pool = rarity_pool if rarity_pool else all_items

    top_primary = _match_from_pool(text, primary_pool, top_k=5)
    top_global = _match_from_pool(text, all_items, top_k=5) if primary_pool is not all_items else top_primary

    picked = top_primary[0] if top_primary else None
    used_fallback = False

    if picked is None or float(picked.get("score", 0.0)) * 100.0 < MATCH_PRIMARY_THRESHOLD:
        if top_global:
            picked = top_global[0]
            used_fallback = True

    if picked is None:
        picked = {
            "augmentId": 0,
            "name_cn": text or "",
            "rarity": "unknown",
            "score": 0.0,
        }

    combined_top = top_primary
    if used_fallback:
        combined_top = top_global

    low_confidence = float(picked.get("score", 0.0)) < FINAL_CONFIDENCE_ICON_FALLBACK
    return picked, combined_top, low_confidence


@dataclass
class CardResult:
    position: str
    augment_id: int
    name_cn: str
    confidence: float
    rarity: str
    low_confidence: bool
    debug: dict[str, object]


def recognize_augment_names(
    *,
    image_bytes: bytes,
    augment_mapping: dict[str, dict[str, object]],
    keep_debug_images: bool = False,
    ui_rarities: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    cv2, image_raw = _decode_image(image_bytes)
    image = _resize_to_width(cv2, image_raw)

    _, by_rarity, all_items = _build_name_pool(augment_mapping)
    if not all_items:
        raise OcrAugmentError("augment mapping is empty; cannot resolve OCR names")

    cards = _detect_cards(cv2, image)
    if len(cards) != 3:
        raise OcrAugmentError("failed to detect exactly 3 augment cards")

    overlay = image.copy()
    card_images: list[np.ndarray] = []
    title_images: list[np.ndarray] = []
    card_results: list[CardResult] = []
    ocr_engine = "unknown"

    normalized_ui_rarities: list[str] = []
    for rarity in ui_rarities or []:
        normalized_ui_rarities.append(_normalize_rarity(rarity))

    for idx, card in enumerate(cards):
        x, y, w, h = int(card["x"]), int(card["y"]), int(card["w"]), int(card["h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)

        warped = _warp_card(cv2, image, card)
        card_images.append(warped)

        detected_rarity, rarity_metrics = _detect_rarity_from_card(cv2, warped)
        ui_rarity = normalized_ui_rarities[idx] if idx < len(normalized_ui_rarities) else "unknown"
        final_rarity = ui_rarity if ui_rarity != "unknown" else detected_rarity

        title_crop = _crop_title_line(warped)
        title_images.append(title_crop)
        title_processed = _preprocess_title(cv2, title_crop)

        ocr_lines, engine = _run_ocr(title_processed)
        ocr_engine = engine

        if not ocr_lines:
            best_text = ""
            best_text_conf = 0.0
        else:
            ocr_lines.sort(key=lambda row: (row[1], len(row[0])), reverse=True)
            best_text, best_text_conf = ocr_lines[0]

        best_match, top_candidates, low_conf = _resolve_name(
            text=best_text,
            rarity_hint=final_rarity,
            by_rarity=by_rarity,
            all_items=all_items,
        )

        final_augment_id = int(best_match.get("augmentId", 0))
        final_name = str(best_match.get("name_cn") or f"#{final_augment_id}")
        final_conf = float(best_match.get("score", 0.0))
        used_icon_fallback = False

        if final_conf < FINAL_CONFIDENCE_ICON_FALLBACK:
            icon_match = _icon_match(
                card=warped,
                rarity=final_rarity,
                augment_mapping=augment_mapping,
            )
            if icon_match is not None and float(icon_match.get("score", 0.0)) > final_conf:
                final_augment_id = int(icon_match["augmentId"])
                final_name = str(icon_match["name_cn"])
                final_conf = float(icon_match["score"])
                low_conf = final_conf < FINAL_CONFIDENCE_ICON_FALLBACK
                used_icon_fallback = True

        card_results.append(
            CardResult(
                position=POSITIONS[idx] if idx < len(POSITIONS) else str(idx),
                augment_id=final_augment_id,
                name_cn=final_name,
                confidence=max(0.0, min(1.0, final_conf)),
                rarity=final_rarity,
                low_confidence=bool(low_conf),
                debug={
                    "position": POSITIONS[idx] if idx < len(POSITIONS) else str(idx),
                    "rarity": {
                        "ui": ui_rarity,
                        "detected": detected_rarity,
                        "final": final_rarity,
                        "hsv": rarity_metrics,
                    },
                    "ocr_raw_lines": [
                        {"text": text, "conf": conf} for text, conf in ocr_lines
                    ],
                    "best_text": best_text,
                    "best_text_conf": best_text_conf,
                    "candidates_top5": top_candidates,
                    "final": {
                        "augmentId": final_augment_id,
                        "name_cn": final_name,
                        "confidence": final_conf,
                        "low_confidence": bool(low_conf),
                        "used_icon_fallback": used_icon_fallback,
                    },
                },
            )
        )

    names_out: list[dict[str, object]] = []
    debug_cards_out: list[dict[str, object]] = []
    for card, row in zip(cards, card_results):
        names_out.append(
            {
                "position": row.position,
                "name_cn": row.name_cn,
                "confidence": round(row.confidence, 4),
                "augmentId": row.augment_id,
                "rarity": row.rarity,
                "low_confidence": row.low_confidence,
            }
        )
        debug_cards_out.append(
            {
                "x": int(card["x"]),
                "y": int(card["y"]),
                "w": int(card["w"]),
                "h": int(card["h"]),
                **row.debug,
            }
        )

    result = {
        "names": names_out,
        "debug": {
            "engine": ocr_engine,
            "cards": [
                {
                    "x": int(card["x"]),
                    "y": int(card["y"]),
                    "w": int(card["w"]),
                    "h": int(card["h"]),
                }
                for card in cards
            ],
            "ocr": debug_cards_out,
        },
    }

    artifacts: dict[str, object] | None = None
    if keep_debug_images:
        artifacts = {
            "overlay": overlay,
            "cards": card_images,
            "titles": title_images,
            "titles_processed": [
                _preprocess_title(cv2, title) for title in title_images
            ],
        }

    return result, artifacts


def resolve_augment_names(
    *,
    names: list[str],
    augment_mapping: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    _, by_rarity, all_items = _build_name_pool(augment_mapping)
    if not all_items:
        return []

    resolved: list[dict[str, object]] = []
    for raw in names:
        text = _clean_text(str(raw or ""))
        if not text:
            continue

        best, _, _ = _resolve_name(
            text=text,
            rarity_hint="unknown",
            by_rarity=by_rarity,
            all_items=all_items,
        )
        resolved.append(
            {
                "name_cn": str(raw),
                "augmentId": int(best.get("augmentId") or 0),
                "confidence": round(float(best.get("score") or 0.0), 4),
            }
        )

    return resolved
