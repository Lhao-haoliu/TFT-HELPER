from __future__ import annotations

import difflib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TARGET_WIDTH = 1280
_POSITION_LABELS = ("left", "middle", "right")
_TITLE_CROP_RATIOS = (
    (0.08, 0.26, 0.92, 0.50),
    (0.08, 0.32, 0.92, 0.58),
)


class OcrAugmentError(Exception):
    """Base error for augment OCR flow."""


class OcrInputError(OcrAugmentError):
    """Raised when image input is invalid."""


class OcrDependencyError(OcrAugmentError):
    """Raised when OCR dependencies are not installed."""


class OcrEngineUnavailableError(OcrAugmentError):
    """Raised when no OCR backend is available."""


@dataclass
class CardRegion:
    quad: Any
    bbox: tuple[int, int, int, int]
    score: float


_cv2_np_lock = threading.Lock()
_cv2_np_cache: tuple[Any, Any] | None = None


def _get_cv2_np() -> tuple[Any, Any]:
    global _cv2_np_cache
    with _cv2_np_lock:
        if _cv2_np_cache is not None:
            return _cv2_np_cache

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OcrDependencyError(
                "opencv-python-headless and numpy are required for OCR card detection"
            ) from exc

        _cv2_np_cache = (cv2, np)
        return _cv2_np_cache


class _OcrBackend:
    def __init__(self) -> None:
        self.name = "none"
        self._paddle = None
        self._easyocr = None
        self._pytesseract = None
        self._pytesseract_output = None
        self._init_backend()

    def _init_backend(self) -> None:
        # Preferred backend: PaddleOCR.
        try:
            from paddleocr import PaddleOCR  # type: ignore

            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                use_gpu=False,
            )
            self.name = "paddleocr"
            return
        except Exception:  # noqa: BLE001
            self._paddle = None

        # Fallback backend: EasyOCR.
        try:
            import easyocr  # type: ignore

            self._easyocr = easyocr.Reader(["ch_sim", "en"], gpu=False)
            self.name = "easyocr"
            return
        except Exception:  # noqa: BLE001
            self._easyocr = None

        # Final fallback backend: pytesseract + system tesseract.
        try:
            import pytesseract  # type: ignore
            from pytesseract import Output  # type: ignore

            tesseract_cmd = self._resolve_tesseract_cmd()
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            tessdata_prefix = self._resolve_tessdata_prefix()
            if tessdata_prefix:
                os.environ.setdefault("TESSDATA_PREFIX", tessdata_prefix)

            self._pytesseract = pytesseract
            self._pytesseract_output = Output
            self.name = "pytesseract"
        except Exception:  # noqa: BLE001
            self._pytesseract = None
            self._pytesseract_output = None
            self.name = "none"

    def _resolve_tesseract_cmd(self) -> str | None:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return None

    def _resolve_tessdata_prefix(self) -> str | None:
        backend_root = Path(__file__).resolve().parents[3]
        candidate = backend_root / ".tessdata"
        if candidate.exists():
            return str(candidate)
        return None

    def read_lines(self, image_bgr: Any) -> list[tuple[str, float]]:
        cv2, _ = _get_cv2_np()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self._paddle is not None:
            try:
                result = self._paddle.ocr(rgb, cls=True)
            except Exception as exc:  # noqa: BLE001
                raise OcrEngineUnavailableError(f"paddleocr failed: {exc}") from exc

            lines: list[tuple[str, float]] = []
            page = result[0] if isinstance(result, list) and result else []
            for row in page or []:
                if not isinstance(row, list) or len(row) < 2:
                    continue
                text_meta = row[1]
                if not isinstance(text_meta, (list, tuple)) or len(text_meta) < 2:
                    continue
                text = str(text_meta[0]).strip()
                if not text:
                    continue
                conf = float(text_meta[1])
                lines.append((text, max(0.0, min(1.0, conf))))
            return lines

        if self._easyocr is not None:
            try:
                result = self._easyocr.readtext(rgb, detail=1, paragraph=False)
            except Exception as exc:  # noqa: BLE001
                raise OcrEngineUnavailableError(f"easyocr failed: {exc}") from exc

            lines = []
            for row in result or []:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                text = str(row[1]).strip()
                if not text:
                    continue
                conf = float(row[2])
                lines.append((text, max(0.0, min(1.0, conf))))
            return lines

        if self._pytesseract is not None:
            try:
                data = self._pytesseract.image_to_data(
                    rgb,
                    lang="chi_sim+eng",
                    output_type=self._pytesseract_output.DICT,
                    config="--psm 6",
                )
            except Exception as exc:  # noqa: BLE001
                raise OcrEngineUnavailableError(
                    f"pytesseract failed (is Tesseract OCR installed?): {exc}"
                ) from exc

            lines = []
            texts = data.get("text", [])
            confs = data.get("conf", [])
            for idx, text in enumerate(texts):
                value = str(text).strip()
                if not value:
                    continue
                try:
                    conf_raw = float(confs[idx])
                except Exception:  # noqa: BLE001
                    conf_raw = 0.0
                if conf_raw <= 0:
                    continue
                lines.append((value, max(0.0, min(1.0, conf_raw / 100.0))))
            return lines

        raise OcrEngineUnavailableError(
            "No OCR backend available. Install one of: paddleocr, easyocr, "
            "or pytesseract + system tesseract OCR."
        )


_ocr_backend_lock = threading.Lock()
_ocr_backend_instance: _OcrBackend | None = None


def _get_ocr_backend() -> _OcrBackend:
    global _ocr_backend_instance
    with _ocr_backend_lock:
        if _ocr_backend_instance is None:
            _ocr_backend_instance = _OcrBackend()
        return _ocr_backend_instance


def _decode_image(image_bytes: bytes) -> Any:
    if not image_bytes:
        raise OcrInputError("empty image payload")

    cv2, np = _get_cv2_np()
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise OcrInputError("failed to decode image")
    return image


def _resize_for_detection(image_bgr: Any, target_width: int = _TARGET_WIDTH) -> Any:
    cv2, _ = _get_cv2_np()
    h, w = image_bgr.shape[:2]
    if w <= target_width:
        return image_bgr
    scale = target_width / float(w)
    new_size = (target_width, int(h * scale))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def _order_quad_points(quad: Any) -> Any:
    _, np = _get_cv2_np()
    points = quad.astype(np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_to_bbox(quad: Any) -> tuple[int, int, int, int]:
    _, np = _get_cv2_np()
    xs = quad[:, 0]
    ys = quad[:, 1]
    x = int(np.floor(xs.min()))
    y = int(np.floor(ys.min()))
    w = int(np.ceil(xs.max() - xs.min()))
    h = int(np.ceil(ys.max() - ys.min()))
    return (x, y, max(1, w), max(1, h))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _fallback_cards(image_bgr: Any) -> list[CardRegion]:
    _, np = _get_cv2_np()
    h, w = image_bgr.shape[:2]
    y = int(h * 0.12)
    card_h = int(h * 0.82)
    card_w = int(w * 0.27)
    gap = max(8, int((w - card_w * 3) / 4))

    cards: list[CardRegion] = []
    for idx in range(3):
        x = gap + idx * (card_w + gap)
        x = max(0, min(x, w - card_w))
        quad = np.array(
            [
                [x, y],
                [x + card_w, y],
                [x + card_w, y + card_h],
                [x, y + card_h],
            ],
            dtype=np.float32,
        )
        cards.append(CardRegion(quad=quad, bbox=(x, y, card_w, card_h), score=0.0))
    return cards


def _detect_cards(image_bgr: Any) -> list[CardRegion]:
    cv2, np = _get_cv2_np()
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[CardRegion] = []

    for cnt in contours:
        cnt_area = float(cv2.contourArea(cnt))
        if cnt_area < img_area * 0.01:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            quad = cv2.boxPoints(rect).astype(np.float32)

        quad = _order_quad_points(quad)
        x, y, bw, bh = _quad_to_bbox(quad)

        if x < 0 or y < 0 or x + bw > w or y + bh > h:
            continue

        box_area = float(bw * bh)
        if box_area < img_area * 0.035 or box_area > img_area * 0.75:
            continue

        aspect = bh / max(1.0, float(bw))
        if aspect < 1.15 or aspect > 3.8:
            continue

        contour_area = abs(float(cv2.contourArea(quad)))
        solidity = min(1.0, contour_area / max(1.0, box_area))
        center_penalty = abs((x + bw * 0.5) - w * 0.5) / max(1.0, w * 0.5)
        score = (
            (box_area / img_area) * 3.0
            + max(0.0, 1.0 - abs(aspect - 2.1) / 2.1)
            + solidity
            - center_penalty * 0.2
        )

        candidates.append(CardRegion(quad=quad, bbox=(x, y, bw, bh), score=score))

    candidates.sort(key=lambda item: item.score, reverse=True)

    selected: list[CardRegion] = []
    for candidate in candidates:
        if any(_bbox_iou(candidate.bbox, item.bbox) > 0.35 for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= 6:
            break

    if not selected:
        return _fallback_cards(image_bgr)

    selected.sort(key=lambda item: item.bbox[0])

    if len(selected) >= 3:
        return selected[:3]

    fallback = _fallback_cards(image_bgr)
    merged = selected[:]
    for item in fallback:
        if any(_bbox_iou(item.bbox, existing.bbox) > 0.5 for existing in merged):
            continue
        merged.append(item)
        if len(merged) >= 3:
            break

    merged.sort(key=lambda item: item.bbox[0])
    while len(merged) < 3:
        merged.append(fallback[len(merged)])
    return merged[:3]


def _warp_card(image_bgr: Any, quad: Any) -> Any:
    cv2, np = _get_cv2_np()
    rect = _order_quad_points(quad)

    width_a = np.linalg.norm(rect[2] - rect[3])
    width_b = np.linalg.norm(rect[1] - rect[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(rect[1] - rect[2])
    height_b = np.linalg.norm(rect[0] - rect[3])
    max_height = int(max(height_a, height_b))

    max_width = max(80, max_width)
    max_height = max(120, max_height)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))
    return warped


def _crop_title_regions(card_bgr: Any) -> list[Any]:
    cv2, _ = _get_cv2_np()
    h, w = card_bgr.shape[:2]
    regions: list[Any] = []

    for left, top, right, bottom in _TITLE_CROP_RATIOS:
        x1 = int(w * left)
        x2 = int(w * right)
        y1 = int(h * top)
        y2 = int(h * bottom)
        if x2 <= x1 or y2 <= y1:
            continue
        region = card_bgr[y1:y2, x1:x2]
        if region.size == 0:
            continue

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        denoise = cv2.bilateralFilter(gray, 5, 40, 40)
        min_side = min(denoise.shape[:2])
        block_size = max(3, min(31, (min_side // 2) * 2 + 1))
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(
            denoise,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            11,
        )

        region_bgr = region.copy()
        gray_bgr = cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        regions.extend([region_bgr, gray_bgr, thresh_bgr])

    if not regions:
        regions.append(card_bgr)
    return regions


def _clean_text(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return text


def _pick_best_ocr_line(lines: list[tuple[str, float]]) -> tuple[str, float, list[str]]:
    best_text = ""
    best_conf = 0.0
    best_score = -999.0
    raw_lines: list[str] = []

    for raw_text, conf in lines:
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue

        raw_lines.append(cleaned)
        han_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
        length_penalty = abs(len(cleaned) - 4) * 0.03
        score = conf + min(0.35, han_count * 0.06) - length_penalty

        if score > best_score:
            best_score = score
            best_text = cleaned
            best_conf = max(0.0, min(1.0, conf))

    return best_text, best_conf, raw_lines


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.strip().lower())


def _match_name_fuzzy(text: str, names: list[str]) -> tuple[str, float]:
    normalized_text = _normalize_for_match(text)
    if not normalized_text or not names:
        return "", 0.0

    normalized_names = [_normalize_for_match(name) for name in names]

    try:
        from rapidfuzz import fuzz, process  # type: ignore

        result = process.extractOne(
            normalized_text,
            normalized_names,
            scorer=fuzz.WRatio,
        )
        if result is not None:
            _, score, index = result
            if 0 <= index < len(names):
                return names[index], max(0.0, min(1.0, float(score) / 100.0))
    except Exception:  # noqa: BLE001
        pass

    best_index = -1
    best_ratio = 0.0
    for idx, candidate in enumerate(normalized_names):
        ratio = difflib.SequenceMatcher(None, normalized_text, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = idx

    if best_index >= 0:
        return names[best_index], max(0.0, min(1.0, best_ratio))
    return "", 0.0


def _build_overlay(image_bgr: Any, cards: list[CardRegion], positions: list[str]) -> Any:
    cv2, _ = _get_cv2_np()
    overlay = image_bgr.copy()
    for idx, card in enumerate(cards):
        points = card.quad.astype(int).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], True, (76, 175, 80), 2)
        x, y, _, _ = card.bbox
        label = positions[idx] if idx < len(positions) else f"card{idx + 1}"
        cv2.putText(
            overlay,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def resolve_augment_names(
    names: list[str],
    augment_mapping: dict[str, dict[str, str | None]],
) -> list[dict[str, Any]]:
    names_source: list[str] = []
    name_to_id: dict[str, int] = {}
    normalized_to_name: dict[str, str] = {}

    for key, value in augment_mapping.items():
        if not isinstance(value, dict):
            continue
        name = value.get("name_cn")
        if not isinstance(name, str) or not name.strip():
            continue
        clean_name = name.strip()
        names_source.append(clean_name)
        if clean_name not in name_to_id:
            try:
                name_to_id[clean_name] = int(value.get("augmentId") or key)
            except Exception:  # noqa: BLE001
                continue
        normalized_to_name[_normalize_for_match(clean_name)] = clean_name

    unique_names = list(dict.fromkeys(names_source))
    resolved: list[dict[str, Any]] = []

    for raw in names:
        original = str(raw or "").strip()
        if not original:
            resolved.append(
                {"name_cn": original, "augmentId": None, "confidence": 0.0}
            )
            continue

        normalized = _normalize_for_match(original)
        exact_name = normalized_to_name.get(normalized)
        if exact_name and exact_name in name_to_id:
            resolved.append(
                {
                    "name_cn": original,
                    "augmentId": int(name_to_id[exact_name]),
                    "confidence": 1.0,
                }
            )
            continue

        matched_name, fuzzy_score = _match_name_fuzzy(original, unique_names)
        if matched_name and matched_name in name_to_id:
            resolved.append(
                {
                    "name_cn": original,
                    "augmentId": int(name_to_id[matched_name]),
                    "confidence": round(max(0.0, min(1.0, fuzzy_score)), 2),
                }
            )
        else:
            resolved.append(
                {"name_cn": original, "augmentId": None, "confidence": 0.0}
            )

    return resolved


def recognize_augment_names(
    image_bytes: bytes,
    augment_mapping: dict[str, dict[str, str | None]],
    keep_debug_images: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    backend = _get_ocr_backend()

    image = _decode_image(image_bytes)
    image = _resize_for_detection(image)

    cards = _detect_cards(image)
    cards.sort(key=lambda item: item.bbox[0])

    names_source = []
    for value in augment_mapping.values():
        name = value.get("name_cn") if isinstance(value, dict) else None
        if isinstance(name, str) and name.strip():
            names_source.append(name.strip())

    unique_names = list(dict.fromkeys(names_source))

    names_out: list[dict[str, Any]] = []
    card_debug: list[dict[str, int]] = []
    raw_debug: list[dict[str, Any]] = []

    card_images: list[Any] = []
    title_images: list[Any] = []

    for idx, card in enumerate(cards[:3]):
        position = _POSITION_LABELS[idx]
        x, y, w, h = card.bbox
        card_debug.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

        warped = _warp_card(image, card.quad)
        card_images.append(warped)

        ocr_lines: list[tuple[str, float]] = []
        best_title_img = None
        best_text = ""
        best_conf = 0.0
        collected_raw: list[str] = []

        for region in _crop_title_regions(warped):
            lines = backend.read_lines(region)
            ocr_lines.extend(lines)
            text, conf, raw_lines = _pick_best_ocr_line(lines)
            if raw_lines:
                collected_raw.extend(raw_lines)
            if text and conf >= best_conf:
                best_text = text
                best_conf = conf
                best_title_img = region

        if best_title_img is None:
            best_title_img = _crop_title_regions(warped)[0]
        title_images.append(best_title_img)

        matched_name, fuzzy_score = _match_name_fuzzy(best_text, unique_names)
        if matched_name:
            name_cn = matched_name
        else:
            name_cn = best_text or ""

        confidence = 0.0
        if name_cn:
            confidence = max(0.0, min(1.0, best_conf * 0.45 + fuzzy_score * 0.55))

        names_out.append(
            {
                "position": position,
                "name_cn": name_cn,
                "confidence": round(confidence, 2),
            }
        )

        raw_debug.append(
            {
                "position": position,
                "best_text": best_text,
                "best_text_conf": round(best_conf, 3),
                "matched_name": matched_name,
                "match_score": round(fuzzy_score, 3),
                "raw_lines": collected_raw,
            }
        )

    # Pad result to fixed 3 positions.
    while len(names_out) < 3:
        position = _POSITION_LABELS[len(names_out)]
        names_out.append({"position": position, "name_cn": "", "confidence": 0.0})

    overlay = _build_overlay(image, cards[:3], list(_POSITION_LABELS))

    result = {
        "names": names_out,
        "debug": {
            "cards": card_debug,
            "engine": backend.name,
            "ocr": raw_debug,
        },
    }

    artifacts = None
    if keep_debug_images:
        artifacts = {
            "overlay": overlay,
            "cards": card_images,
            "titles": title_images,
        }

    return result, artifacts
