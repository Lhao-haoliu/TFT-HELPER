from __future__ import annotations

import json
import re
import socket
import time
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

C_DRAGON_AUGMENTS_URLS = (
    "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/zh_cn/v1/cherry-augments.json",
)
DEFAULT_TIMEOUT_SECONDS = 8.0
CACHE_TTL_SECONDS = 24 * 60 * 60
_USER_AGENT = "Mozilla/5.0 (tft-helper)"
_ID_SUFFIX_PATTERN = re.compile(r"(\d+)$")
_RARITY_UNKNOWN = "unknown"


class AugmentMappingError(Exception):
    """Raised when augment mapping cannot be fetched or parsed."""


_cache_lock = Lock()
_cached_mapping: dict[str, dict[str, str | None]] | None = None
_cache_timestamp: float = 0.0


def _icon_to_public_url(icon_path: str | None) -> str | None:
    if not icon_path:
        return None
    cleaned = icon_path.strip()
    if not cleaned:
        return None

    normalized = cleaned.replace("\\", "/").strip()
    lower = normalized.lower()

    prefixes = (
        "/lol-game-data/assets/assets/",
        "/lol-game-data/assets/",
    )
    suffix = None
    for prefix in prefixes:
        if lower.startswith(prefix):
            suffix = lower[len(prefix) :]
            break

    if suffix:
        return (
            "https://raw.communitydragon.org/latest/"
            f"plugins/rcp-be-lol-game-data/global/default/assets/{suffix}"
        )
    return None


def _normalize_numeric_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            return str(int(cleaned))
    return None


def _normalize_rarity(value: object) -> str:
    if value is None:
        return _RARITY_UNKNOWN
    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric >= 3:
            return "prismatic"
        if numeric == 2:
            return "gold"
        if numeric == 1:
            return "silver"
        return _RARITY_UNKNOWN

    if not isinstance(value, str):
        return _RARITY_UNKNOWN

    text = value.strip().lower()
    if not text:
        return _RARITY_UNKNOWN

    if "prismatic" in text or "kprismatic" in text or "棱彩" in text:
        return "prismatic"
    if "gold" in text or "kgold" in text or "黄金" in text:
        return "gold"
    if "silver" in text or "ksilver" in text or "白银" in text:
        return "silver"
    return _RARITY_UNKNOWN


def _extract_rarity(raw: dict[str, object]) -> str:
    for key in (
        "rarity",
        "rarityName",
        "rarityType",
        "quality",
        "tier",
        "color",
        "rarityColor",
        "tierColor",
    ):
        rarity = _normalize_rarity(raw.get(key))
        if rarity != _RARITY_UNKNOWN:
            return rarity
    return _RARITY_UNKNOWN


def _extract_augment_id(raw: dict[str, object]) -> str | None:
    for key in ("id", "augmentId"):
        normalized = _normalize_numeric_id(raw.get(key))
        if normalized is not None:
            return normalized

    for key in ("apiName", "nameId", "name", "displayName"):
        value = raw.get(key)
        if isinstance(value, str):
            match = _ID_SUFFIX_PATTERN.search(value.strip())
            if match:
                return str(int(match.group(1)))

    return None


def _extract_localized_text(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, dict):
        for key in ("zh_cn", "zh-CN", "zh", "default", "en_us"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _normalize_item(raw: object) -> tuple[str, dict[str, str | None]] | None:
    if not isinstance(raw, dict):
        return None

    augment_id = _extract_augment_id(raw)
    if augment_id is None:
        return None

    name_cn = None
    for key in ("name", "nameTRA", "displayName", "title"):
        name_cn = _extract_localized_text(raw.get(key))
        if name_cn:
            break
    if not name_cn:
        return None

    desc_cn = None
    for key in ("description", "desc", "tooltip", "descTRA"):
        desc_cn = _extract_localized_text(raw.get(key))
        if desc_cn:
            break

    icon_path = None
    for key in ("augmentSmallIconPath", "iconSmall", "icon", "iconPath"):
        maybe_icon = raw.get(key)
        if isinstance(maybe_icon, str) and maybe_icon.strip():
            icon_path = maybe_icon.strip()
            break

    icon = _icon_to_public_url(icon_path)
    rarity = _extract_rarity(raw)
    return augment_id, {
        "augmentId": augment_id,
        "name_cn": name_cn,
        "desc_cn": desc_cn,
        "icon": icon,
        "rarity": rarity,
    }


def _fetch_remote(timeout_seconds: float) -> dict[str, dict[str, str | None]]:
    last_error: Exception | None = None

    for source_url in C_DRAGON_AUGMENTS_URLS:
        request = Request(source_url, headers={"User-Agent": _USER_AGENT})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status != 200:
                    last_error = AugmentMappingError(
                        f"cdragon returned non-200 status: {status}"
                    )
                    continue
                payload = json.load(response)
        except json.JSONDecodeError as exc:
            last_error = AugmentMappingError("failed to parse cdragon JSON payload")
            last_error.__cause__ = exc
            continue
        except HTTPError as exc:
            last_error = AugmentMappingError(f"cdragon returned HTTP error: {exc.code}")
            continue
        except URLError as exc:
            last_error = AugmentMappingError(f"failed to reach cdragon: {exc.reason}")
            continue
        except socket.timeout:
            last_error = AugmentMappingError("request to cdragon timed out")
            continue
        except TimeoutError:
            last_error = AugmentMappingError("request to cdragon timed out")
            continue

        if not isinstance(payload, list):
            last_error = AugmentMappingError("unexpected cdragon payload shape: expected list")
            continue

        mapping: dict[str, dict[str, str | None]] = {}
        for raw in payload:
            normalized = _normalize_item(raw)
            if normalized is None:
                continue
            augment_id, item = normalized
            mapping[augment_id] = item

        if mapping:
            return mapping

        last_error = AugmentMappingError("no augment mapping entries found in cdragon payload")

    raise last_error or AugmentMappingError("failed to fetch cdragon augment mapping")


def get_augment_mapping(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    force_refresh: bool = False,
) -> dict[str, dict[str, str | None]]:
    global _cached_mapping, _cache_timestamp

    now = time.time()
    with _cache_lock:
        if (
            not force_refresh
            and _cached_mapping is not None
            and now - _cache_timestamp < CACHE_TTL_SECONDS
        ):
            return _cached_mapping

    try:
        mapping = _fetch_remote(timeout_seconds=timeout_seconds)
        with _cache_lock:
            _cached_mapping = mapping
            _cache_timestamp = now
        return mapping
    except AugmentMappingError:
        with _cache_lock:
            if _cached_mapping is not None:
                return _cached_mapping
        raise
