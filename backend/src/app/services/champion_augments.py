from __future__ import annotations

import html as html_lib
import json
import re
import socket
import time
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DTODO_CHAMPION_PAGE_URL = "https://hextech.dtodo.cn/zh-CN/champion-stats/{champion_id}"
DEFAULT_TIMEOUT_SECONDS = 10.0
CACHE_TTL_SECONDS = 5 * 60
_USER_AGENT = "Mozilla/5.0 (tft-helper)"

_NEXT_AUGMENTS_PATTERN = re.compile(
    r'self\.__next_f\.push\(\[1,"(?P<payload>\{\\\"augments\\\"[\s\S]*?)"\]\)'
)
_AUGMENT_ROW_PATTERN = re.compile(
    r"#(?P<augment_id>\d+)\s*T\s*(?P<tier>\d+)\s*(?P<win>\d+(?:\.\d+)?)%\s*"
    r"(?P<pick>\d+(?:\.\d+)?)%(?:\s+(?P<games>(?:\d{1,3}(?:,\d{3})+|\d{4,})))?"
)
_SECTION_START_MARKERS = ("海克斯推荐", "推薦海克斯", "Recommended Augments")
_SECTION_END_MARKERS = (
    "推荐海克斯组合 TOP 10",
    "推薦海克斯組合 TOP 10",
    "Recommended Augment Combos TOP 10",
    "推荐海克斯组合",
    "推薦海克斯組合",
)


class ChampionAugmentsError(Exception):
    """Raised when champion augment stats cannot be fetched or parsed."""


_cache_lock = Lock()
_cached_by_champion: dict[str, tuple[float, list[dict[str, int | float | str | None]]]] = {}


def _normalize_rate(value: object) -> float:
    rate = float(value)
    if rate > 1:
        rate = rate / 100.0
    return rate


def _normalize_tier(value: object) -> str:
    raw = str(value).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        raise ValueError("missing tier")
    return f"T{int(digits)}"


def _parse_optional_games(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value >= 0:
            return int(value)
        return None
    cleaned = str(value).replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _fetch_champion_page(champion_id: str, timeout_seconds: float) -> str:
    url = DTODO_CHAMPION_PAGE_URL.format(champion_id=champion_id)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise ChampionAugmentsError(f"dtodo returned non-200 status: {status}")
            return response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        raise ChampionAugmentsError(f"dtodo returned HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise ChampionAugmentsError(f"failed to reach dtodo: {exc.reason}") from exc
    except socket.timeout as exc:
        raise ChampionAugmentsError("request to dtodo timed out") from exc
    except TimeoutError as exc:
        raise ChampionAugmentsError("request to dtodo timed out") from exc


def _parse_from_next_payload(page_html: str) -> list[dict[str, int | float | str | None]]:
    match = _NEXT_AUGMENTS_PATTERN.search(page_html)
    if not match:
        raise ValueError("embedded augment payload not found")

    escaped_json = match.group("payload")
    decoded_json = json.loads(f'"{escaped_json}"')
    payload = json.loads(decoded_json)
    augments = payload.get("augments")
    if not isinstance(augments, dict):
        raise ValueError("embedded augment payload shape is invalid")

    dedup: dict[int, dict[str, int | float | str | None]] = {}
    for augment_id_raw, stats_raw in augments.items():
        if not isinstance(stats_raw, dict):
            continue
        try:
            augment_id = int(str(augment_id_raw).strip())
            tier = _normalize_tier(stats_raw.get("tier"))
            win_rate = _normalize_rate(stats_raw.get("win_rate"))
            pick_rate = _normalize_rate(stats_raw.get("pick_rate"))
            games = _parse_optional_games(stats_raw.get("num_games"))
        except (TypeError, ValueError):
            continue

        dedup[augment_id] = {
            "augmentId": augment_id,
            "tier": tier,
            "winRate": win_rate,
            "pickRate": pick_rate,
            "games": games,
        }

    if not dedup:
        raise ValueError("no augment entries in embedded payload")

    return list(dedup.values())


def _html_to_text(page_html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", page_html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return " ".join(text.split())


def _slice_augment_section(page_text: str) -> str:
    start = -1
    for marker in _SECTION_START_MARKERS:
        idx = page_text.find(marker)
        if idx != -1:
            start = idx
            break

    if start == -1:
        return page_text

    end = len(page_text)
    for marker in _SECTION_END_MARKERS:
        idx = page_text.find(marker, start)
        if idx != -1:
            end = min(end, idx)

    return page_text[start:end]


def _parse_from_text_fallback(page_html: str) -> list[dict[str, int | float | str | None]]:
    page_text = _html_to_text(page_html)
    section = _slice_augment_section(page_text)

    dedup: dict[int, dict[str, int | float | str | None]] = {}
    for match in _AUGMENT_ROW_PATTERN.finditer(section):
        augment_id = int(match.group("augment_id"))
        tier = f"T{int(match.group('tier'))}"
        win_rate = float(match.group("win")) / 100.0
        pick_rate = float(match.group("pick")) / 100.0
        games_raw = match.group("games")
        games = int(games_raw.replace(",", "")) if games_raw else None
        dedup[augment_id] = {
            "augmentId": augment_id,
            "tier": tier,
            "winRate": win_rate,
            "pickRate": pick_rate,
            "games": games,
        }

    if not dedup:
        raise ValueError("no augment entries in text fallback parser")

    return list(dedup.values())


def get_champion_augment_stats(
    champion_id: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, int | float | str | None]]:
    normalized_champion_id = str(champion_id).strip()
    if not normalized_champion_id:
        raise ChampionAugmentsError("championId is required")

    now = time.time()
    with _cache_lock:
        cached = _cached_by_champion.get(normalized_champion_id)
        if cached is not None:
            cached_at, cached_rows = cached
            if now - cached_at < CACHE_TTL_SECONDS:
                return cached_rows

    page_html = _fetch_champion_page(
        champion_id=normalized_champion_id, timeout_seconds=timeout_seconds
    )
    try:
        rows = _parse_from_next_payload(page_html)
    except (ValueError, json.JSONDecodeError):
        try:
            rows = _parse_from_text_fallback(page_html)
        except ValueError as exc:
            raise ChampionAugmentsError(
                "failed to parse champion augment stats from dtodo page"
            ) from exc

    with _cache_lock:
        _cached_by_champion[normalized_champion_id] = (now, rows)

    return rows
