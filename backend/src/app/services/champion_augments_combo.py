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

_COMBO_PATTERN = re.compile(
    r"(?P<rank>\d+)\s*#(?P<a>\d+)\s*#(?P<b>\d+)\s*#(?P<c>\d+)\s*T\s*(?P<tier>\d+)",
    re.S,
)
_NEXT_AUGMENTS_PATTERN = re.compile(
    r'self\.__next_f\.push\(\[1,"(?P<payload>\{\\\"augments\\\"[\s\S]*?)"\]\)'
)
_VERSION_PATTERN = re.compile(r"\b(?P<version>\d{2}\.\d)\b")
_SECTION_START_MARKERS = (
    "推荐海克斯组合 TOP 10",
    "推薦海克斯組合 TOP 10",
    "Recommended Augment Combos TOP 10",
)
_SECTION_END_MARKERS = ("海克斯推荐", "推薦海克斯", "Recommended Augments")


class ChampionAugmentCombosError(Exception):
    """Raised when champion augment combos cannot be fetched or parsed."""


_cache_lock = Lock()
_cached_by_champion: dict[str, tuple[float, dict[str, object]]] = {}


def _fetch_champion_page(champion_id: str, timeout_seconds: float) -> str:
    url = DTODO_CHAMPION_PAGE_URL.format(champion_id=champion_id)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise ChampionAugmentCombosError(f"dtodo returned non-200 status: {status}")
            return response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        raise ChampionAugmentCombosError(f"dtodo returned HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise ChampionAugmentCombosError(f"failed to reach dtodo: {exc.reason}") from exc
    except socket.timeout as exc:
        raise ChampionAugmentCombosError("request to dtodo timed out") from exc
    except TimeoutError as exc:
        raise ChampionAugmentCombosError("request to dtodo timed out") from exc


def _html_to_text(page_html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", page_html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return " ".join(text.split())


def _slice_combo_section(page_text: str) -> str:
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
        idx = page_text.find(marker, start + 1)
        if idx != -1:
            end = min(end, idx)
    return page_text[start:end]


def _combo_key(augment_ids: list[int], tier: str) -> tuple[tuple[int, int, int], str]:
    sorted_ids = tuple(sorted(augment_ids))
    return sorted_ids, tier


def _parse_top10_from_text(page_text: str) -> list[dict[str, object]]:
    section = _slice_combo_section(page_text)
    parsed: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int, int], str]] = set()

    for match in _COMBO_PATTERN.finditer(section):
        augment_ids = [
            int(match.group("a")),
            int(match.group("b")),
            int(match.group("c")),
        ]
        tier = f"T{int(match.group('tier'))}"
        key = _combo_key(augment_ids, tier)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(
            {
                "rank": int(match.group("rank")),
                "tier": tier,
                "augmentIds": augment_ids,
            }
        )

    parsed.sort(key=lambda item: int(item["rank"]))
    return parsed


def _parse_all_from_next_payload(page_html: str) -> list[dict[str, object]]:
    match = _NEXT_AUGMENTS_PATTERN.search(page_html)
    if not match:
        return []

    decoded_json = json.loads(f'"{match.group("payload")}"')
    payload = json.loads(decoded_json)
    augment_trios = payload.get("augment_trios")
    if not isinstance(augment_trios, dict):
        return []

    combos: list[dict[str, object]] = []
    for combo_key, stats in augment_trios.items():
        if not isinstance(combo_key, str) or not isinstance(stats, dict):
            continue
        parts = combo_key.split(":")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        augment_ids = [int(part) for part in parts]
        tier_raw = stats.get("win_rate_tier") or stats.get("pick_rate_tier")
        if tier_raw is None:
            continue
        try:
            tier = f"T{int(str(tier_raw).strip())}"
        except ValueError:
            continue
        try:
            games = int(str(stats.get("num_games", "0")).replace(",", ""))
        except ValueError:
            games = 0
        combos.append({"tier": tier, "augmentIds": augment_ids, "games": games})

    combos.sort(
        key=lambda item: (
            int(str(item["tier"]).lstrip("T") or "99"),
            -int(item["games"]),
            tuple(sorted(item["augmentIds"])),
        )
    )
    return combos


def _extract_version(page_html: str, page_text: str) -> str | None:
    for source in (page_text, page_html):
        match = _VERSION_PATTERN.search(source)
        if match:
            return match.group("version")
    return None


def get_champion_augment_combos(
    champion_id: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    normalized_champion_id = str(champion_id).strip()
    if not normalized_champion_id:
        raise ChampionAugmentCombosError("championId is required")

    now = time.time()
    with _cache_lock:
        cached = _cached_by_champion.get(normalized_champion_id)
        if cached is not None:
            cached_at, payload = cached
            if now - cached_at < CACHE_TTL_SECONDS:
                return payload

    page_html = _fetch_champion_page(normalized_champion_id, timeout_seconds)
    page_text = _html_to_text(page_html)

    top10 = _parse_top10_from_text(page_text)
    all_from_payload = _parse_all_from_next_payload(page_html)

    merged: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int, int], str]] = set()
    for combo in top10:
        key = _combo_key(combo["augmentIds"], str(combo["tier"]))
        seen.add(key)
        merged.append(combo)

    next_rank = len(merged) + 1
    for combo in all_from_payload:
        key = _combo_key(combo["augmentIds"], str(combo["tier"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "rank": next_rank,
                "tier": combo["tier"],
                "augmentIds": combo["augmentIds"],
            }
        )
        next_rank += 1

    if not merged:
        raise ChampionAugmentCombosError(
            "failed to parse champion augment combos from dtodo page"
        )

    payload: dict[str, object] = {
        "championId": int(normalized_champion_id)
        if normalized_champion_id.isdigit()
        else normalized_champion_id,
        "version": _extract_version(page_html, page_text),
        "combos": merged,
    }

    with _cache_lock:
        _cached_by_champion[normalized_champion_id] = (now, payload)

    return payload
