from __future__ import annotations

import re
import socket
from html.parser import HTMLParser
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DTODO_ZH_CN_URL = "https://hextech.dtodo.cn/zh-CN"
DEFAULT_TIMEOUT_SECONDS = 8.0
_HREF_ID_PATTERN = re.compile(r"/zh-CN/champion-stats/([^/?#]+)")


class ChampionMappingError(Exception):
    """Raised when champion mapping cannot be fetched or parsed."""


class _ChampionLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_target_anchor = False
        self._current_href = ""
        self._text_parts: list[str] = []
        self.entries: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        href = ""
        for key, value in attrs:
            if key.lower() == "href" and value:
                href = value
                break

        if "/zh-CN/champion-stats/" not in href:
            return

        self._in_target_anchor = True
        self._current_href = href
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_target_anchor:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._in_target_anchor:
            return

        text = " ".join(part.strip() for part in self._text_parts if part.strip())
        if text:
            self.entries.append((self._current_href, text))

        self._in_target_anchor = False
        self._current_href = ""
        self._text_parts = []


_cache_lock = Lock()
_cached_mapping: dict[str, dict[str, str]] | None = None


def _parse_name_parts(raw_text: str) -> tuple[str, str]:
    cleaned = " ".join(raw_text.split())
    if not cleaned:
        raise ValueError("empty name text")

    parts = cleaned.split(" ")
    if len(parts) == 1:
        return cleaned, cleaned
    return cleaned, parts[-1]


def _extract_champion_id(href: str) -> str | None:
    match = _HREF_ID_PATTERN.search(href)
    if not match:
        return None
    return match.group(1).strip()


def _merge_record(
    existing: dict[str, str] | None, fullname_cn: str, name_cn: str
) -> dict[str, str]:
    if existing is None:
        return {"name_cn": name_cn, "fullname_cn": fullname_cn}

    # Prefer fuller display name, but keep a concise short name.
    best_fullname = (
        fullname_cn if len(fullname_cn) > len(existing["fullname_cn"]) else existing["fullname_cn"]
    )
    best_name = name_cn if len(name_cn) <= len(existing["name_cn"]) else existing["name_cn"]
    return {"name_cn": best_name, "fullname_cn": best_fullname}


def _fetch_remote(timeout_seconds: float) -> dict[str, dict[str, str]]:
    try:
        with urlopen(DTODO_ZH_CN_URL, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise ChampionMappingError(f"dtodo returned non-200 status: {status}")
            html = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        raise ChampionMappingError(f"dtodo returned HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise ChampionMappingError(f"failed to reach dtodo: {exc.reason}") from exc
    except socket.timeout as exc:
        raise ChampionMappingError("request to dtodo timed out") from exc
    except TimeoutError as exc:
        raise ChampionMappingError("request to dtodo timed out") from exc

    parser = _ChampionLinkParser()
    parser.feed(html)
    parser.close()

    mapping: dict[str, dict[str, str]] = {}
    for href, raw_text in parser.entries:
        champion_id = _extract_champion_id(href)
        if not champion_id:
            continue
        try:
            fullname_cn, name_cn = _parse_name_parts(raw_text)
        except ValueError:
            continue
        mapping[champion_id] = _merge_record(mapping.get(champion_id), fullname_cn, name_cn)

    if not mapping:
        raise ChampionMappingError("no champion mapping entries found in dtodo page")

    return mapping


def get_champion_mapping(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, dict[str, str]]:
    global _cached_mapping

    try:
        mapping = _fetch_remote(timeout_seconds=timeout_seconds)
        with _cache_lock:
            _cached_mapping = mapping
        return mapping
    except ChampionMappingError:
        with _cache_lock:
            if _cached_mapping is not None:
                return _cached_mapping
        raise
