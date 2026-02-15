from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DTODO_CHAMPION_STATS_URL = "https://hextech.dtodo.cn/data/champions-stats.json"
DEFAULT_TIMEOUT_SECONDS = 8.0


class DtodoClientError(Exception):
    """Raised when dtodo data cannot be fetched or parsed."""


@dataclass(frozen=True)
class ChampionStats:
    championId: str
    winRate: float
    pickRate: float
    matches: int | None = None


_cache_lock = Lock()
_cached_stats: list[dict[str, str | float | int | None]] | None = None


def _normalize_item(raw: object) -> dict[str, str | float | int | None]:
    if not isinstance(raw, dict):
        raise ValueError("item is not an object")

    champion_id = str(raw["championId"])
    win_rate = float(raw["winRate"])
    pick_rate = float(raw["pickRate"])

    matches_raw = raw.get("numGames")
    matches = int(matches_raw) if matches_raw is not None else None

    return ChampionStats(
        championId=champion_id,
        winRate=win_rate,
        pickRate=pick_rate,
        matches=matches,
    ).__dict__


def _fetch_remote(timeout_seconds: float) -> list[dict[str, str | float | int | None]]:
    try:
        with urlopen(DTODO_CHAMPION_STATS_URL, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise DtodoClientError(f"dtodo returned non-200 status: {status}")

            try:
                payload = json.load(response)
            except json.JSONDecodeError as exc:
                raise DtodoClientError("failed to parse dtodo JSON payload") from exc
    except HTTPError as exc:
        raise DtodoClientError(f"dtodo returned HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise DtodoClientError(f"failed to reach dtodo: {exc.reason}") from exc
    except socket.timeout as exc:
        raise DtodoClientError("request to dtodo timed out") from exc
    except TimeoutError as exc:
        raise DtodoClientError("request to dtodo timed out") from exc

    if not isinstance(payload, list):
        raise DtodoClientError("unexpected dtodo payload shape: expected list")

    normalized: list[dict[str, str | float | int | None]] = []
    for idx, item in enumerate(payload):
        try:
            normalized.append(_normalize_item(item))
        except (KeyError, TypeError, ValueError) as exc:
            raise DtodoClientError(
                f"invalid champion stats at index {idx}: {exc}"
            ) from exc

    if not normalized:
        raise DtodoClientError("dtodo payload is empty")

    return normalized


def get_champion_stats(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, str | float | int | None]]:
    global _cached_stats

    try:
        stats = _fetch_remote(timeout_seconds=timeout_seconds)
        with _cache_lock:
            _cached_stats = stats
        return stats
    except DtodoClientError:
        with _cache_lock:
            if _cached_stats is not None:
                return _cached_stats
        raise
