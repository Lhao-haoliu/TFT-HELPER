from __future__ import annotations

from itertools import combinations
from typing import Any


class BattleRecommendError(Exception):
    """Raised when battle recommendation input is invalid."""


def _normalize_id_list(raw: list[int | str]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for item in raw:
        try:
            value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_str_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _build_augment_stats_index(
    champion_augments: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for row in champion_augments:
        try:
            augment_id = int(row.get("augmentId"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        index[augment_id] = row
    return index


def _build_candidates(
    candidate_ids: list[int],
    augment_stats_index: dict[int, dict[str, Any]],
    augment_mapping: dict[str, dict[str, str | None]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for augment_id in candidate_ids:
        stats = augment_stats_index.get(augment_id, {})
        mapped = augment_mapping.get(str(augment_id), {})
        name_cn = _as_str_or_default(
            mapped.get("name_cn") if isinstance(mapped, dict) else None,
            f"#{augment_id}",
        )
        tier = _as_str_or_default(stats.get("tier"), "-")
        rarity = _as_str_or_default(
            mapped.get("rarity") if isinstance(mapped, dict) else None,
            "unknown",
        )

        out.append(
            {
                "augmentId": augment_id,
                "name_cn": name_cn,
                "winRate": _as_float_or_none(stats.get("winRate")),
                "pickRate": _as_float_or_none(stats.get("pickRate")),
                "tier": tier,
                "rarity": rarity,
            }
        )
    return out


def _top10_combos(champion_combos: dict[str, Any]) -> list[dict[str, Any]]:
    raw = champion_combos.get("combos")
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for combo in raw:
        if not isinstance(combo, dict):
            continue
        ids_raw = combo.get("augmentIds")
        if not isinstance(ids_raw, list):
            continue
        ids = _normalize_id_list(ids_raw)  # type: ignore[arg-type]
        if len(ids) != 3:
            continue
        try:
            rank = int(combo.get("rank"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rank = 999
        normalized.append({"rank": rank, "augmentIds": ids})

    normalized.sort(key=lambda item: int(item["rank"]))
    return normalized[:10]


def _combo_matches(
    *,
    round_number: int,
    picked_ids: list[int],
    candidate_id: int,
    combo_ids: list[int],
) -> bool:
    if candidate_id not in combo_ids:
        return False

    combo_set = set(combo_ids)
    picked = picked_ids[:]

    if round_number == 2:
        return len(picked) >= 1 and picked[0] in combo_set

    if round_number == 3:
        return len(picked) >= 2 and picked[0] in combo_set and picked[1] in combo_set

    if round_number >= 4:
        if len(picked) < 2:
            return False
        for left, right in combinations(picked, 2):
            if left in combo_set and right in combo_set:
                return True
        return False

    return False


def _find_single_recommendation(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (
            item["winRate"] if isinstance(item.get("winRate"), (int, float)) else -1.0,
            -int(item["augmentId"]),
        ),
    )
    return {"augmentId": int(best["augmentId"]), "reason": "winRate最高"}


def _find_combo_recommendation(
    *,
    candidates: list[dict[str, Any]],
    round_number: int,
    picked_ids: list[int],
    top10: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if round_number < 2 or not candidates or not top10:
        return None

    best: dict[str, Any] | None = None
    for candidate in candidates:
        augment_id = int(candidate["augmentId"])
        win_rate = (
            float(candidate["winRate"])
            if isinstance(candidate.get("winRate"), (int, float))
            else -1.0
        )
        for combo in top10:
            combo_ids = combo["augmentIds"]
            if not isinstance(combo_ids, list):
                continue
            if not _combo_matches(
                round_number=round_number,
                picked_ids=picked_ids,
                candidate_id=augment_id,
                combo_ids=combo_ids,
            ):
                continue

            current = {
                "augmentId": augment_id,
                "comboRank": int(combo["rank"]),
                "winRate": win_rate,
                "reason": "命中TOP10组合",
            }
            if best is None:
                best = current
                continue
            if current["comboRank"] < best["comboRank"]:
                best = current
                continue
            if (
                current["comboRank"] == best["comboRank"]
                and current["winRate"] > best["winRate"]
            ):
                best = current
                continue
            if (
                current["comboRank"] == best["comboRank"]
                and current["winRate"] == best["winRate"]
                and current["augmentId"] < best["augmentId"]
            ):
                best = current

    if best is None:
        return None

    return {
        "augmentId": int(best["augmentId"]),
        "comboRank": int(best["comboRank"]),
        "reason": str(best["reason"]),
    }


def build_battle_recommendation(
    *,
    round_number: int,
    picked_augment_ids: list[int | str],
    candidate_augment_ids: list[int | str],
    champion_augments: list[dict[str, Any]],
    champion_combos: dict[str, Any],
    augment_mapping: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    if round_number < 1 or round_number > 4:
        raise BattleRecommendError("round must be in range 1..4")

    picked_ids = _normalize_id_list(picked_augment_ids)
    candidate_ids = _normalize_id_list(candidate_augment_ids)

    augment_stats_index = _build_augment_stats_index(champion_augments)
    candidates = _build_candidates(candidate_ids, augment_stats_index, augment_mapping)
    top10 = _top10_combos(champion_combos)

    recommend_single = _find_single_recommendation(candidates)
    recommend_combo = _find_combo_recommendation(
        candidates=candidates,
        round_number=round_number,
        picked_ids=picked_ids,
        top10=top10,
    )

    return {
        "candidates": candidates,
        "recommendSingle": recommend_single,
        "recommendCombo": recommend_combo,
        "comboSource": "TOP10",
    }

