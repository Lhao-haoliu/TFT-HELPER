from __future__ import annotations

from app.services.battle_recommend import build_battle_recommendation  # type: ignore[import-not-found]


def _mapping() -> dict[str, dict[str, str]]:
    return {
        "10": {"augmentId": "10", "name_cn": "A10", "rarity": "gold"},
        "20": {"augmentId": "20", "name_cn": "A20", "rarity": "gold"},
        "30": {"augmentId": "30", "name_cn": "A30", "rarity": "silver"},
        "40": {"augmentId": "40", "name_cn": "A40", "rarity": "prismatic"},
        "50": {"augmentId": "50", "name_cn": "A50", "rarity": "gold"},
        "60": {"augmentId": "60", "name_cn": "A60", "rarity": "gold"},
    }


def _augments() -> list[dict[str, object]]:
    return [
        {"augmentId": 40, "tier": "T1", "winRate": 0.64, "pickRate": 0.12},
        {"augmentId": 50, "tier": "T2", "winRate": 0.61, "pickRate": 0.11},
        {"augmentId": 60, "tier": "T2", "winRate": 0.58, "pickRate": 0.07},
    ]


def test_round2_combo_match() -> None:
    result = build_battle_recommendation(
        round_number=2,
        picked_augment_ids=[10],
        candidate_augment_ids=[40, 50, 60],
        champion_augments=_augments(),
        champion_combos={
            "combos": [
                {"rank": 2, "augmentIds": [10, 40, 99]},
                {"rank": 5, "augmentIds": [10, 50, 98]},
            ]
        },
        augment_mapping=_mapping(),
    )

    assert result["recommendSingle"]["augmentId"] == 40
    assert result["recommendCombo"]["augmentId"] == 40
    assert result["recommendCombo"]["comboRank"] == 2


def test_round3_requires_two_picks_plus_candidate() -> None:
    result = build_battle_recommendation(
        round_number=3,
        picked_augment_ids=[10, 20],
        candidate_augment_ids=[40, 50],
        champion_augments=_augments(),
        champion_combos={
            "combos": [
                {"rank": 1, "augmentIds": [10, 20, 50]},
                {"rank": 2, "augmentIds": [10, 40, 99]},
            ]
        },
        augment_mapping=_mapping(),
    )

    assert result["recommendCombo"]["augmentId"] == 50
    assert result["recommendCombo"]["comboRank"] == 1


def test_round4_uses_any_two_picked_plus_candidate_and_rank_priority() -> None:
    result = build_battle_recommendation(
        round_number=4,
        picked_augment_ids=[10, 20, 30],
        candidate_augment_ids=[40, 50],
        champion_augments=_augments(),
        champion_combos={
            "combos": [
                {"rank": 3, "augmentIds": [10, 20, 40]},
                {"rank": 1, "augmentIds": [20, 30, 50]},
            ]
        },
        augment_mapping=_mapping(),
    )

    assert result["recommendCombo"]["augmentId"] == 50
    assert result["recommendCombo"]["comboRank"] == 1


def test_same_combo_rank_breaks_tie_by_candidate_win_rate() -> None:
    result = build_battle_recommendation(
        round_number=2,
        picked_augment_ids=[10],
        candidate_augment_ids=[40, 50],
        champion_augments=_augments(),
        champion_combos={
            "combos": [
                {"rank": 2, "augmentIds": [10, 40, 99]},
                {"rank": 2, "augmentIds": [10, 50, 98]},
            ]
        },
        augment_mapping=_mapping(),
    )

    # same rank=2, choose higher winRate candidate (40 > 50)
    assert result["recommendCombo"]["augmentId"] == 40

