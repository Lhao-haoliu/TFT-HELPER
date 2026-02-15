import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from .services.augment_mapping import AugmentMappingError
from .services.battle_recommend import (
    BattleRecommendError,
    build_battle_recommendation,
)
from .services.champion_augments_combo import (
    ChampionAugmentCombosError,
)
from .services.champion_augments import ChampionAugmentsError
from .services.champion_mapping import ChampionMappingError
from .services.cache_manager import cache_manager
from .services.dtodo_client import DtodoClientError
from .services.ocr_augments import (
    OcrAugmentError,
    OcrDependencyError,
    OcrEngineUnavailableError,
    OcrInputError,
    recognize_augment_names,
    resolve_augment_names,
)
from .services.static_assets import get_augment_icon_url, get_champion_icon_url


class CacheControlStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: dict[str, object]):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code in (200, 304):
            response.headers["Cache-Control"] = "public, max-age=31536000"
        return response


app = FastAPI(title="TFT Helper API")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", CacheControlStaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup_cache() -> None:
    cache_manager.start()


@app.on_event("shutdown")
def _shutdown_cache() -> None:
    cache_manager.stop()


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/champions/stats")
def champions_stats() -> list[dict[str, str | float | int | None]]:
    try:
        return cache_manager.get_champion_stats()
    except DtodoClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _match_query(q: str, name_cn: str, fullname_cn: str) -> bool:
    needle = q.strip().lower()
    if not needle:
        return True
    return needle in name_cn.lower() or needle in fullname_cn.lower()


@app.get("/api/champions")
def champions(
    q: str = "",
    sort: Literal["winRate", "pickRate"] = "winRate",
    order: Literal["desc", "asc"] = "desc",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, str | float]]:
    try:
        merged_base = cache_manager.get_champion_merged_base()
    except (DtodoClientError, ChampionMappingError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    merged: list[dict[str, str | float]] = []
    for row in merged_base:
        if not _match_query(
            q=q,
            name_cn=str(row["name_cn"]),
            fullname_cn=str(row["fullname_cn"]),
        ):
            continue
        merged.append(
            {
                "championId": str(row["championId"]),
                "name_cn": str(row["name_cn"]),
                "fullname_cn": str(row["fullname_cn"]),
                "winRate": float(row["winRate"]),
                "pickRate": float(row["pickRate"]),
                "iconUrl": get_champion_icon_url(str(row["championId"])),
            }
        )

    reverse = order == "desc"
    merged.sort(key=lambda x: float(x[sort]), reverse=reverse)

    return merged[offset : offset + limit]


@app.get("/api/champions/mapping")
def champions_mapping() -> dict[str, dict[str, str]]:
    try:
        return cache_manager.get_champion_mapping()
    except ChampionMappingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/champions/{championId}/augments")
def champion_augments(
    championId: str,
    sort: Literal["winRate", "pickRate"] = "winRate",
    order: Literal["desc", "asc"] = "desc",
) -> list[dict[str, int | float | str | None]]:
    try:
        augments = cache_manager.get_champion_augments(championId)
    except ChampionAugmentsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        augment_mapping = cache_manager.get_augment_mapping()
    except AugmentMappingError:
        augment_mapping = {}

    merged: list[dict[str, int | float | str | None]] = []
    for row in augments:
        augment_id = int(row["augmentId"])
        mapped = augment_mapping.get(str(augment_id), {})
        name_cn = mapped.get("name_cn") if isinstance(mapped, dict) else None
        rarity = mapped.get("rarity") if isinstance(mapped, dict) else None
        if not isinstance(name_cn, str) or not name_cn.strip():
            continue
        rarity_value = str(rarity or "unknown")
        merged.append(
            {
                "augmentId": augment_id,
                "name_cn": name_cn.strip(),
                "tier": str(row["tier"]),
                "winRate": float(row["winRate"]),
                "pickRate": float(row["pickRate"]),
                "games": int(row["games"]) if row.get("games") is not None else None,
                "iconUrl": get_augment_icon_url(augment_id),
                "rarity": rarity_value,
            }
        )

    reverse = order == "desc"
    merged.sort(key=lambda item: float(item[sort]), reverse=reverse)
    return merged


@app.get("/api/champions/{championId}/augment-combos")
def champion_augment_combos(championId: str) -> dict[str, object]:
    try:
        parsed = cache_manager.get_champion_combos(championId)
    except ChampionAugmentCombosError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        augment_mapping = cache_manager.get_augment_mapping()
    except AugmentMappingError:
        augment_mapping = {}

    combos_out: list[dict[str, object]] = []
    combos = parsed.get("combos", [])
    if isinstance(combos, list):
        for combo in combos:
            if not isinstance(combo, dict):
                continue
            ids = combo.get("augmentIds")
            if not isinstance(ids, list):
                continue
            augments_out: list[dict[str, object]] = []
            for augment_id in ids:
                aid = int(augment_id)
                mapped = augment_mapping.get(str(aid), {})
                name_cn = mapped.get("name_cn") if isinstance(mapped, dict) else None
                augments_out.append(
                    {"augmentId": aid, "name_cn": str(name_cn or f"#{aid}")}
                )
            combos_out.append(
                {
                    "rank": int(combo["rank"]),
                    "augments": augments_out,
                }
            )

    return {
        "championId": parsed.get("championId"),
        "version": parsed.get("version"),
        "combos": combos_out,
    }


@app.get("/api/augments/mapping")
def augments_mapping() -> dict[str, dict[str, str | None]]:
    try:
        return cache_manager.get_augment_mapping()
    except AugmentMappingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/battle/recommend")
async def battle_recommend(payload: dict[str, object]) -> dict[str, object]:
    raw_champion_id = payload.get("championId")
    raw_round = payload.get("round")
    raw_picked = payload.get("pickedAugmentIds")
    raw_candidates = payload.get("candidateAugmentIds")

    champion_id = str(raw_champion_id or "").strip()
    if not champion_id:
        raise HTTPException(status_code=400, detail="'championId' is required")

    try:
        round_number = int(raw_round)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="'round' must be an integer") from exc

    picked = raw_picked if isinstance(raw_picked, list) else []
    candidates = raw_candidates if isinstance(raw_candidates, list) else []

    try:
        champion_augments = cache_manager.get_champion_augments(champion_id)
        champion_combos = cache_manager.get_champion_combos(champion_id)
        augment_mapping = cache_manager.get_augment_mapping()
    except (ChampionAugmentsError, ChampionAugmentCombosError, AugmentMappingError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = build_battle_recommendation(
            round_number=round_number,
            picked_augment_ids=picked,  # type: ignore[arg-type]
            candidate_augment_ids=candidates,  # type: ignore[arg-type]
            champion_augments=champion_augments,
            champion_combos=champion_combos,
            augment_mapping=augment_mapping,
        )
    except BattleRecommendError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candidates_out: list[dict[str, object]] = []
    candidate_name_map: dict[int, str] = {}
    for row in result.get("candidates", []):
        if not isinstance(row, dict):
            continue
        try:
            augment_id = int(row.get("augmentId"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        name_cn = str(row.get("name_cn") or f"#{augment_id}")
        candidate_name_map[augment_id] = name_cn
        candidates_out.append(
            {
                "augmentId": augment_id,
                "name_cn": name_cn,
                "winRate": row.get("winRate"),
                "pickRate": row.get("pickRate"),
                "tier": str(row.get("tier") or "-"),
                "rarity": str(row.get("rarity") or "unknown"),
                "iconUrl": get_augment_icon_url(augment_id),
            }
        )

    recommend_single_raw = result.get("recommendSingle")
    recommend_single: dict[str, object] | None = None
    if isinstance(recommend_single_raw, dict):
        try:
            single_id = int(recommend_single_raw.get("augmentId"))  # type: ignore[arg-type]
            recommend_single = {
                "augmentId": single_id,
                "name_cn": candidate_name_map.get(single_id, f"#{single_id}"),
                "reason": str(recommend_single_raw.get("reason") or ""),
            }
        except (TypeError, ValueError):
            recommend_single = None

    recommend_combo_raw = result.get("recommendCombo")
    recommend_combo: dict[str, object] | None = None
    if isinstance(recommend_combo_raw, dict):
        try:
            combo_id = int(recommend_combo_raw.get("augmentId"))  # type: ignore[arg-type]
            recommend_combo = {
                "augmentId": combo_id,
                "name_cn": candidate_name_map.get(combo_id, f"#{combo_id}"),
                "comboRank": int(recommend_combo_raw.get("comboRank")),  # type: ignore[arg-type]
                "reason": str(recommend_combo_raw.get("reason") or ""),
            }
        except (TypeError, ValueError):
            recommend_combo = None

    return {
        "candidates": candidates_out,
        "recommendSingle": recommend_single,
        "recommendCombo": recommend_combo,
        "comboSource": str(result.get("comboSource") or "TOP10"),
    }


@app.post("/api/ocr/augments-names")
async def ocr_augment_names(
    request: Request,
) -> dict[str, object]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=400,
            detail="multipart/form-data with 'image' field is required",
        )

    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                "multipart parser unavailable. Install dependency: "
                "pip install python-multipart"
            ),
        ) from exc

    image = form.get("image")
    if image is None:
        raise HTTPException(status_code=400, detail="image field is required")

    image_content_type = str(getattr(image, "content_type", "")).lower()
    if image_content_type and not image_content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image file is required")

    if hasattr(image, "read"):
        payload = await image.read()
    else:
        payload = bytes(image)
    if not payload:
        raise HTTPException(status_code=400, detail="empty image payload")

    try:
        augment_mapping = cache_manager.get_augment_mapping()
    except AugmentMappingError:
        augment_mapping = {}

    try:
        result, _ = recognize_augment_names(
            image_bytes=payload,
            augment_mapping=augment_mapping,
            keep_debug_images=False,
        )
        return result
    except OcrInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OcrDependencyError, OcrEngineUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OcrAugmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/augments/resolve-names")
async def resolve_augment_names_api(payload: dict[str, object]) -> dict[str, object]:
    raw_names = payload.get("names")
    if not isinstance(raw_names, list):
        raise HTTPException(status_code=400, detail="'names' must be a list")

    names: list[str] = [str(item or "").strip() for item in raw_names]
    if not names:
        return {"resolved": []}

    try:
        augment_mapping = cache_manager.get_augment_mapping()
    except AugmentMappingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    resolved = resolve_augment_names(names=names, augment_mapping=augment_mapping)
    return {"resolved": resolved}
