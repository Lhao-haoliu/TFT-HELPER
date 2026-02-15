import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
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
logger = logging.getLogger("app.main")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = _env_int("PORT", 8000)
APP_VERSION = os.getenv("APP_VERSION", "dev")
APP_COMMIT = os.getenv("COMMIT_SHA", "unknown")
OCR_MAX_UPLOAD_MB = _env_float("OCR_MAX_UPLOAD_MB", 8.0)
OCR_MAX_UPLOAD_BYTES = max(1, int(OCR_MAX_UPLOAD_MB * 1024 * 1024))
OCR_TIMEOUT_SECONDS = max(1.0, _env_float("OCR_TIMEOUT_SECONDS", 15.0))
CACHE_PERSIST_ENABLED = os.getenv("CACHE_PERSIST_ENABLED", "1")
CACHE_PERSIST_PATH = os.getenv("CACHE_PERSIST_PATH", "")
CACHE_REFRESH_INTERVAL_SECONDS = os.getenv("CACHE_REFRESH_INTERVAL_SECONDS", "")


def _ocr_error(
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", CacheControlStaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    started = time.perf_counter()
    method = request.method
    path = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "request failed method=%s path=%s duration_ms=%.2f",
            method,
            path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f",
        method,
        path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
def _startup_cache() -> None:
    logger.info(
        "startup config host=%s port=%s version=%s commit=%s",
        APP_HOST,
        APP_PORT,
        APP_VERSION,
        APP_COMMIT,
    )
    logger.info(
        "startup config cache_persist_enabled=%s cache_persist_path=%s cache_refresh_interval_seconds=%s",
        CACHE_PERSIST_ENABLED,
        CACHE_PERSIST_PATH or "(default)",
        CACHE_REFRESH_INTERVAL_SECONDS or "(daily-00:00)",
    )
    logger.info(
        "startup config ocr_max_upload_mb=%.2f ocr_timeout_seconds=%.2f",
        OCR_MAX_UPLOAD_MB,
        OCR_TIMEOUT_SECONDS,
    )
    cache_manager.start()
    logger.info("startup cache manager ready")


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


@app.post("/api/ocr/augments-names", response_model=None)
async def ocr_augment_names(
    request: Request,
) -> object:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type:
        return _ocr_error(
            status_code=400,
            code="INVALID_CONTENT_TYPE",
            message="multipart/form-data with 'image' field is required",
        )

    content_length_raw = (request.headers.get("content-length") or "").strip()
    if content_length_raw:
        try:
            content_length = int(content_length_raw)
            if content_length > OCR_MAX_UPLOAD_BYTES:
                return _ocr_error(
                    status_code=413,
                    code="FILE_TOO_LARGE",
                    message=f"image payload exceeds {OCR_MAX_UPLOAD_MB:.2f} MB limit",
                )
        except ValueError:
            pass

    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        logger.exception("ocr form parse failed")
        return _ocr_error(
            status_code=503,
            code="MULTIPART_UNAVAILABLE",
            message=(
                "multipart parser unavailable. Install dependency: "
                "pip install python-multipart"
            ),
        )

    image = form.get("image")
    if image is None:
        return _ocr_error(
            status_code=400,
            code="MISSING_IMAGE",
            message="image field is required",
        )

    image_content_type = str(getattr(image, "content_type", "")).lower()
    if image_content_type and not image_content_type.startswith("image/"):
        return _ocr_error(
            status_code=400,
            code="INVALID_IMAGE_TYPE",
            message="image file is required",
        )

    if hasattr(image, "read"):
        payload = await image.read()
    else:
        payload = bytes(image)
    if not payload:
        return _ocr_error(
            status_code=400,
            code="EMPTY_IMAGE",
            message="empty image payload",
        )
    if len(payload) > OCR_MAX_UPLOAD_BYTES:
        return _ocr_error(
            status_code=413,
            code="FILE_TOO_LARGE",
            message=f"image payload exceeds {OCR_MAX_UPLOAD_MB:.2f} MB limit",
        )

    try:
        augment_mapping = cache_manager.get_augment_mapping()
    except AugmentMappingError:
        augment_mapping = {}

    try:
        result, _ = await asyncio.wait_for(
            asyncio.to_thread(
                recognize_augment_names,
                image_bytes=payload,
                augment_mapping=augment_mapping,
                keep_debug_images=False,
            ),
            timeout=OCR_TIMEOUT_SECONDS,
        )
        return {
            "ok": True,
            "names": result.get("names", []),
            "debug": result.get("debug", {}),
        }
    except asyncio.TimeoutError:
        logger.exception("ocr timed out")
        return _ocr_error(
            status_code=504,
            code="OCR_TIMEOUT",
            message=f"ocr exceeded timeout {OCR_TIMEOUT_SECONDS:.2f}s",
        )
    except OcrInputError as exc:
        return _ocr_error(
            status_code=400,
            code="OCR_INPUT_ERROR",
            message=str(exc),
        )
    except (OcrDependencyError, OcrEngineUnavailableError) as exc:
        logger.exception("ocr dependency unavailable")
        return _ocr_error(
            status_code=503,
            code="OCR_ENGINE_UNAVAILABLE",
            message=str(exc),
        )
    except OcrAugmentError as exc:
        logger.exception("ocr processing failed")
        return _ocr_error(
            status_code=500,
            code="OCR_PROCESSING_ERROR",
            message=str(exc),
        )
    except Exception:
        logger.exception("unexpected ocr failure")
        return _ocr_error(
            status_code=500,
            code="OCR_UNKNOWN_ERROR",
            message="unexpected OCR failure",
        )


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
