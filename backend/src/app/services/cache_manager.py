from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .augment_mapping import AugmentMappingError, get_augment_mapping
from .champion_augments import ChampionAugmentsError, get_champion_augment_stats
from .champion_augments_combo import (
    ChampionAugmentCombosError,
    get_champion_augment_combos,
)
from .champion_mapping import ChampionMappingError, get_champion_mapping
from .dtodo_client import DtodoClientError, get_champion_stats

logger = logging.getLogger("app.cache")


class CacheManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._cache: dict[str, Any] = {
            "champion_stats": None,
            "champion_mapping": None,
            "augment_mapping": None,
            "champion_merged_base": None,
            "champion_ids": [],
            "champion_augments": {},
            "champion_combos": {},
        }
        self._persist_path = Path(
            os.getenv(
                "CACHE_PERSIST_PATH",
                Path(__file__).resolve().parents[3] / ".cache" / "cache_snapshot.json",
            )
        )
        self._persist_enabled = os.getenv("CACHE_PERSIST_ENABLED", "1") == "1"

    def start(self) -> None:
        self._load_snapshot()
        self.prewarm_startup(max_seconds=5.0)
        self._start_scheduler()

    def stop(self) -> None:
        self._stop_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2)

    def prewarm_startup(self, max_seconds: float = 5.0) -> None:
        started = time.perf_counter()
        logger.info("cache prewarm started")
        self.refresh_core()

        champion_ids = self.get_champion_ids()
        if champion_ids:
            # Initialize id index for on-demand champion routes.
            with self._lock:
                for cid in champion_ids:
                    self._cache["champion_augments"].setdefault(cid, None)
                    self._cache["champion_combos"].setdefault(cid, None)

            # Prewarm one champion for /augments and /augment-combos.
            sample_id = champion_ids[0]
            try:
                self.get_champion_augments(sample_id)
                self.get_champion_combos(sample_id)
                logger.info("cache prewarmed sample champion: %s", sample_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sample champion prewarm failed: %s", exc)

        elapsed = time.perf_counter() - started
        logger.info("cache prewarm finished in %.2fs", elapsed)
        if elapsed > max_seconds:
            logger.warning("cache prewarm exceeded target time %.2fs", max_seconds)

    def refresh_core(self) -> None:
        logger.info("cache refresh core started")
        stats = get_champion_stats()
        mapping = get_champion_mapping()
        augment_map = get_augment_mapping(force_refresh=True)
        merged = self._build_champion_merged_base(stats, mapping)
        champion_ids = [str(item["championId"]) for item in stats]

        with self._lock:
            self._cache["champion_stats"] = stats
            self._cache["champion_mapping"] = mapping
            self._cache["augment_mapping"] = augment_map
            self._cache["champion_merged_base"] = merged
            self._cache["champion_ids"] = champion_ids

        self._save_snapshot()
        logger.info("cache refresh core finished: champions=%d", len(champion_ids))

    def refresh_all(self) -> None:
        logger.info("cache refresh all started")
        try:
            self.refresh_core()
        except Exception as exc:  # noqa: BLE001
            logger.exception("core refresh failed, keep old cache: %s", exc)
            return

        champion_ids = self.get_champion_ids()
        updated_augments = 0
        updated_combos = 0
        for cid in champion_ids:
            try:
                rows = get_champion_augment_stats(cid)
                with self._lock:
                    self._cache["champion_augments"][cid] = rows
                updated_augments += 1
            except ChampionAugmentsError as exc:
                logger.warning("augment refresh failed for %s: %s", cid, exc)

            try:
                combos = get_champion_augment_combos(cid)
                with self._lock:
                    self._cache["champion_combos"][cid] = combos
                updated_combos += 1
            except ChampionAugmentCombosError as exc:
                logger.warning("combo refresh failed for %s: %s", cid, exc)

        self._save_snapshot()
        logger.info(
            "cache refresh all finished: augments=%d combos=%d",
            updated_augments,
            updated_combos,
        )

    def get_champion_stats(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._cache["champion_stats"]
        if cached is not None:
            return cached
        self.refresh_core()
        with self._lock:
            return self._cache["champion_stats"] or []

    def get_champion_mapping(self) -> dict[str, dict[str, str]]:
        with self._lock:
            cached = self._cache["champion_mapping"]
        if cached is not None:
            return cached
        self.refresh_core()
        with self._lock:
            return self._cache["champion_mapping"] or {}

    def get_augment_mapping(self) -> dict[str, dict[str, str | None]]:
        with self._lock:
            cached = self._cache["augment_mapping"]
        if cached is not None:
            return cached
        self.refresh_core()
        with self._lock:
            return self._cache["augment_mapping"] or {}

    def get_champion_merged_base(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._cache["champion_merged_base"]
        if cached is not None:
            return cached
        self.refresh_core()
        with self._lock:
            return self._cache["champion_merged_base"] or []

    def get_champion_ids(self) -> list[str]:
        with self._lock:
            ids = self._cache["champion_ids"]
        if ids:
            return ids
        self.refresh_core()
        with self._lock:
            return self._cache["champion_ids"] or []

    def get_champion_augments(self, champion_id: str) -> list[dict[str, Any]]:
        cid = str(champion_id).strip()
        with self._lock:
            cached = self._cache["champion_augments"].get(cid)
        if cached is not None:
            return cached
        rows = get_champion_augment_stats(cid)
        with self._lock:
            self._cache["champion_augments"][cid] = rows
        return rows

    def get_champion_combos(self, champion_id: str) -> dict[str, Any]:
        cid = str(champion_id).strip()
        with self._lock:
            cached = self._cache["champion_combos"].get(cid)
        if cached is not None:
            return cached
        payload = get_champion_augment_combos(cid)
        with self._lock:
            self._cache["champion_combos"][cid] = payload
        return payload

    def _build_champion_merged_base(
        self,
        stats: list[dict[str, Any]],
        mapping: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for item in stats:
            champion_id = str(item["championId"])
            mapped = mapping.get(champion_id, {})
            merged.append(
                {
                    "championId": champion_id,
                    "name_cn": str(mapped.get("name_cn") or f"ID:{champion_id}"),
                    "fullname_cn": str(mapped.get("fullname_cn") or ""),
                    "winRate": float(item["winRate"]),
                    "pickRate": float(item["pickRate"]),
                }
            )
        return merged

    def _start_scheduler(self) -> None:
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="cache-refresh-scheduler", daemon=True
        )
        self._scheduler_thread.start()
        logger.info("cache refresh scheduler started")

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            override = os.getenv("CACHE_REFRESH_INTERVAL_SECONDS", "").strip()
            if override:
                try:
                    wait_seconds = max(1, int(override))
                except ValueError:
                    wait_seconds = 60
                logger.info("next cache refresh in %ds (override)", wait_seconds)
            else:
                now = datetime.now()
                next_midnight = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                wait_seconds = max(1, int((next_midnight - now).total_seconds()))
                logger.info("next cache refresh at %s", next_midnight.isoformat())

            if self._stop_event.wait(timeout=wait_seconds):
                break

            try:
                self.refresh_all()
            except Exception as exc:  # noqa: BLE001
                logger.exception("scheduled cache refresh failed: %s", exc)

    def _load_snapshot(self) -> None:
        if not self._persist_enabled or not self._persist_path.exists():
            return
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            with self._lock:
                for key in self._cache.keys():
                    if key in payload:
                        self._cache[key] = payload[key]
            logger.info("cache snapshot loaded: %s", self._persist_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load cache snapshot: %s", exc)

    def _save_snapshot(self) -> None:
        if not self._persist_enabled:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {key: self._cache[key] for key in self._cache.keys()}
            temp_path = self._persist_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(self._persist_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to save cache snapshot: %s", exc)


cache_manager = CacheManager()
