from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DTODO_CHAMPION_LIST_URL: Final[str] = "https://hextech.dtodo.cn/zh-CN"
USER_AGENT: Final[str] = "Mozilla/5.0 (tft-helper-assets)"
VALID_EXTS: Final[set[str]] = {".png", ".webp", ".jpg", ".jpeg", ".gif"}
CONTENT_TYPE_TO_EXT: Final[dict[str, str]] = {
    "image/png": ".png",
    "image/webp": ".webp",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
}

BACKEND_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BACKEND_DIR / "static" / "champions"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
_ID_PATTERN = re.compile(r"/zh-CN/champion-stats/(\d+)")
_STYLE_URL_PATTERN = re.compile(r"url\(([^)]+)\)")


class ChampionAssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_target_anchor = False
        self.current_champion_id: str | None = None
        self.items: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()

        if tag == "a":
            href = attrs_dict.get("href", "")
            match = _ID_PATTERN.search(href)
            if not match:
                self.in_target_anchor = False
                self.current_champion_id = None
                return
            self.in_target_anchor = True
            self.current_champion_id = match.group(1)
            cid = self.current_champion_id
            if cid not in self.items:
                for key in ("data-src", "data-image", "data-bg"):
                    candidate = attrs_dict.get(key, "").strip()
                    if candidate:
                        self.items[cid] = candidate
                        break
                if cid not in self.items:
                    style = attrs_dict.get("style", "")
                    if style:
                        m = _STYLE_URL_PATTERN.search(style)
                        if m:
                            self.items[cid] = m.group(1).strip("'\"")
            return

        if not self.in_target_anchor or not self.current_champion_id:
            return

        cid = self.current_champion_id
        if cid in self.items:
            return

        img_url = ""
        if tag == "img":
            img_url = attrs_dict.get("src") or attrs_dict.get("data-src") or ""
            srcset = attrs_dict.get("srcset", "")
            if not img_url and srcset:
                img_url = srcset.split(",")[0].strip().split(" ")[0]

        if not img_url:
            style = attrs_dict.get("style", "")
            if style:
                m = _STYLE_URL_PATTERN.search(style)
                if m:
                    img_url = m.group(1).strip("'\"")

        if img_url:
            self.items[cid] = img_url

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.in_target_anchor:
            self.in_target_anchor = False
            self.current_champion_id = None


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_champion_assets(html: str) -> dict[str, str]:
    parser = ChampionAssetParser()
    parser.feed(html)
    parser.close()
    return parser.items


def resolve_extension(image_url: str, content_type: str) -> str:
    path_ext = Path(urlparse(image_url).path).suffix.lower()
    if path_ext in VALID_EXTS:
        return path_ext
    normalized_type = content_type.split(";")[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(normalized_type, ".png")


def download_image(image_url: str, champion_id: str) -> str:
    req = Request(image_url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
        content_type = resp.headers.get("Content-Type", "")
    ext = resolve_extension(image_url, content_type)
    filename = f"{champion_id}{ext}"
    (OUTPUT_DIR / filename).write_bytes(data)
    return filename


def find_existing_file(champion_id: str) -> str | None:
    for ext in VALID_EXTS:
        path = OUTPUT_DIR / f"{champion_id}{ext}"
        if path.exists():
            return path.name
    return None


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html = fetch_html(DTODO_CHAMPION_LIST_URL)
    parsed = parse_champion_assets(html)
    if not parsed:
        raise RuntimeError("no champion assets parsed from dtodo page")

    saved: dict[str, str] = {}
    failed = 0
    reused = 0
    for champion_id, raw_url in sorted(parsed.items(), key=lambda x: int(x[0])):
        existing = find_existing_file(champion_id)
        if existing:
            saved[champion_id] = existing
            reused += 1
            continue
        image_url = urljoin(DTODO_CHAMPION_LIST_URL, raw_url)
        try:
            saved[champion_id] = download_image(image_url, champion_id)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[WARN] failed to download champion {champion_id}: {exc}")

    manifest = {
        "updatedAt": int(time.time()),
        "source": DTODO_CHAMPION_LIST_URL,
        "items": saved,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    print(
        f"[OK] champion assets synced: {len(saved)} files, failed={failed}, reused={reused}, "
        f"manifest={MANIFEST_PATH}, elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
