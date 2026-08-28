#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Everlight Photo Coupler crawler reference
=========================================

用途
----
這是一支給 IT / 後續維護人員參考的「單檔版」爬蟲範例。

它參考原本 Everlight crawler 的資料結構與限制，將公開網頁/PDF整理成目前
RAG 前處理可直接讀取的格式：

DATA_DIR/
├── raw/
│   ├── html/<document_id>.html
│   └── pdf/<document_id>.pdf
├── text/<document_id>.txt
├── everlight.db
└── rag_ready/
    └── documents.jsonl

RAG 後續可直接執行：
    python rag.py inspect
    python rag.py prepare-html
    python rag.py prepare-pdf
    python rag.py chunk
    python rag.py build-index

重要說明
--------
1. document_id 使用下載內容的 SHA-256。
2. HTML 會保存 raw HTML 與抽出的 TXT。
3. PDF 會保存原始 PDF；同時額外保存文字 TXT 供人工檢查，
   但目前 RAG 的 prepare-pdf 仍會直接以 PDF page image + Qwen3.5-4B 轉 Markdown。
4. 同時更新 everlight.db 與 rag_ready/documents.jsonl。
   這點很重要，因為目前 RAG 若偵測到 everlight.db，會優先讀 DB。
5. 預設僅允許億光繁中站的光耦產品區、光耦應用手冊下載頁與相關 PDF。
6. 預設 strict robots.txt；robots.txt 不允許的 URL 不抓取。
7. 預設每次 HTTP request 至少相隔 10 秒，避免造成網站負擔。

需要的套件
----------
beautifulsoup4
pymupdf

其餘均使用 Python standard library。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sqlite3
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, build_opener
from urllib.robotparser import RobotFileParser

import fitz  # PyMuPDF
from bs4 import BeautifulSoup


LOGGER = logging.getLogger("everlight-crawler-reference")

DEFAULT_SEEDS = (
    "https://www.everlight.com/photo_coupler_igbt_ssr/",
    "https://www.everlight.com/download-category/application_note_coupler/",
)

ALLOWED_HOSTS = {
    "www.everlight.com",
    "everlight.com",
}

PHOTO_COUPLER_PREFIX = "/photo_coupler_igbt_ssr/"
APPLICATION_NOTE_PREFIX = "/download-category/application_note_coupler/"
DOWNLOAD_PREFIX = "/download/"

BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".css", ".js", ".ico",
    ".zip", ".rar", ".7z",
    ".mp3", ".mp4", ".avi", ".mov",
    ".woff", ".woff2", ".ttf",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """Canonicalize Everlight URLs while retaining only stable wpdmdl query id."""
    absolute = urljoin(base_url or "", url.strip())
    parts = urlsplit(absolute)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return None

    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    path = re.sub(r"/{2,}", "/", parts.path or "/")

    # 網頁目錄網址統一補 /
    if not Path(path).suffix and not path.endswith("/"):
        path += "/"

    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() == "wpdmdl"
    ]
    query = urlencode(query_items)
    return urlunsplit((scheme, host, path, query, ""))


def encode_request_url(url: str) -> str:
    """Percent-encode non-ASCII IRI characters before urllib sends the request."""
    parts = urlsplit(url)
    path = quote(parts.path, safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="%/:?@!$&'()*+,;=-._~")
    return urlunsplit((parts.scheme, parts.netloc, path, query, ""))


def is_download_endpoint(url: str) -> bool:
    return any(k.lower() == "wpdmdl" for k, _ in parse_qsl(urlsplit(url).query))


def looks_like_pdf_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf") or is_download_endpoint(url)


def is_allowed_url(url: str) -> bool:
    """Keep crawling tightly scoped to Everlight photo-coupler public content."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path.lower()
    suffix = Path(path).suffix.lower()

    if host not in ALLOWED_HOSTS:
        return False

    if suffix in BLOCKED_EXTENSIONS:
        return False

    if looks_like_pdf_url(url):
        return True

    if suffix and suffix not in {".html", ".htm"}:
        return False

    return (
        path.startswith(PHOTO_COUPLER_PREFIX)
        or path.startswith(APPLICATION_NOTE_PREFIX)
        or path.startswith(DOWNLOAD_PREFIX)
    )


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping section boundaries."""
    out: list[str] = []
    previous = None
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t \u00a0]+", " ", raw_line).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if line == previous:
            continue
        out.append(line)
        previous = line
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out).strip()


def extract_html(raw: bytes, source_url: str) -> tuple[str, str, list[str]]:
    """Return title, cleaned text, discovered links."""
    html_text = raw.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    # 先抓 links，再移除 navigation 等區塊。
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag.get("href") or "").strip()
        if href:
            links.append(href)

    # 有些下載網址藏在 data-* 或 inline attributes。
    for tag in soup.find_all(True):
        for value in tag.attrs.values():
            if isinstance(value, list):
                values = [str(x) for x in value]
            else:
                values = [str(value)]
            for candidate in values:
                candidate = candidate.replace("\\/", "/").strip()
                if ".pdf" in candidate.lower() or "wpdmdl=" in candidate.lower():
                    links.append(candidate)

    # 再從 raw HTML 中補抓直接 PDF / wpdmdl URL。
    raw_for_scan = html_text.replace("\\/", "/")
    links.extend(
        m.rstrip(".,);]")
        for m in re.findall(
            r"(?i)(?:https?://[^\s\"'<>]+?\.pdf(?:\?[^\s\"'<>]*)?|"
            r"/[^\s\"'<>]+?\.pdf(?:\?[^\s\"'<>]*)?)",
            raw_for_scan,
        )
    )
    links.extend(
        m.rstrip(".,);]")
        for m in re.findall(
            r"(?i)(?:https?://|/|\?)[^\s\"'<>]*?\bwpdmdl=\d+[^\s\"'<>]*",
            raw_for_scan,
        )
    )

    h1 = soup.find("h1")
    title_tag = soup.find("title")
    title = ""
    if h1:
        title = clean_text(h1.get_text(" ", strip=True))
    if not title and title_tag:
        title = clean_text(title_tag.get_text(" ", strip=True))
    if not title:
        title = source_url

    for tag_name in ("script", "style", "noscript", "svg", "canvas", "template", "iframe"):
        for node in soup.find_all(tag_name):
            node.decompose()

    # 網站全域 UI 優先移除；真正產品內容通常位於 main/article/body。
    for tag_name in ("header", "footer", "nav", "form", "aside"):
        for node in soup.find_all(tag_name):
            node.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean_text(root.get_text("\n", strip=True))
    if title and text and not text.startswith(title):
        text = f"{title}\n\n{text}"

    return title, text, list(dict.fromkeys(links))


def extract_pdf_info(raw: bytes, source_url: str) -> tuple[str, str, int]:
    """Extract only metadata/text for crawler storage; RAG later still uses page images."""
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        meta = doc.metadata or {}
        title = str(meta.get("title") or "").strip()

        pages: list[str] = []
        first_line = ""
        for page_no, page in enumerate(doc, start=1):
            page_text = clean_text(page.get_text("text") or "")
            if not first_line and page_text:
                first_line = page_text.splitlines()[0][:200]
            pages.append(f"--- Page {page_no} ---\n{page_text}".rstrip())

        if not title:
            title = first_line or Path(urlsplit(source_url).path).name or source_url

        return title, "\n\n".join(pages), len(doc)
    finally:
        doc.close()


@dataclass(slots=True)
class FetchResult:
    status: int
    final_url: str
    content_type: str
    body: bytes


class PoliteHttpClient:
    def __init__(
        self,
        user_agent: str,
        min_delay: float,
        max_delay: float,
        timeout: float,
        max_bytes: int,
        batch_size: int,
        batch_pause: float,
    ) -> None:
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.batch_size = max(1, batch_size)
        self.batch_pause = max(0.0, batch_pause)
        self.opener = build_opener()
        self.last_request_at: float | None = None
        self.request_count = 0

    def _wait(self) -> None:
        if self.last_request_at is not None:
            target = random.uniform(self.min_delay, self.max_delay)
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < target:
                wait = target - elapsed
                LOGGER.info("等待 %.1f 秒後再送出 request", wait)
                time.sleep(wait)

        if self.request_count > 0 and self.request_count % self.batch_size == 0:
            if self.batch_pause > 0:
                LOGGER.info("已完成 %s 次 request，額外休息 %.1f 秒", self.request_count, self.batch_pause)
                time.sleep(self.batch_pause)

    def get(self, url: str) -> FetchResult:
        self._wait()
        req = Request(
            encode_request_url(url),
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1",
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.5",
            },
            method="GET",
        )

        try:
            response = self.opener.open(req, timeout=self.timeout)
        except HTTPError as exc:
            self.last_request_at = time.monotonic()
            self.request_count += 1
            return FetchResult(
                status=int(exc.code),
                final_url=exc.geturl(),
                content_type=str(exc.headers.get("Content-Type", "")),
                body=b"",
            )

        self.last_request_at = time.monotonic()
        self.request_count += 1

        with response:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                if int(content_length) > self.max_bytes:
                    raise ValueError(
                        f"Content-Length exceeds limit: {int(content_length)} > {self.max_bytes}"
                    )

            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, self.max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.max_bytes:
                    raise ValueError(f"Response exceeds {self.max_bytes} bytes")

            return FetchResult(
                status=int(getattr(response, "status", response.getcode())),
                final_url=response.geturl(),
                content_type=str(response.headers.get("Content-Type", "")),
                body=b"".join(chunks),
            )


class RobotsCache:
    """Strict robots.txt cache.

    robots.txt 讀取失敗時，預設拒絕抓取該 host。
    """

    def __init__(self, user_agent: str, timeout: float = 20.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host_key = f"{parts.scheme.lower()}://{parts.netloc.lower()}"

        if host_key not in self._cache:
            robots_url = f"{host_key}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)

            req = Request(
                encode_request_url(robots_url),
                headers={"User-Agent": self.user_agent},
                method="GET",
            )
            try:
                with build_opener().open(req, timeout=self.timeout) as resp:
                    if int(getattr(resp, "status", resp.getcode())) >= 400:
                        raise RuntimeError(f"robots HTTP {resp.getcode()}")
                    raw = resp.read(1024 * 1024)
                parser.parse(raw.decode("utf-8", errors="ignore").splitlines())
                self._cache[host_key] = parser
            except Exception as exc:
                LOGGER.warning("robots.txt 無法取得，strict mode 拒絕該 host：%s (%s)", host_key, exc)
                self._cache[host_key] = None

        parser = self._cache[host_key]
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)


class MetadataStore:
    """Write both everlight.db and rag_ready/documents.jsonl."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "everlight.db"
        self.docs_path = data_dir / "rag_ready" / "documents.jsonl"
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

        self.documents_by_url = self._load_documents_jsonl()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        # 與原 crawler/RAG metadata.py 相容的核心 schema。
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                canonical_url TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL CHECK(kind IN ('html', 'pdf')),
                language TEXT NOT NULL DEFAULT 'zh-TW',
                status TEXT NOT NULL DEFAULT 'done',
                depth INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 100,
                discovered_from TEXT,
                discovered_at TEXT NOT NULL,
                last_fetch_at TEXT,
                next_fetch_at TEXT,
                http_status INTEGER,
                content_type TEXT,
                etag TEXT,
                last_modified TEXT,
                content_hash TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );

            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER NOT NULL REFERENCES urls(id) ON DELETE CASCADE,
                sha256 TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                text_path TEXT NOT NULL,
                text_content TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                text_chars INTEGER NOT NULL,
                page_count INTEGER,
                parser_version TEXT NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 1 CHECK(is_current IN (0, 1)),
                UNIQUE(url_id, sha256)
            );

            CREATE INDEX IF NOT EXISTS idx_versions_current
                ON document_versions(url_id, is_current);
            CREATE INDEX IF NOT EXISTS idx_versions_hash
                ON document_versions(sha256);
            """
        )
        self.conn.commit()

    def _load_documents_jsonl(self) -> dict[str, dict]:
        if not self.docs_path.exists():
            return {}
        out: dict[str, dict] = {}
        for line in self.docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_url = str(rec.get("source_url") or rec.get("canonical_url") or "").strip()
            if source_url:
                out[source_url] = rec
        return out

    def existing_record(self, source_url: str) -> dict | None:
        return self.documents_by_url.get(source_url)

    def _write_documents_jsonl(self) -> None:
        rows = sorted(
            self.documents_by_url.values(),
            key=lambda x: str(x.get("source_url", "")),
        )
        with self.docs_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def upsert_document(
        self,
        *,
        source_url: str,
        kind: str,
        title: str,
        document_id: str,
        raw_path: str,
        text_path: str,
        text_content: str,
        byte_size: int,
        page_count: int | None,
        depth: int,
        discovered_from: str | None,
        content_type: str,
    ) -> None:
        now = utc_now()

        row = self.conn.execute(
            "SELECT id FROM urls WHERE canonical_url=?",
            (source_url,),
        ).fetchone()

        if row is None:
            cursor = self.conn.execute(
                """
                INSERT INTO urls(
                    url, canonical_url, kind, language, status, depth,
                    discovered_from, discovered_at, last_fetch_at,
                    http_status, content_type, content_hash
                ) VALUES (?, ?, ?, 'zh-TW', 'done', ?, ?, ?, ?, 200, ?, ?)
                """,
                (
                    source_url,
                    source_url,
                    kind,
                    depth,
                    discovered_from,
                    now,
                    now,
                    content_type,
                    document_id,
                ),
            )
            url_id = int(cursor.lastrowid)
        else:
            url_id = int(row["id"])
            self.conn.execute(
                """
                UPDATE urls
                SET url=?, kind=?, language='zh-TW', status='done', depth=?,
                    discovered_from=?, last_fetch_at=?, http_status=200,
                    content_type=?, content_hash=?, last_error=NULL
                WHERE id=?
                """,
                (
                    source_url,
                    kind,
                    depth,
                    discovered_from,
                    now,
                    content_type,
                    document_id,
                    url_id,
                ),
            )

        existing_version = self.conn.execute(
            "SELECT id FROM document_versions WHERE url_id=? AND sha256=?",
            (url_id, document_id),
        ).fetchone()

        if existing_version is None:
            self.conn.execute(
                "UPDATE document_versions SET is_current=0 WHERE url_id=?",
                (url_id,),
            )
            cursor = self.conn.execute(
                """
                INSERT INTO document_versions(
                    url_id, sha256, fetched_at, content_type, title,
                    raw_path, text_path, text_content, byte_size, text_chars,
                    page_count, parser_version, is_current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'crawler-reference-v1', 1)
                """,
                (
                    url_id,
                    document_id,
                    now,
                    content_type,
                    title,
                    raw_path,
                    text_path,
                    text_content,
                    byte_size,
                    len(text_content),
                    page_count,
                ),
            )
            version_id = int(cursor.lastrowid)
        else:
            version_id = int(existing_version["id"])
            self.conn.execute(
                "UPDATE document_versions SET is_current=1 WHERE id=?",
                (version_id,),
            )

        self.conn.commit()

        # 同時寫 RAG 可直接讀的 JSONL metadata。
        self.documents_by_url[source_url] = {
            "canonical_url": source_url,
            "document_id": document_id,
            "fetched_at": now,
            "language": "zh-TW",
            "metadata_json": json.dumps(
                {
                    "discovered_from": discovered_from,
                    "source_parser_version": "crawler-reference-v1",
                },
                ensure_ascii=False,
            ),
            "page_count": page_count,
            "parser_version": "native-text-v1",
            "quality_flags_json": "[]",
            "raw_path": raw_path,
            "sha256": document_id,
            "source_kind": kind,
            "source_title": title,
            "source_url": source_url,
            "source_version_id": version_id,
            "text_chars": len(text_content),
            "text_path": text_path,
            "title": title,
        }
        self._write_documents_jsonl()


def ensure_dirs(data_dir: Path) -> None:
    for path in (
        data_dir / "raw" / "html",
        data_dir / "raw" / "pdf",
        data_dir / "text",
        data_dir / "rag_ready",
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_document(
    *,
    data_dir: Path,
    store: MetadataStore,
    source_url: str,
    kind: str,
    raw: bytes,
    title: str,
    text: str,
    page_count: int | None,
    depth: int,
    discovered_from: str | None,
    content_type: str,
) -> str:
    document_id = sha256_bytes(raw)

    raw_rel = Path("raw") / kind / f"{document_id}.{'pdf' if kind == 'pdf' else 'html'}"
    text_rel = Path("text") / f"{document_id}.txt"

    raw_abs = data_dir / raw_rel
    text_abs = data_dir / text_rel
    raw_abs.write_bytes(raw)
    text_abs.write_text(text, encoding="utf-8")

    store.upsert_document(
        source_url=source_url,
        kind=kind,
        title=title,
        document_id=document_id,
        raw_path=raw_rel.as_posix(),
        text_path=text_rel.as_posix(),
        text_content=text,
        byte_size=len(raw),
        page_count=page_count,
        depth=depth,
        discovered_from=discovered_from,
        content_type=content_type,
    )
    return document_id


def classify_response(result: FetchResult) -> str | None:
    content_type = result.content_type.lower()
    if "application/pdf" in content_type or result.body.startswith(b"%PDF"):
        return "pdf"
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        return "html"

    # Everlight 有些 endpoint 的 Content-Type 不完全標準。
    head = result.body[:500].lstrip().lower()
    if head.startswith(b"<!doctype html") or b"<html" in head:
        return "html"
    return None


def rediscover_from_cached_html(
    *,
    data_dir: Path,
    record: dict,
) -> list[str]:
    raw_path = str(record.get("raw_path") or "")
    if not raw_path:
        return []
    path = data_dir / raw_path
    if not path.exists() or path.suffix.lower() not in {".html", ".htm"}:
        return []
    try:
        _, _, links = extract_html(path.read_bytes(), str(record.get("source_url") or ""))
        return links
    except Exception:
        return []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reference crawler for Everlight photo-coupler public pages. "
            "Writes raw/html, raw/pdf, text, everlight.db and rag_ready/documents.jsonl."
        )
    )
    p.add_argument(
        "--data-dir",
        default="data_photo_coupler",
        help="RAG DATA_DIR；預設 data_photo_coupler",
    )
    p.add_argument(
        "--contact-email",
        default="",
        help="選填：放入 User-Agent 的管理者聯絡信箱",
    )
    p.add_argument(
        "--seed",
        action="append",
        default=[],
        help="額外 seed URL，可重複指定；不指定時使用內建光耦 seeds",
    )
    p.add_argument("--max-pages", type=int, default=300, help="本次最多處理的 URL 數")
    p.add_argument("--max-depth", type=int, default=12, help="最大連結深度")
    p.add_argument(
        "--min-delay",
        type=float,
        default=10.0,
        help="每次 HTTP request 最少間隔秒數；預設 10",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=12.0,
        help="每次 HTTP request 最大間隔秒數；預設 12",
    )
    p.add_argument("--batch-size", type=int, default=10, help="每幾次 request 額外休息")
    p.add_argument("--batch-pause", type=float, default=5.0, help="每批額外休息秒數")
    p.add_argument("--request-timeout", type=float, default=45.0, help="單次 request timeout")
    p.add_argument("--max-download-mb", type=float, default=50.0, help="單檔最大 MB")
    p.add_argument(
        "--refresh-existing",
        action="store_true",
        help="即使 documents.jsonl 已存在該 URL，也重新下載；預設使用本地 HTML 重新發現 links",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示 seed / output 路徑，不下載",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_dir = Path(args.data_dir).resolve()
    ensure_dirs(data_dir)

    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    seeds = tuple(
        x for x in (canonicalize_url(seed) for seed in seeds)
        if x and is_allowed_url(x)
    )

    print(f"DATA_DIR: {data_dir}")
    print("Seeds:")
    for seed in seeds:
        print(f"  - {seed}")

    if args.dry_run:
        print("\nDry-run：不下載。")
        print(f"Metadata JSONL: {data_dir / 'rag_ready' / 'documents.jsonl'}")
        print(f"SQLite DB:      {data_dir / 'everlight.db'}")
        return 0

    user_agent = "EverlightRAGReferenceCrawler/1.0"
    if args.contact_email:
        user_agent += f" (+mailto:{args.contact_email})"

    min_delay = max(10.0, float(args.min_delay))
    max_delay = max(min_delay, float(args.max_delay))

    http = PoliteHttpClient(
        user_agent=user_agent,
        min_delay=min_delay,
        max_delay=max_delay,
        timeout=max(5.0, float(args.request_timeout)),
        max_bytes=max(1, int(float(args.max_download_mb) * 1024 * 1024)),
        batch_size=max(1, int(args.batch_size)),
        batch_pause=max(0.0, float(args.batch_pause)),
    )
    robots = RobotsCache(user_agent=user_agent)
    store = MetadataStore(data_dir)

    queue: deque[tuple[str, int, str | None]] = deque()
    queued: set[str] = set()
    processed: set[str] = set()

    def enqueue(url: str, depth: int, discovered_from: str | None) -> None:
        canonical = canonicalize_url(url, discovered_from)
        if not canonical:
            return
        if canonical in queued or canonical in processed:
            return
        if depth > args.max_depth:
            return
        if not is_allowed_url(canonical):
            return
        queued.add(canonical)
        queue.append((canonical, depth, discovered_from))

    for seed in seeds:
        enqueue(seed, 0, None)

    count = 0
    try:
        while queue and count < max(1, int(args.max_pages)):
            url, depth, discovered_from = queue.popleft()
            processed.add(url)
            count += 1

            LOGGER.info("[%s/%s] depth=%s %s", count, args.max_pages, depth, url)

            existing = store.existing_record(url)
            if existing and not args.refresh_existing:
                LOGGER.info("已存在，使用本地 HTML 重新發現連結，不重新下載")
                for link in rediscover_from_cached_html(data_dir=data_dir, record=existing):
                    enqueue(link, depth + 1, url)
                continue

            if not robots.allowed(url):
                LOGGER.warning("robots.txt 不允許，跳過：%s", url)
                continue

            try:
                result = http.get(url)
            except (URLError, TimeoutError, ValueError, OSError) as exc:
                LOGGER.warning("request 失敗：%s | %s", url, exc)
                continue

            if result.status != 200:
                LOGGER.warning("HTTP %s：%s", result.status, url)
                continue

            final_url = canonicalize_url(result.final_url) or url
            if not is_allowed_url(final_url):
                LOGGER.warning("redirect 離開允許範圍，跳過保存：%s -> %s", url, final_url)
                continue

            kind = classify_response(result)
            if kind is None:
                LOGGER.info("非 HTML/PDF，跳過：%s (%s)", final_url, result.content_type)
                continue

            try:
                if kind == "html":
                    title, text, links = extract_html(result.body, final_url)
                    page_count = None
                else:
                    title, text, page_count = extract_pdf_info(result.body, final_url)
                    links = []
            except Exception as exc:
                LOGGER.warning("內容解析失敗：%s | %s", final_url, exc)
                continue

            if not text.strip() and kind == "html":
                LOGGER.info("HTML 無有效文字，但仍保存 raw：%s", final_url)

            doc_id = save_document(
                data_dir=data_dir,
                store=store,
                source_url=final_url,
                kind=kind,
                raw=result.body,
                title=title,
                text=text,
                page_count=page_count,
                depth=depth,
                discovered_from=discovered_from,
                content_type=result.content_type,
            )

            LOGGER.info(
                "saved kind=%s document_id=%s title=%s",
                kind,
                doc_id,
                title[:100],
            )

            if kind == "html":
                for link in links:
                    enqueue(link, depth + 1, final_url)

        print("\n完成。")
        print(f"Processed URLs: {len(processed)}")
        print(f"Remaining queue: {len(queue)}")
        print(f"HTML dir:       {data_dir / 'raw' / 'html'}")
        print(f"PDF dir:        {data_dir / 'raw' / 'pdf'}")
        print(f"TXT dir:        {data_dir / 'text'}")
        print(f"Metadata JSONL: {data_dir / 'rag_ready' / 'documents.jsonl'}")
        print(f"SQLite DB:      {data_dir / 'everlight.db'}")
        print("\n後續 RAG：")
        print("  python rag.py inspect")
        print("  python rag.py prepare-html")
        print("  python rag.py prepare-pdf")
        print("  python rag.py chunk")
        print("  python rag.py build-index")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())