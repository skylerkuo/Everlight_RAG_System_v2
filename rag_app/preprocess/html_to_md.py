from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from rag_app.config import Settings
from rag_app.metadata import SourceMeta, load_source_metadata, upsert_manifest
from rag_app.models.qwen35_vl import Qwen35VL
from rag_app.utils import meaningful_text, strip_code_fence, write_md_with_front_matter

LOGGER = logging.getLogger(__name__)

# 依使用者要求，提示詞維持原內容不變，只把執行模型統一改成 Qwen3.5-4B。
SYSTEM_PROMPT = """You are a deterministic document-cleaning engine for RAG ingestion.
Your job is to convert extracted web-page text into faithful Markdown.

Rules:
1. Preserve every technical fact, model number, numerical value, unit, product feature,
   application, and meaningful heading from the source.
2. Do NOT invent, infer, translate, or add knowledge that is not in the input.
3. Remove obvious navigation boilerplate, cookie text, repeated menu labels, empty
   download placeholders, and duplicated headers/footers when they carry no content.
4. Organize the remaining content with Markdown headings, paragraphs, bullet lists,
   and tables when the source clearly represents tabular data.
5. Keep the original language(s) of the source.
6. Return Markdown only. Do not wrap the result in a code fence.
7. If the input contains no substantive retrievable content, return exactly: __EMPTY__
"""


def _fallback_html_to_text(path: Path) -> str:
    soup = BeautifulSoup(
        path.read_text(encoding="utf-8", errors="ignore"),
        "html.parser",
    )
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    return "\n".join(x.strip() for x in soup.stripped_strings if x.strip())


def _load_or_create_txt(settings: Settings, meta: SourceMeta) -> Path:
    """優先使用 crawler 已產生的 TXT；沒有才從 raw HTML 萃取純文字。"""
    if meta.text_path:
        crawler_txt = settings.data_dir / meta.text_path
        if crawler_txt.exists():
            return crawler_txt

    raw = settings.data_dir / meta.raw_path
    if not raw.exists():
        raise FileNotFoundError(raw)

    target = settings.txt_html_dir / f"{meta.document_id}.txt"
    if not target.exists():
        target.write_text(_fallback_html_to_text(raw), encoding="utf-8")
    return target


def prepare_html(
    settings: Settings,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    settings.ensure_dirs()
    sources = [
        item
        for item in load_source_metadata(settings.data_dir).values()
        if item.source_kind == "html"
    ]
    sources.sort(key=lambda item: item.document_id)
    if limit is not None:
        sources = sources[:limit]

    model = Qwen35VL(
        settings.qwen_model_id,
        image_scale=settings.qwen_image_scale,
    )

    records: list[dict] = []
    counts = {
        "processed": 0,
        "written": 0,
        "skipped_empty": 0,
        "skipped_existing": 0,
        "failed": 0,
    }

    for index, meta in enumerate(sources, start=1):
        out_path = settings.md_html_dir / f"{meta.document_id}.md"
        try:
            if out_path.exists() and not force:
                counts["skipped_existing"] += 1
                LOGGER.info("HTML %s/%s skip existing: %s", index, len(sources), out_path.name)
                continue

            txt_path = _load_or_create_txt(settings, meta)
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            counts["processed"] += 1
            if not meaningful_text(raw_text):
                counts["skipped_empty"] += 1
                LOGGER.info("HTML %s/%s empty before Qwen: %s", index, len(sources), meta.title)
                continue

            user_prompt = f"""SOURCE TITLE: {meta.title}
SOURCE URL: {meta.source_url}

EXTRACTED TEXT:
{raw_text}
"""
            md = strip_code_fence(
                model.generate(
                    prompt=user_prompt,
                    image_paths=None,
                    system=SYSTEM_PROMPT,
                    max_new_tokens=settings.qwen_max_new_tokens_html,
                )
            ).strip()

            if md == "__EMPTY__" or not meaningful_text(re.sub(r"[#*`>|_-]", "", md)):
                counts["skipped_empty"] += 1
                LOGGER.info("HTML %s/%s removed as empty/noise: %s", index, len(sources), meta.title)
                continue

            source_txt_path = (
                str(txt_path.relative_to(settings.data_dir))
                if txt_path.is_relative_to(settings.data_dir)
                else str(txt_path)
            )
            write_md_with_front_matter(
                out_path,
                {
                    "document_id": meta.document_id,
                    "source_kind": "html",
                    "title": meta.title,
                    "source_url": meta.source_url,
                    "source_raw_path": meta.raw_path,
                    "source_txt_path": source_txt_path,
                    "language": meta.language or "",
                },
                md,
            )
            counts["written"] += 1
            records.append(
                {
                    "document_id": meta.document_id,
                    "source_kind": "html",
                    "title": meta.title,
                    "source_url": meta.source_url,
                    "md_path": str(out_path),
                    "page_number": None,
                    "page_image": None,
                    "status": "ready",
                }
            )
            LOGGER.info("HTML %s/%s -> %s", index, len(sources), out_path.name)
        except Exception as exc:
            counts["failed"] += 1
            LOGGER.exception("HTML failed %s: %s", meta.document_id, exc)

    if records:
        upsert_manifest(settings.manifest_path, records)
    return counts
