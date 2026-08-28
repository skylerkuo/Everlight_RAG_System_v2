from __future__ import annotations

import json
import re
from typing import Any

from rag_app.qa.prompts import ENTITY_SYSTEM, RETRIEVAL_REVIEW_SYSTEM


def extract_json_object(text: str) -> dict[str, Any]:
    """容忍 code fence 或少量前後文字，抽出第一個 JSON object。"""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return {}


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def keywords_materially_changed(old_keywords: list[str], new_keywords: list[str]) -> bool:
    """忽略順序與大小寫，只在 keyword 集合真的變動時回傳 True。"""
    old_set = {str(x).strip().casefold() for x in old_keywords if str(x).strip()}
    new_set = {str(x).strip().casefold() for x in new_keywords if str(x).strip()}
    return old_set != new_set


def extract_search_info(engine: Any, question: str) -> dict[str, list[str]]:
    """使用已載入的 Qwen，只根據原始問題抽 keywords / proper_nouns。"""
    raw = engine.answer_model.generate(
        prompt=question,
        image_paths=None,
        system=ENTITY_SYSTEM,
        max_new_tokens=engine.settings.qwen_max_new_tokens_query_analyzer,
    )
    parsed = extract_json_object(raw)
    return {
        "keywords": clean_string_list(parsed.get("keywords", [])),
        "proper_nouns": clean_string_list(parsed.get("proper_nouns", [])),
    }


def build_search_query(
    question: str,
    keywords: list[str],
    proper_nouns: list[str],
) -> str:
    """永遠保留原問題，只把搜尋用 keyword / 型號補在後面。"""
    parts = [question]
    if keywords:
        parts.append("Search keywords: " + " ; ".join(keywords))
    if proper_nouns:
        parts.append("Exact product/model names: " + " ; ".join(proper_nouns))
    return "\n".join(parts)


def review_retrieval_results(
    engine: Any,
    question: str,
    current_keywords: list[str],
    results: list[Any],
    context_text: str,
) -> dict[str, list[str]]:
    """V3 專用：讓 Qwen 判斷是否值得再搜尋一次。"""
    prompt = f"""ORIGINAL USER QUESTION:
{question}

CURRENT SEARCH KEYWORDS:
{json.dumps(current_keywords, ensure_ascii=False)}

CURRENT RETRIEVED EVIDENCE:
{context_text}

Return the required JSON only.
"""

    raw = engine.answer_model.generate(
        prompt=prompt,
        image_paths=None,
        system=RETRIEVAL_REVIEW_SYSTEM,
        max_new_tokens=engine.settings.qwen_max_new_tokens_retrieval_review,
    )
    parsed = extract_json_object(raw)

    irrelevant_sources = clean_string_list(parsed.get("irrelevant_sources", []))
    revised_keywords = clean_string_list(parsed.get("revised_keywords", []))

    valid_sources = {f"S{i}" for i in range(1, len(results) + 1)}
    cleaned_sources: list[str] = []
    seen: set[str] = set()
    for source in irrelevant_sources:
        source_id = source.strip().upper()
        if source_id not in valid_sources or source_id in seen:
            continue
        seen.add(source_id)
        cleaned_sources.append(source_id)

    return {
        "irrelevant_sources": cleaned_sources,
        "revised_keywords": revised_keywords,
    }
