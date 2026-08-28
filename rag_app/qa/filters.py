from __future__ import annotations

import re
from typing import Any


def result_search_text(result: Any) -> str:
    """把 SearchResult-like object 中可用字串攤平成搜尋文字。"""
    try:
        data = result.to_dict()
    except Exception:
        data = {}

    chunk = data.get("chunk", data) if isinstance(data, dict) else {}
    strings: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(chunk)
    return "\n".join(strings)


def result_document_id(result: Any) -> str | None:
    """取得 SearchResult-like object 的 document_id。

    目前 SearchResult.to_dict() 會把 chunk metadata 攤平；此 helper 同時兼容
    `result.chunk` / `to_dict()["chunk"]` / 攤平後的 `document_id`。
    """
    chunk = getattr(result, "chunk", None)
    if isinstance(chunk, dict):
        value = chunk.get("document_id") or chunk.get("doc_id") or chunk.get("source_id")
        if value is not None and str(value).strip():
            return str(value).strip()

    try:
        data = result.to_dict()
    except Exception:
        data = {}

    if not isinstance(data, dict):
        return None

    nested = data.get("chunk") if isinstance(data.get("chunk"), dict) else {}
    for container in (data, nested):
        for key in ("document_id", "doc_id", "source_id"):
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

    return None


def contains_exact_name(text: str, name: str) -> bool:
    if not name:
        return False
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def apply_exact_product_filter(
    results: list[Any],
    proper_nouns: list[str],
) -> tuple[list[Any], bool]:
    """以「文件層級」套用 Exact Product Filter。

    規則：
    1. 只在原始 BGE Candidate-K 內工作，不額外擴張候選池。
    2. 先找出「chunk 文字中有 exact product/model name」的命中 chunk。
    3. 只要某個命中 chunk 屬於 document X，就保留原始 Candidate-K 中
       所有同屬 document X 的 chunks，讓規格 / 應用 / Description 等不同
       section 的 chunk 都有機會進入後續 reranker。
    4. 如果完全沒有 exact match，回退原始 candidates。
    5. 若命中 chunk 缺少 document_id，退化成舊行為：至少保留該命中 chunk。

    這樣可避免「型號只出現在 Description，但真正答案在 Product Features」時，
    Exact Filter 把同一文件內的重要 chunk 過度刪除。
    """
    if not proper_nouns:
        return results, False

    matched_results: list[Any] = []
    matched_document_ids: set[str] = set()

    for result in results:
        text = result_search_text(result)
        if any(contains_exact_name(text, name) for name in proper_nouns):
            matched_results.append(result)
            document_id = result_document_id(result)
            if document_id:
                matched_document_ids.add(document_id)

    # 完全沒有 exact match：維持原本 fallback，不刪任何 candidate。
    if not matched_results:
        return results, False

    # 有 document_id 時，以 document 為範圍保留「原始 BGE Candidate-K 中」
    # 所有同文件 chunks；不從 index 額外載入其他 chunk。
    if matched_document_ids:
        matched_result_ids = {id(result) for result in matched_results}
        filtered = [
            result
            for result in results
            if (
                result_document_id(result) in matched_document_ids
                or (
                    id(result) in matched_result_ids
                    and result_document_id(result) is None
                )
            )
        ]
        return filtered, True

    # 極端情況：命中 exact name，但 metadata 沒有 document_id。
    # 為保持相容性，退化成舊版 chunk-level exact match。
    return matched_results, True
