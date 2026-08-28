from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_app.retrieval.bge_m3_index import BGEM3Index, SearchResult


def _chunk_location(chunk: dict) -> str:
    location: list[str] = []
    if chunk.get("source_kind"):
        location.append(str(chunk["source_kind"]).upper())
    if chunk.get("page_number"):
        location.append(f"page {chunk['page_number']}")
    heading = " > ".join(chunk.get("heading_path") or [])
    if heading:
        location.append(heading)
    return " | ".join(location)


def _context(results: list[SearchResult]) -> str:
    """Compact Top-K context used by retrieval review/debug.

    V3 retrieval control reviews only the actual reranked Top-K. Neighbor expansion
    is intentionally reserved for final answer generation.
    """
    parts: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        parts.append(
            f"[S{index}]\n"
            f"Title: {chunk.get('title', '')}\n"
            f"Location: {_chunk_location(chunk)}\n"
            f"URL: {chunk.get('source_url', '')}\n"
            f"Evidence:\n{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _adjacent_chunk_text(role: str, chunk: dict) -> str:
    return (
        f"[{role}]\n"
        f"Location: {_chunk_location(chunk)}\n"
        f"Text:\n{chunk.get('content', '')}"
    )


def _answer_context(
    results: list[SearchResult],
    index: BGEM3Index,
    neighbor_radius: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Expand each final Top-K result with same-document previous/next text chunks.

    Neighbor chunks are attached inside the same S-label group. They are not new
    ranked evidence and do not participate in retrieval or reranking.
    """
    parts: list[str] = []
    audit: list[dict[str, Any]] = []

    for evidence_index, result in enumerate(results, start=1):
        chunk = result.chunk
        previous, following = index.get_chunk_neighbors(
            chunk,
            radius=neighbor_radius,
        )

        group = [
            f"[S{evidence_index}]",
            f"Title: {chunk.get('title', '')}",
            f"Location: {_chunk_location(chunk)}",
            f"URL: {chunk.get('source_url', '')}",
            "MAIN RETRIEVED CHUNK:",
            str(chunk.get("content", "")),
        ]

        if previous or following:
            group.append(
                "ADJACENT TEXT CONTEXT FROM THE SAME DOCUMENT "
                "(context only; not additional ranked evidence):"
            )
            for neighbor in previous:
                group.append(_adjacent_chunk_text("PREVIOUS CHUNK", neighbor))
            for neighbor in following:
                group.append(_adjacent_chunk_text("NEXT CHUNK", neighbor))

        parts.append("\n".join(group))
        audit.append(
            {
                "evidence": f"S{evidence_index}",
                "main_chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "main_chunk_index": chunk.get("chunk_index"),
                "previous_chunk_ids": [item.get("chunk_id") for item in previous],
                "next_chunk_ids": [item.get("chunk_id") for item in following],
            }
        )

    return "\n\n---\n\n".join(parts), audit


def _images(results: list[SearchResult], max_images: int) -> list[Path]:
    """Attach PDF images only for the actual reranked Top-K results."""
    out: list[Path] = []
    seen: set[str] = set()
    for result in results:
        chunk = result.chunk
        if chunk.get("source_kind") != "pdf" or not chunk.get("page_image"):
            continue
        path = Path(chunk["page_image"])
        key = str(path.resolve())
        if key in seen or not path.exists():
            continue
        seen.add(key)
        out.append(path)
        if len(out) >= max_images:
            break
    return out


def build_answer_prompt(question: str, context_text: str) -> str:
    return f"""USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context_text}

Instructions:
- Answer the user question using the evidence above.
- Each [S1], [S2], ... group contains one MAIN RETRIEVED CHUNK and may include
  PREVIOUS/NEXT CHUNK text from the same document for local context.
- Adjacent chunks are context for the same evidence group; do not treat them as
  separately ranked sources.
- If PDF page images are attached, use them to verify tables, figures, values, labels,
  or layout details that may not be fully represented in the extracted Markdown.
- Cite evidence with [S1], [S2], etc.
"""
