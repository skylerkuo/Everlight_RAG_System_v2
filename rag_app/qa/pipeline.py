from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rag_app.config import Settings
from rag_app.qa.context import _context
from rag_app.qa.engine import RAGEngine
from rag_app.qa.filters import apply_exact_product_filter
from rag_app.qa.query_tools import (
    build_search_query,
    extract_search_info,
    keywords_materially_changed,
    review_retrieval_results,
)
from rag_app.retrieval.bge_m3_index import SearchResult
from rag_app.retrieval.reranker import BGEReranker


@dataclass(slots=True)
class RetrievalRound:
    round_no: int
    keywords: list[str]
    proper_nouns: list[str]
    retrieval_query: str
    results: list[SearchResult]
    candidate_count: int
    filtered_candidate_count: int
    exact_product_filter_applied: bool
    retrieval_seconds: float
    rerank_seconds: float
    irrelevant_sources: list[str] | None = None
    revised_keywords: list[str] | None = None
    review_seconds: float = 0.0

    def to_dict(self, include_content: bool = False, content_chars: int = 1500) -> dict:
        evidence: list[dict[str, Any]] = []
        for result in self.results:
            item = result.to_dict()
            compact = {
                "rank": result.rank,
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "page_number": item.get("page_number"),
                "source_url": item.get("source_url"),
                "score": result.score,
            }
            if include_content:
                content = str(item.get("content", ""))
                compact["content"] = content[:content_chars]
            evidence.append(compact)

        return {
            "round": self.round_no,
            "keywords": self.keywords,
            "proper_nouns": self.proper_nouns,
            "retrieval_query": self.retrieval_query,
            "candidate_count": self.candidate_count,
            "filtered_candidate_count": self.filtered_candidate_count,
            "exact_product_filter_applied": self.exact_product_filter_applied,
            "retrieval_seconds": round(self.retrieval_seconds, 3),
            "rerank_seconds": round(self.rerank_seconds, 3),
            "review_seconds": round(self.review_seconds, 3),
            "irrelevant_sources": self.irrelevant_sources or [],
            "revised_keywords": self.revised_keywords or [],
            "evidence": evidence,
        }


class RAGPipeline:
    """V2 / V3 共用的 RAG 執行服務。

    V2：單輪 Query Analyzer -> Retrieval -> Document-level Exact Filter -> Reranker -> Answer。
    V3：V2 基礎上增加 Retrieval Reviewer，最多依 config 搜尋三輪。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.fixed()
        self.settings.validate_runtime()
        self.engine = RAGEngine(self.settings)

        self.reranker: BGEReranker | None = None
        if self.settings.reranker_enabled:
            self.reranker = BGEReranker(
                model_name=self.settings.reranker_model_id,
                use_fp16=self.settings.reranker_use_fp16,
            )

    @staticmethod
    def _relabel_results(
        ranked_items: list[dict[str, Any]],
    ) -> list[SearchResult]:
        """Rerank 後重新標成 S1..Sn，避免 evidence label 沿用舊 BGE rank。"""
        results: list[SearchResult] = []
        for final_rank, item in enumerate(ranked_items, start=1):
            original = item["result"]
            results.append(
                SearchResult(
                    rank=final_rank,
                    score=float(item.get("rerank_score", original.score)),
                    chunk=original.chunk,
                    dense_score=original.dense_score,
                    sparse_score=original.sparse_score,
                    bge_pair_score=original.bge_pair_score,
                )
            )
        return results

    @staticmethod
    def _relabel_plain(results: list[SearchResult]) -> list[SearchResult]:
        return [
            SearchResult(
                rank=rank,
                score=result.score,
                chunk=result.chunk,
                dense_score=result.dense_score,
                sparse_score=result.sparse_score,
                bge_pair_score=result.bge_pair_score,
            )
            for rank, result in enumerate(results, start=1)
        ]

    def _retrieve_once(
        self,
        *,
        round_no: int,
        question: str,
        keywords: list[str],
        proper_nouns: list[str],
    ) -> RetrievalRound:
        retrieval_query = build_search_query(question, keywords, proper_nouns)

        started = time.perf_counter()
        candidates = self.engine.index.search(
            retrieval_query,
            top_k=self.settings.candidate_k,
        )
        retrieval_seconds = time.perf_counter() - started

        filtered, filter_applied = apply_exact_product_filter(
            candidates,
            proper_nouns,
        )

        rerank_seconds = 0.0
        if self.reranker is not None:
            original_ranks = {
                id(result): rank
                for rank, result in enumerate(candidates, start=1)
            }
            started = time.perf_counter()
            ranked_items = self.reranker.rerank(
                question,
                filtered,
                top_k=self.settings.top_k,
                original_ranks=original_ranks,
            )
            rerank_seconds = time.perf_counter() - started
            results = self._relabel_results(ranked_items)
        else:
            results = self._relabel_plain(filtered[: self.settings.top_k])

        return RetrievalRound(
            round_no=round_no,
            keywords=list(keywords),
            proper_nouns=list(proper_nouns),
            retrieval_query=retrieval_query,
            results=results,
            candidate_count=len(candidates),
            filtered_candidate_count=len(filtered),
            exact_product_filter_applied=filter_applied,
            retrieval_seconds=retrieval_seconds,
            rerank_seconds=rerank_seconds,
        )

    def ask_v2(self, question: str) -> dict[str, Any]:
        """單輪版本：適合延遲較低、流程較單純的線上查詢。"""
        search_info = extract_search_info(self.engine, question)
        round_data = self._retrieve_once(
            round_no=1,
            question=question,
            keywords=search_info["keywords"],
            proper_nouns=search_info["proper_nouns"],
        )

        answer_data = self.engine.answer_with_results(question, round_data.results)
        return {
            "mode": "v2",
            "question": question,
            "keywords": search_info["keywords"],
            "proper_nouns": search_info["proper_nouns"],
            "search_round_count": 1,
            "search_stop_reason": "single_pass_v2",
            "rounds": [round_data.to_dict()],
            **answer_data,
        }

    def ask_v3(self, question: str) -> dict[str, Any]:
        """迭代版本：最多依 Reviewer 的判斷重新搜尋到 config.max_search_rounds。"""
        search_info = extract_search_info(self.engine, question)
        current_keywords = list(search_info["keywords"])
        proper_nouns = list(search_info["proper_nouns"])

        rounds: list[RetrievalRound] = []
        stop_reason = "max_search_rounds_reached"

        for round_no in range(1, self.settings.max_search_rounds + 1):
            round_data = self._retrieve_once(
                round_no=round_no,
                question=question,
                keywords=current_keywords,
                proper_nouns=proper_nouns,
            )

            started = time.perf_counter()
            review = review_retrieval_results(
                engine=self.engine,
                question=question,
                current_keywords=current_keywords,
                results=round_data.results,
                context_text=_context(round_data.results),
            )
            round_data.review_seconds = time.perf_counter() - started
            round_data.irrelevant_sources = review["irrelevant_sources"]
            round_data.revised_keywords = review["revised_keywords"]
            rounds.append(round_data)

            irrelevant = review["irrelevant_sources"]
            revised = review["revised_keywords"]

            # 使用者指定的停止條件：
            # 沒有指出無用資料，或沒有提出 keyword 調整，就直接回答。
            if not irrelevant:
                stop_reason = "no_irrelevant_sources"
                break
            if not revised:
                stop_reason = "no_keyword_adjustment"
                break
            if not keywords_materially_changed(current_keywords, revised):
                stop_reason = "keyword_adjustment_unchanged"
                break
            if round_no >= self.settings.max_search_rounds:
                stop_reason = "max_search_rounds_reached"
                break

            current_keywords = revised

        final_round = rounds[-1]
        answer_data = self.engine.answer_with_results(question, final_round.results)

        return {
            "mode": "v3",
            "question": question,
            "initial_keywords": search_info["keywords"],
            "keywords": final_round.keywords,
            "proper_nouns": proper_nouns,
            "search_round_count": len(rounds),
            "search_stop_reason": stop_reason,
            "rounds": [item.to_dict() for item in rounds],
            **answer_data,
        }
