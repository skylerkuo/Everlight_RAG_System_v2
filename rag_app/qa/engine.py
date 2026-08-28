from __future__ import annotations

from typing import Any

from rag_app.config import Settings
from rag_app.models.qwen35_vl import Qwen35VL
from rag_app.qa.context import _answer_context, _context, _images, build_answer_prompt
from rag_app.qa.prompts import ANSWER_SYSTEM
from rag_app.retrieval.bge_m3_index import BGEM3Index, SearchResult


class RAGEngine:
    """共用模型與 index loader；V2/V3 都透過這個物件重用同一份模型。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index = BGEM3Index(settings, load_model=True)
        self.index.load()
        self.answer_model = Qwen35VL(
            settings.qwen_model_id,
            image_scale=settings.qwen_image_scale,
        )

    def answer_with_results(
        self,
        question: str,
        results: list[SearchResult],
    ) -> dict[str, Any]:
        # Final answer only: expand text around each reranked Top-K chunk.
        context_text, neighbor_audit = _answer_context(
            results,
            self.index,
            self.settings.answer_neighbor_chunk_radius,
        )

        # PDF images intentionally remain based only on the actual Top-K results.
        images = _images(results, self.settings.max_answer_images)
        answer = self.answer_model.generate(
            build_answer_prompt(question, context_text),
            image_paths=images,
            system=ANSWER_SYSTEM,
            max_new_tokens=self.settings.qwen_max_new_tokens_answer,
        )
        return {
            "answer": answer,
            "attached_pdf_images": [str(path) for path in images],
            "results": [result.to_dict() for result in results],
            "answer_neighbor_chunk_radius": self.settings.answer_neighbor_chunk_radius,
            "answer_context_neighbors": neighbor_audit,
        }

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """基本 ask；未指定 top_k 時使用 config.py 的唯一預設值。"""
        final_top_k = self.settings.top_k if top_k is None else top_k
        results = self.index.search(question, top_k=final_top_k)
        answer_data = self.answer_with_results(question, results)
        return {
            "question": question,
            **answer_data,
        }
