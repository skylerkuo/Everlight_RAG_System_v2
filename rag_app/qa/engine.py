from __future__ import annotations

import time
from typing import Any

from rag_app.config import Settings
from rag_app.models.qwen35_vl import Qwen35VL
from rag_app.qa.context import _answer_context, _context, _images, build_answer_prompt
from rag_app.qa.prompts import ANSWER_SYSTEM
from rag_app.qa.verifier import correct_answer, verify_answer
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

        # --------------------------------------------------------------
        # 1) Draft answer: always NO-THINKING.
        # --------------------------------------------------------------
        started = time.perf_counter()
        draft_answer = self.answer_model.generate(
            build_answer_prompt(question, context_text),
            image_paths=images,
            system=ANSWER_SYSTEM,
            max_new_tokens=self.settings.qwen_max_new_tokens_answer,
            enable_thinking=False,
        ).strip()
        answer_seconds = time.perf_counter() - started

        final_answer = draft_answer
        verification: dict[str, Any] = {
            "verdict": "disabled",
            "issues": [],
            "parse_ok": True,
        }
        verifier_seconds = 0.0
        correction_seconds = 0.0
        verifier_applied_fix = False

        # --------------------------------------------------------------
        # 2) Evidence verifier: also NO-THINKING.
        #    It does not freely re-answer; it only checks concrete
        #    engineering mismatches against the same evidence/images.
        # --------------------------------------------------------------
        if self.settings.answer_verifier_enabled:
            started = time.perf_counter()
            verification = verify_answer(
                answer_model=self.answer_model,
                question=question,
                context_text=context_text,
                draft_answer=draft_answer,
                image_paths=images,
                max_new_tokens=self.settings.qwen_max_new_tokens_verifier,
                enable_thinking=False,
            )
            verifier_seconds = time.perf_counter() - started

            # ----------------------------------------------------------
            # 3) Only an explicit FIX is allowed to modify the draft.
            #    PASS and INSUFFICIENT are audit-only and never enter correction.
            #    This conservative gate avoids changing a possibly-correct answer
            #    when the evidence is incomplete or ambiguous.
            # ----------------------------------------------------------
            if (
                verification.get("parse_ok", False)
                and verification.get("verdict") == "fix"
                and verification.get("issues")
            ):
                started = time.perf_counter()
                corrected = correct_answer(
                    answer_model=self.answer_model,
                    question=question,
                    context_text=context_text,
                    draft_answer=draft_answer,
                    verification=verification,
                    image_paths=images,
                    max_new_tokens=self.settings.qwen_max_new_tokens_correction,
                    enable_thinking=False,
                )
                correction_seconds = time.perf_counter() - started

                if corrected:
                    final_answer = corrected
                    verifier_applied_fix = True

        return {
            "answer": final_answer,
            "draft_answer": draft_answer,
            "verifier_enabled": self.settings.answer_verifier_enabled,
            "verifier_verdict": verification.get("verdict"),
            "verifier_issues": verification.get("issues", []),
            "verifier_parse_ok": verification.get("parse_ok", True),
            "verifier_applied_fix": verifier_applied_fix,
            "answer_seconds": round(answer_seconds, 3),
            "verifier_seconds": round(verifier_seconds, 3),
            "correction_seconds": round(correction_seconds, 3),
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
