from __future__ import annotations

import math
from typing import Any

from rag_app.qa.prompts import ANSWER_SYSTEM


class GeneratedTokenProbabilityCapture:
    """擷取最終回答的平均 token 生成機率。

    只在 system == ANSWER_SYSTEM 時啟用，因此 Query Analyzer 與 V3 Reviewer
    不會被算進 generated_token_probability。

    若目前 Qwen wrapper / backend 不支援 compute_transition_scores，呼叫端可略過安裝，
    批次流程仍可正常執行，只是該欄位會是 None。
    """

    def __init__(self, answer_model: Any) -> None:
        self.answer_model = answer_model
        self.backend_model = self._find_backend_model(answer_model)
        if not hasattr(self.backend_model, "compute_transition_scores"):
            raise RuntimeError(
                "Underlying Hugging Face model does not expose compute_transition_scores()."
            )

        self._original_answer_generate = answer_model.generate
        self._original_backend_generate = self.backend_model.generate
        self._capture_enabled = False
        self._installed = False
        self.last_probability: float | None = None

    @staticmethod
    def _find_backend_model(answer_model: Any) -> Any:
        for candidate in (
            getattr(answer_model, "model", None),
            getattr(answer_model, "_model", None),
            getattr(answer_model, "hf_model", None),
        ):
            if candidate is not None and hasattr(candidate, "generate"):
                return candidate
        raise RuntimeError(
            "Unable to locate underlying Hugging Face model. Expected answer_model.model / _model / hf_model."
        )

    def install(self) -> None:
        if self._installed:
            return

        capture = self
        backend = self.backend_model

        def backend_generate_with_capture(*args: Any, **kwargs: Any) -> Any:
            if not capture._capture_enabled:
                return capture._original_backend_generate(*args, **kwargs)

            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True
            outputs = capture._original_backend_generate(*args, **kwargs)

            if not hasattr(outputs, "sequences") or not hasattr(outputs, "scores"):
                capture.last_probability = None
                return outputs

            sequences = outputs.sequences
            scores = outputs.scores
            if scores is None or len(scores) == 0:
                capture.last_probability = None
                return sequences

            beam_indices = getattr(outputs, "beam_indices", None)
            transition_scores = backend.compute_transition_scores(
                sequences,
                scores,
                beam_indices=beam_indices,
                normalize_logits=True,
            )

            log_probs = transition_scores[0].detach().float()
            log_probs = log_probs[log_probs.isfinite()]
            if log_probs.numel() == 0:
                capture.last_probability = None
                return sequences

            capture.last_probability = float(math.exp(log_probs.mean().item()))
            return sequences

        def answer_generate_with_capture(*args: Any, **kwargs: Any) -> Any:
            is_final_answer = kwargs.get("system") == ANSWER_SYSTEM
            if is_final_answer:
                capture.last_probability = None
                capture._capture_enabled = True
            try:
                return capture._original_answer_generate(*args, **kwargs)
            finally:
                if is_final_answer:
                    capture._capture_enabled = False

        self.backend_model.generate = backend_generate_with_capture
        self.answer_model.generate = answer_generate_with_capture
        self._installed = True

    def reset(self) -> None:
        self.last_probability = None
