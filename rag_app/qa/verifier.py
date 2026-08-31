from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from rag_app.qa.prompts import CORRECTION_SYSTEM, VERIFIER_SYSTEM
from rag_app.qa.query_tools import extract_json_object

_ALLOWED_VERDICTS = {"pass", "fix", "insufficient"}
_ALLOWED_ISSUE_FIELDS = ("type", "claim", "evidence", "correction", "source")


def _normalize_issue(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    issue = {
        field: str(value.get(field, "")).strip()
        for field in _ALLOWED_ISSUE_FIELDS
    }
    if not any(issue.values()):
        return None
    return issue


def _parse_verification(raw: str) -> dict[str, Any]:
    """Parse verifier JSON conservatively.

    If the verifier output cannot be parsed, do not modify a possibly-correct draft.
    The caller receives parse_ok=False and verdict=pass so the original answer is kept.
    """
    parsed = extract_json_object(raw)
    if not parsed:
        return {
            "verdict": "pass",
            "issues": [],
            "parse_ok": False,
            "raw": raw,
        }

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in _ALLOWED_VERDICTS:
        return {
            "verdict": "pass",
            "issues": [],
            "parse_ok": False,
            "raw": raw,
        }

    raw_issues = parsed.get("issues", [])
    issues: list[dict[str, str]] = []
    if isinstance(raw_issues, list):
        for item in raw_issues:
            normalized = _normalize_issue(item)
            if normalized is not None:
                issues.append(normalized)

    # PASS must never trigger correction even if the model accidentally emitted issues.
    if verdict == "pass":
        issues = []

    # A correction verdict without a concrete issue is not actionable.
    # Keep the original draft rather than allowing an unconstrained rewrite.
    if verdict in {"fix", "insufficient"} and not issues:
        return {
            "verdict": "pass",
            "issues": [],
            "parse_ok": False,
            "raw": raw,
        }

    return {
        "verdict": verdict,
        "issues": issues,
        "parse_ok": True,
        "raw": raw,
    }


def build_verifier_prompt(
    question: str,
    context_text: str,
    draft_answer: str,
) -> str:
    return f"""USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context_text}

DRAFT ANSWER:
{draft_answer}

Verify the draft strictly against the supplied evidence and return the required JSON only.
"""


def build_correction_prompt(
    question: str,
    context_text: str,
    draft_answer: str,
    verification: dict[str, Any],
) -> str:
    verifier_payload = {
        "verdict": verification.get("verdict"),
        "issues": verification.get("issues", []),
    }
    return f"""USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context_text}

DRAFT ANSWER:
{draft_answer}

VERIFIER RESULT:
{json.dumps(verifier_payload, ensure_ascii=False, indent=2)}

Apply only the verifier-identified corrections and return the corrected final answer.
"""


def verify_answer(
    *,
    answer_model: Any,
    question: str,
    context_text: str,
    draft_answer: str,
    image_paths: Sequence[str | Path] | None,
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    raw = answer_model.generate(
        build_verifier_prompt(question, context_text, draft_answer),
        image_paths=image_paths,
        system=VERIFIER_SYSTEM,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
    return _parse_verification(raw)


def correct_answer(
    *,
    answer_model: Any,
    question: str,
    context_text: str,
    draft_answer: str,
    verification: dict[str, Any],
    image_paths: Sequence[str | Path] | None,
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> str:
    corrected = answer_model.generate(
        build_correction_prompt(
            question=question,
            context_text=context_text,
            draft_answer=draft_answer,
            verification=verification,
        ),
        image_paths=image_paths,
        system=CORRECTION_SYSTEM,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
    return corrected.strip()
