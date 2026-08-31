from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_app.config import Settings
from rag_app.qa.confidence import GeneratedTokenProbabilityCapture
from rag_app.utils import setup_logging


def validate_question_record(row: dict[str, Any], line_no: int) -> None:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"第 {line_no} 行缺少非空白 question 欄位")


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"第 {line_no} 行必須是 JSON object")
            validate_question_record(row, line_no)
            rows.append(row)
    return rows


def read_completed_ids(output_path: Path) -> set[Any]:
    done: set[Any] = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if row.get("status") == "ok" and "id" in row:
                done.add(row["id"])
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def extract_evaluation_reference(row: dict[str, Any]) -> dict[str, Any]:
    """只在推論完成後保存評估欄位，不讓它們進入 RAG。"""
    excluded = {"id", "category", "question"}
    return {str(k): v for k, v in row.items() if k not in excluded}


def _final_top_k_from_result(
    result: dict[str, Any],
    max_content_chars: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(result.get("results", []), start=1):
        if not isinstance(item, dict):
            item = {"content": str(item)}
        content = str(item.get("content", ""))
        if max_content_chars > 0 and len(content) > max_content_chars:
            content = content[:max_content_chars] + "…"
        output.append(
            {
                "rank": rank,
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "page_number": item.get("page_number"),
                "source_url": item.get("source_url"),
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "bge_pair_score": item.get("bge_pair_score"),
                "content": content,
            }
        )
    return output


def build_batch_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"RAG {mode.upper()} 批次 JSONL 執行器。"
            "Top-K、Candidate-K、RRF、Reranker 等參數統一讀取 rag_app/config.py。"
        )
    )
    parser.add_argument("--input", required=True, help="輸入 JSONL，每行至少需要 question")
    parser.add_argument("--output", required=True, help="輸出 JSONL")
    parser.add_argument("--limit", type=int, default=None, help="只執行前 N 題")
    parser.add_argument("--resume", action="store_true", help="略過輸出檔中 status=ok 的既有 id")
    parser.add_argument("--fail-fast", action="store_true", help="第一題錯誤就停止")
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=12000,
        help="final_top_k 每筆 evidence 最多保存字元數；0 代表不限",
    )
    parser.add_argument(
        "--enable-confidence",
        action="store_true",
        help=(
            "啟用 generated_token_probability 信心度計算。"
            "預設關閉，以避免 generate() 使用 output_scores=True 額外占用 GPU VRAM。"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="開啟詳細 log")
    return parser


def run_batch(mode: str, args: argparse.Namespace) -> int:
    if mode not in {"v2", "v3"}:
        raise ValueError("mode 必須是 v2 或 v3")

    setup_logging(args.verbose)
    settings = Settings.fixed()
    path_problems = settings.validate_paths()
    if path_problems:
        raise SystemExit("資料路徑設定有問題：\n- " + "\n- ".join(path_problems))

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"找不到輸入 JSONL：{input_path}")

    questions = load_questions(input_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    completed = read_completed_ids(output_path) if args.resume else set()
    if not args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    print("=" * 80)
    print(f"啟動 RAG {mode.upper()} 批次模式")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Total : {len(questions)}")
    print(
        f"Top-K={settings.top_k} | Candidate-K={settings.candidate_k} | "
        f"Dense/Sparse={settings.rrf_dense_weight:.2f}/{settings.rrf_sparse_weight:.2f}"
    )
    if mode == "v3":
        print(f"最多搜尋輪數={settings.max_search_rounds}")
    print("檢索參數統一由 rag_app/config.py 控制")
    print("=" * 80)

    # 延後載入大型模型相關模組，讓 --help / 靜態工具不需先安裝 transformers。
    from rag_app.qa.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings)
    ask = pipeline.ask_v3 if mode == "v3" else pipeline.ask_v2

    # ------------------------------------------------------------------
    # generated_token_probability 預設關閉
    # ------------------------------------------------------------------
    # 原本這裡會無條件安裝 GeneratedTokenProbabilityCapture，進而在
    # Hugging Face model.generate() 中強制：
    #   return_dict_in_generate=True
    #   output_scores=True
    #
    # output_scores=True 會保留每一個 generated token 的 vocabulary scores，
    # 回答較長時可能顯著增加 GPU VRAM 使用量。
    #
    # 現在只有使用者明確加入 --enable-confidence 時才啟用。
    confidence_capture: GeneratedTokenProbabilityCapture | None = None

    enable_confidence = bool(getattr(args, "enable_confidence", False))
    if enable_confidence:
        try:
            confidence_capture = GeneratedTokenProbabilityCapture(
                pipeline.engine.answer_model
            )
            confidence_capture.install()
            print("[INFO] generated_token_probability: ENABLED")
            print("[INFO] 注意：confidence 會啟用 output_scores=True，增加 GPU VRAM 使用量")
        except Exception as exc:
            print(f"[WARN] 無法啟用 generated_token_probability：{exc}")
            confidence_capture = None
    else:
        print("[INFO] generated_token_probability: DISABLED")
        print("[INFO] 不使用 output_scores=True，以降低 GPU VRAM 使用量")

    success = 0
    failed = 0
    skipped = 0

    for seq, row in enumerate(questions, start=1):
        qid = row.get("id", seq)
        if args.resume and qid in completed:
            skipped += 1
            print(f"[{seq}/{len(questions)}] SKIP id={qid}")
            continue

        question = str(row["question"]).strip()
        print("\n" + "=" * 80)
        print(f"[{seq}/{len(questions)}] id={qid}")
        print(f"Question: {question}")
        print("=" * 80)

        started = time.perf_counter()
        record: dict[str, Any] = {
            "id": qid,
            "category": row.get("category"),
            "question": question,
            "status": "ok",
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if confidence_capture is not None:
                confidence_capture.reset()

            result = ask(question)
            final_top_k = _final_top_k_from_result(
                result,
                args.max_content_chars,
            )

            record.update(
                {
                    "mode": result.get("mode"),
                    "keywords": result.get("keywords", []),
                    "initial_keywords": result.get(
                        "initial_keywords",
                        result.get("keywords", []),
                    ),
                    "proper_nouns": result.get("proper_nouns", []),
                    "search_round_count": result.get("search_round_count", 1),
                    "search_stop_reason": result.get("search_stop_reason"),
                    "search_rounds": result.get("rounds", []),
                    "candidate_top_k": settings.candidate_k,
                    "final_top_k_count": len(final_top_k),
                    "reranker_enabled": settings.reranker_enabled,
                    "reranker_model": (
                        settings.reranker_model_id
                        if settings.reranker_enabled
                        else None
                    ),
                    "model_answer": result.get("answer"),
                    "final_top_k": final_top_k,
                    # 即使 confidence 關閉仍保留欄位，維持 JSONL schema 相容。
                    "generated_token_probability": (
                        confidence_capture.last_probability
                        if confidence_capture is not None
                        else None
                    ),
                    "answer_neighbor_chunk_radius": result.get(
                        "answer_neighbor_chunk_radius",
                        settings.answer_neighbor_chunk_radius,
                    ),
                    "answer_context_neighbors": result.get(
                        "answer_context_neighbors",
                        [],
                    ),
                    "attached_pdf_images": result.get(
                        "attached_pdf_images",
                        [],
                    ),
                }
            )

            evaluation_reference = extract_evaluation_reference(row)
            if evaluation_reference:
                record["evaluation_reference"] = evaluation_reference
            success += 1

        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "model_answer": None,
                    "final_top_k": [],
                    "generated_token_probability": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            evaluation_reference = extract_evaluation_reference(row)
            if evaluation_reference:
                record["evaluation_reference"] = evaluation_reference
            failed += 1

            if args.fail_fast:
                record["elapsed_seconds"] = round(
                    time.perf_counter() - started,
                    3,
                )
                append_jsonl(output_path, record)
                raise

        record["elapsed_seconds"] = round(
            time.perf_counter() - started,
            3,
        )
        append_jsonl(output_path, record)

        probability = record.get("generated_token_probability")
        probability_text = (
            "N/A"
            if probability is None
            else f"{float(probability):.6f}"
        )
        print(
            f"status={record['status']} "
            f"final_top_k={len(record.get('final_top_k', []))} "
            f"rounds={record.get('search_round_count', 0)} "
            f"generated_token_probability={probability_text} "
            f"time={record.get('elapsed_seconds')}s"
        )

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {output_path}")
    return 0
