from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from rag_app.config import Settings
from rag_app.utils import setup_logging


def _print_debug(result: dict[str, Any], settings: Settings) -> None:
    debug = {
        "mode": result.get("mode"),
        "keywords": result.get("keywords"),
        "initial_keywords": result.get("initial_keywords"),
        "proper_nouns": result.get("proper_nouns"),
        "search_round_count": result.get("search_round_count"),
        "search_stop_reason": result.get("search_stop_reason"),
        "rounds": result.get("rounds"),
        "answer_neighbor_chunk_radius": result.get("answer_neighbor_chunk_radius"),
        "answer_context_neighbors": result.get("answer_context_neighbors"),
        "attached_pdf_images": result.get("attached_pdf_images"),
    }
    print("\n--- Debug ---")
    print(json.dumps(debug, ensure_ascii=False, indent=2))


def run_interactive(
    *,
    mode: str,
    question: str | None = None,
    debug: bool = False,
    verbose: bool = False,
) -> None:
    setup_logging(verbose)
    settings = Settings.fixed()

    path_problems = settings.validate_paths()
    if path_problems:
        raise SystemExit("資料路徑設定有問題：\n- " + "\n- ".join(path_problems))

    print("=" * 80)
    print(f"啟動 RAG {mode.upper()}，模型只載入一次。")
    print(
        f"Top-K={settings.top_k} | Candidate-K={settings.candidate_k} | "
        f"Dense/Sparse={settings.rrf_dense_weight:.2f}/{settings.rrf_sparse_weight:.2f}"
    )
    if mode == "v3":
        print(f"最多搜尋輪數={settings.max_search_rounds}")
    print("所有主要參數請統一修改 rag_app/config.py")
    print("=" * 80)

    from rag_app.qa.pipeline import RAGPipeline

    pipeline = RAGPipeline(settings)
    ask: Callable[[str], dict[str, Any]] = (
        pipeline.ask_v3 if mode == "v3" else pipeline.ask_v2
    )

    def handle(user_question: str) -> None:
        user_question = user_question.strip()
        if not user_question:
            return
        result = ask(user_question)
        print("\n回答：")
        print(result["answer"])
        if debug:
            _print_debug(result, settings)
        print()

    if question is not None:
        handle(question)
        return

    print("請直接輸入問題。輸入 exit / quit / q 離開。\n")
    while True:
        try:
            user_question = input("問題> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n結束。")
            return

        if user_question.casefold() in {"exit", "quit", "q"}:
            print("結束。")
            return

        try:
            handle(user_question)
        except Exception as exc:
            # 單題失敗不終止整個 terminal，方便現場使用與維護。
            print(f"查詢失敗：{type(exc).__name__}: {exc}\n")


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--question",
        help="單次查詢；未指定時進入互動式輸入模式。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="顯示 keyword、搜尋輪數、evidence 等除錯資訊。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="開啟詳細 log。",
    )
    return parser
