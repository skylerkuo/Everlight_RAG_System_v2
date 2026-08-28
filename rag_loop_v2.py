#!/usr/bin/env python3
"""RAG Loop V2：批次讀取 JSONL 並逐題回答。

這支程式保留原本 rag_loop_v2 的定位：
- 從 JSONL 逐題讀取 question
- 每一題獨立執行 V2 RAG
- 每題立即寫入 output JSONL
- 支援 --resume / --limit / --fail-fast

Top-K、Candidate-K、RRF、Reranker 等檢索參數不在此檔案設定，
統一由 rag_app/config.py 管理。
"""

from rag_app.qa.batch import build_batch_parser, run_batch


def main() -> int:
    parser = build_batch_parser("v2")
    args = parser.parse_args()
    return run_batch("v2", args)


if __name__ == "__main__":
    raise SystemExit(main())
