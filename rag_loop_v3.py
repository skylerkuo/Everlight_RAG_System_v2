#!/usr/bin/env python3
"""RAG Loop V3：批次讀取 JSONL，逐題執行最多三輪的迭代搜尋。

每一題仍是獨立問題，不共享前一題對話歷史。
V3 每輪 Retrieval / Rerank 後，由 Qwen Reviewer 判斷：
- 哪些 Evidence 明顯無用
- 是否要調整搜尋 keywords

只有「有無用 Evidence + keyword 有實質修改」才會進下一輪；
最多搜尋輪數統一由 rag_app/config.py 的 max_search_rounds 控制。
"""

from rag_app.qa.batch import build_batch_parser, run_batch


def main() -> int:
    parser = build_batch_parser("v3")
    args = parser.parse_args()
    return run_batch("v3", args)


if __name__ == "__main__":
    raise SystemExit(main())
