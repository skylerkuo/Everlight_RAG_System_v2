#!/usr/bin/env python3
"""RAG Answer V2：終端機連續輸入多個獨立問題。

模型與 index 只載入一次；每次輸入都是新的單輪問題，
不保留上一題的對話歷史。
"""

from rag_app.qa.interactive import build_common_parser, run_interactive


def main() -> None:
    parser = build_common_parser(
        "RAG Answer V2：終端機連續輸入多個獨立問題"
    )
    args = parser.parse_args()
    run_interactive(
        mode="v2",
        question=args.question,
        debug=args.debug,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
