#!/usr/bin/env python3
"""RAG Answer V3：終端機連續輸入多個獨立問題，使用迭代搜尋。

模型與 index 只載入一次；每次輸入都是新的問題，不保留對話歷史。
單一問題內部可依 Reviewer 判斷重新搜尋，最多輪數由 config.py 控制。
"""

from rag_app.qa.interactive import build_common_parser, run_interactive


def main() -> None:
    parser = build_common_parser(
        "RAG Answer V3：終端機連續輸入多個獨立問題，最多三輪搜尋"
    )
    args = parser.parse_args()
    run_interactive(
        mode="v3",
        question=args.question,
        debug=args.debug,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
