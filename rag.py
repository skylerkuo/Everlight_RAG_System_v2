from __future__ import annotations

import argparse
import json

from rag_app.config import Settings
from rag_app.inspect_data import inspect_data
from rag_app.utils import release_accelerator_memory, setup_logging


def get_settings() -> Settings:
    return Settings.fixed()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Everlight PDF/HTML 本地 RAG 建置與基本查詢工具"
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("paths", help="顯示資料路徑與主要執行參數")
    sub.add_parser("inspect", help="檢查目前設定的來源資料")

    prepare_html_parser = sub.add_parser(
        "prepare-html",
        help="HTML/TXT -> Qwen3.5-4B -> Markdown",
    )
    prepare_html_parser.add_argument("--force", action="store_true")
    prepare_html_parser.add_argument("--limit", type=int)

    prepare_pdf_parser = sub.add_parser(
        "prepare-pdf",
        help="PDF page image -> Qwen3.5-4B -> 每頁 Markdown",
    )
    prepare_pdf_parser.add_argument("--force", action="store_true")
    prepare_pdf_parser.add_argument("--limit", type=int, help="測試時限制 PDF 數量")
    prepare_pdf_parser.add_argument("--page-limit", type=int, help="測試時限制每份 PDF 處理頁數")
    prepare_pdf_parser.add_argument("--context-radius", type=int, help="每側鄰頁數；未指定使用 config.py")

    prepare_parser = sub.add_parser("prepare", help="依序處理 HTML 與 PDF")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--html-limit", type=int)
    prepare_parser.add_argument("--pdf-limit", type=int)
    prepare_parser.add_argument("--page-limit", type=int)
    prepare_parser.add_argument("--context-radius", type=int)

    chunk_parser = sub.add_parser("chunk", help="只從已產生的 Markdown 建立 chunks")
    chunk_parser.add_argument(
        "--approx-tokenizer",
        action="store_true",
        help="除錯用近似 tokenizer；正式建置建議不要開啟",
    )

    sub.add_parser("index", help="建立 BGE-M3 dense+sparse index")

    build_parser = sub.add_parser("build", help="HTML/PDF -> Markdown -> Chunk -> Index")
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--html-limit", type=int)
    build_parser.add_argument("--pdf-limit", type=int)
    build_parser.add_argument("--page-limit", type=int)
    build_parser.add_argument("--context-radius", type=int)

    search_parser = sub.add_parser("search", help="直接測試 BGE-M3 index")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--top-k",
        type=int,
        help="臨時覆寫；未指定時使用 rag_app/config.py 的 top_k",
    )

    ask_parser = sub.add_parser("ask", help="基本單輪 RAG 查詢")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        help="臨時覆寫；未指定時使用 rag_app/config.py 的 top_k",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    settings = get_settings()

    if args.command == "paths":
        print(
            json.dumps(
                {
                    "paths": {
                        "data_dir": str(settings.data_dir),
                        "raw_html_dir": str(settings.raw_html_dir),
                        "raw_pdf_dir": str(settings.raw_pdf_dir),
                        "crawler_text_dir": str(settings.crawler_text_dir),
                        "db_path": str(settings.db_path),
                        "work_dir": str(settings.work_dir),
                        "md_html_dir": str(settings.md_html_dir),
                        "md_pdf_dir": str(settings.md_pdf_dir),
                        "chunks_path": str(settings.chunks_path),
                        "index_dir": str(settings.index_dir),
                    },
                    "runtime": settings.runtime_summary(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    path_problems = settings.validate_paths()
    if path_problems:
        raise SystemExit(
            "資料路徑設定有問題。請先修改 rag_app/config.py 或設定 RAG_DATA_DIR：\n- "
            + "\n- ".join(path_problems)
        )

    settings.ensure_dirs()

    if args.command == "inspect":
        print(json.dumps(inspect_data(settings.data_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "prepare-html":
        from rag_app.preprocess.html_to_md import prepare_html

        print(
            json.dumps(
                prepare_html(settings, args.force, args.limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "prepare-pdf":
        from rag_app.preprocess.pdf_to_md import prepare_pdf

        print(
            json.dumps(
                prepare_pdf(
                    settings,
                    args.force,
                    args.limit,
                    args.page_limit,
                    args.context_radius,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "prepare":
        from rag_app.preprocess.html_to_md import prepare_html
        from rag_app.preprocess.pdf_to_md import prepare_pdf

        html_result = prepare_html(settings, args.force, args.html_limit)
        release_accelerator_memory()
        pdf_result = prepare_pdf(
            settings,
            args.force,
            args.pdf_limit,
            args.page_limit,
            args.context_radius,
        )
        print(
            json.dumps(
                {"html": html_result, "pdf": pdf_result},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "chunk":
        from rag_app.chunking.markdown_chunker import build_chunks

        print(
            json.dumps(
                build_chunks(settings, use_hf_tokenizer=not args.approx_tokenizer),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "index":
        from rag_app.retrieval.bge_m3_index import BGEM3Index

        index = BGEM3Index(settings, load_model=True)
        print(json.dumps(index.build(), ensure_ascii=False, indent=2))
        return

    if args.command == "build":
        from rag_app.chunking.markdown_chunker import build_chunks
        from rag_app.preprocess.html_to_md import prepare_html
        from rag_app.preprocess.pdf_to_md import prepare_pdf
        from rag_app.retrieval.bge_m3_index import BGEM3Index

        output: dict[str, object] = {}
        output["html"] = prepare_html(settings, args.force, args.html_limit)
        release_accelerator_memory()
        output["pdf"] = prepare_pdf(
            settings,
            args.force,
            args.pdf_limit,
            args.page_limit,
            args.context_radius,
        )
        release_accelerator_memory()
        output["chunk"] = build_chunks(settings, use_hf_tokenizer=True)
        output["index"] = BGEM3Index(settings, load_model=True).build()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    if args.command == "search":
        from rag_app.retrieval.bge_m3_index import BGEM3Index

        index = BGEM3Index(settings, load_model=True)
        index.load()
        results = index.search(args.query, top_k=args.top_k)
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        from rag_app.qa.engine import RAGEngine

        engine = RAGEngine(settings)
        print(
            json.dumps(
                engine.ask(args.question, top_k=args.top_k),
                ensure_ascii=False,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
