from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 單一設定來源（Single Source of Truth）
# =============================================================================
# 交付後原則：
# 1. 資料路徑、Top-K、RRF 權重、Reranker、Qwen 輸出長度、V3 搜尋輪數，
#    都只在本檔案設定。
# 2. rag_loop_v2.py / rag_loop_v3.py 不再各自保存一份 Top-K 預設值。
# 3. 可用環境變數 RAG_DATA_DIR 覆寫 DATA_DIR，方便不同機器部署。
# =============================================================================

DATA_DIR = Path(
    os.environ.get(
        "RAG_DATA_DIR",
        "/home/.../data_photo_coupler_v2", # path to your data
    )
).expanduser()

# 原始爬蟲資料。
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
CRAWLER_TEXT_DIR = DATA_DIR / "text"
DB_PATH = DATA_DIR / "everlight.db"
RAG_READY_DOCUMENTS_PATH = DATA_DIR / "rag_ready" / "documents.jsonl"

# RAG 產出資料。
WORK_DIR = DATA_DIR / "rag_v6"
TXT_HTML_DIR = WORK_DIR / "txt" / "html"
MD_HTML_DIR = WORK_DIR / "md" / "html"
MD_PDF_DIR = WORK_DIR / "md" / "pdf"
PAGE_IMAGE_DIR = WORK_DIR / "page_images"
MANIFEST_PATH = WORK_DIR / "manifest.jsonl"
CHUNKS_PATH = WORK_DIR / "chunks.jsonl"
INDEX_DIR = WORK_DIR / "index"
INDEX_DENSE_PATH = INDEX_DIR / "dense.npy"
INDEX_SPARSE_PATH = INDEX_DIR / "sparse.jsonl"
INDEX_CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
BGE_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


@dataclass(slots=True)
class Settings:
    """整個 RAG 專案唯一的執行參數來源。"""

    # ---------- Paths ----------
    data_dir: Path
    raw_html_dir: Path
    raw_pdf_dir: Path
    crawler_text_dir: Path
    db_path: Path
    rag_ready_documents_path: Path

    work_dir: Path
    txt_html_dir: Path
    md_html_dir: Path
    md_pdf_dir: Path
    page_image_dir: Path
    manifest_path: Path
    chunks_path: Path
    index_dir: Path
    index_dense_path: Path
    index_sparse_path: Path
    index_chunks_path: Path
    index_meta_path: Path

    # ---------- Qwen3.5-4B ----------
    qwen_model_id: str = QWEN_MODEL_ID
    qwen_image_scale: float = 0.5
    qwen_max_new_tokens_html: int = 1800
    qwen_max_new_tokens_page: int = 2200
    qwen_max_new_tokens_answer: int = 1000
    qwen_max_new_tokens_query_analyzer: int = 160
    qwen_max_new_tokens_retrieval_review: int = 240

    # ---------- BGE-M3 ----------
    bge_model_id: str = BGE_MODEL_ID
    bge_use_fp16: bool = True
    bge_batch_size: int = 12
    bge_max_length: int = 1024

    # ---------- Retrieval ----------
    # candidate_k：先取多少候選給 Exact Product Filter / Reranker。
    # top_k：最後交給 Qwen 回答的 Evidence 數量。
    candidate_k: int = 50
    top_k: int = 5

    # Weighted RRF。Dense + Sparse 權重統一在此設定。
    rrf_dense_weight: float = 0.40
    rrf_sparse_weight: float = 0.60
    rrf_k: int = 60

    # BGE-M3 自身的 pair scoring（第一階段候選重排）。
    use_bge_pair_rerank: bool = True
    bge_pair_mode_weights: tuple[float, float, float] = (0.4, 0.2, 0.4)

    # ---------- Dedicated cross-encoder reranker ----------
    reranker_enabled: bool = True
    reranker_model_id: str = RERANKER_MODEL_ID
    reranker_use_fp16: bool = True

    # ---------- RAG Loop V3 ----------
    # V3 最多只能搜尋 3 輪；只有 Reviewer 同時指出無用 evidence 且提出新 keyword 才重搜。
    max_search_rounds: int = 3

    # ---------- Markdown chunking ----------
    chunk_target_tokens: int = 450
    chunk_max_tokens: int = 650
    chunk_overlap_tokens: int = 70
    min_chunk_tokens: int = 35

    # ---------- Final answer context ----------
    # Reranker 選出最終 Top-K 後，每個 evidence 額外補同 document 的前/後 chunk
    # 給最終 Qwen 文字推理。只擴充文字 context，不增加 PDF 圖片。
    answer_neighbor_chunk_radius: int = 1

    # ---------- PDF / multimodal ----------
    pdf_render_dpi: int = 150
    pdf_context_radius: int = 1
    max_answer_images: int = 5

    # CLI debug 顯示時，單一 evidence 最多顯示多少字元。
    debug_content_chars: int = 1500

    @classmethod
    def fixed(cls) -> "Settings":
        settings = cls(
            data_dir=DATA_DIR,
            raw_html_dir=RAW_HTML_DIR,
            raw_pdf_dir=RAW_PDF_DIR,
            crawler_text_dir=CRAWLER_TEXT_DIR,
            db_path=DB_PATH,
            rag_ready_documents_path=RAG_READY_DOCUMENTS_PATH,
            work_dir=WORK_DIR,
            txt_html_dir=TXT_HTML_DIR,
            md_html_dir=MD_HTML_DIR,
            md_pdf_dir=MD_PDF_DIR,
            page_image_dir=PAGE_IMAGE_DIR,
            manifest_path=MANIFEST_PATH,
            chunks_path=CHUNKS_PATH,
            index_dir=INDEX_DIR,
            index_dense_path=INDEX_DENSE_PATH,
            index_sparse_path=INDEX_SPARSE_PATH,
            index_chunks_path=INDEX_CHUNKS_PATH,
            index_meta_path=INDEX_META_PATH,
        )
        settings.validate_runtime()
        return settings

    def validate_runtime(self) -> None:
        """在載入大型模型前，先檢查容易出錯的執行參數。"""
        if self.top_k < 1:
            raise ValueError("top_k 必須 >= 1")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k 必須 >= top_k")
        if self.rrf_dense_weight < 0 or self.rrf_sparse_weight < 0:
            raise ValueError("RRF 權重不可為負數")
        weight_sum = self.rrf_dense_weight + self.rrf_sparse_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                "rrf_dense_weight + rrf_sparse_weight 必須等於 1.0"
            )
        if self.rrf_k < 0:
            raise ValueError("rrf_k 必須 >= 0")
        if not 1 <= self.max_search_rounds <= 3:
            raise ValueError("max_search_rounds 必須介於 1 到 3")
        if self.qwen_image_scale <= 0:
            raise ValueError("qwen_image_scale 必須 > 0")
        if self.answer_neighbor_chunk_radius < 0:
            raise ValueError("answer_neighbor_chunk_radius 必須 >= 0")

    def ensure_dirs(self) -> None:
        for path in (
            self.work_dir,
            self.txt_html_dir,
            self.md_html_dir,
            self.md_pdf_dir,
            self.page_image_dir,
            self.index_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> list[str]:
        """回傳資料路徑問題；空陣列代表路徑正常。"""
        problems: list[str] = []
        if not self.data_dir.exists():
            problems.append(f"DATA_DIR 不存在: {self.data_dir}")
        if not self.raw_html_dir.exists():
            problems.append(f"缺少 raw HTML 目錄: {self.raw_html_dir}")
        if not self.raw_pdf_dir.exists():
            problems.append(f"缺少 raw PDF 目錄: {self.raw_pdf_dir}")
        if not self.crawler_text_dir.exists():
            problems.append(f"缺少 crawler text 目錄: {self.crawler_text_dir}")
        if not self.db_path.exists() and not self.rag_ready_documents_path.exists():
            problems.append(
                "缺少 metadata：everlight.db 與 rag_ready/documents.jsonl 至少要有一個"
            )
        return problems

    def runtime_summary(self) -> dict:
        """提供給 CLI / 維護人員查看，不需要到多個檔案找參數。"""
        return {
            "data_dir": str(self.data_dir),
            "qwen_model_id": self.qwen_model_id,
            "bge_model_id": self.bge_model_id,
            "candidate_k": self.candidate_k,
            "top_k": self.top_k,
            "rrf_dense_weight": self.rrf_dense_weight,
            "rrf_sparse_weight": self.rrf_sparse_weight,
            "rrf_k": self.rrf_k,
            "use_bge_pair_rerank": self.use_bge_pair_rerank,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model_id": self.reranker_model_id,
            "max_search_rounds": self.max_search_rounds,
            "answer_neighbor_chunk_radius": self.answer_neighbor_chunk_radius,
            "max_answer_images": self.max_answer_images,
        }
