# Everlight 本地 RAG 系統

本專案是一套針對技術文件的本地 Retrieval-Augmented Generation（RAG）系統。資料來源可包含爬蟲取得的 HTML/TXT 與 PDF。HTML/TXT 與 PDF 都統一使用 **Qwen3.5-4B** 轉成 Markdown，再進行結構化切塊、BGE-M3 Dense + Sparse 混合檢索、Weighted RRF、Exact Product Filter、Cross-Encoder Reranker，最後由 Qwen3.5-4B 根據檢索證據回答。

本交付版本將「批次評估」與「終端互動查詢」明確拆開，並把共用功能集中到 `rag_app/qa/`，避免 V2/V3 各自維護一整套重複程式。

---

## 1. 四個主要執行入口

### 1.1 `rag_loop_v2.py`：批次 JSONL，V2 單輪搜尋

用途：自動讀取 JSONL，每行取出 `question`，逐題執行 RAG 並把結果寫入另一個 JSONL。

```bash
python rag_loop_v2.py \
  --input photo_coupler_eval_400_full.jsonl \
  --output rag_model_outputs_v2.jsonl
```

常用參數：

```bash
--limit 20       # 只跑前 20 題
--resume         # 跳過 output 中已完成的 id
--fail-fast      # 第一題錯誤就停止
--verbose        # 詳細 log
```

這支程式不負責互動輸入。

---

### 1.2 `rag_loop_v3.py`：批次 JSONL，V3 迭代搜尋

用途與 V2 相同，也是讀 JSONL 批次回答，但每一題內部可依 Retrieval Reviewer 判斷重新搜尋。

```bash
python rag_loop_v3.py \
  --input photo_coupler_eval_400_full.jsonl \
  --output rag_model_outputs_v3.jsonl
```

單一問題的 V3 流程最多進行 `max_search_rounds` 輪；目前預設 3 輪。不同 JSONL 題目彼此獨立，不共享對話歷史。

---

### 1.3 `rag_ans_v2.py`：終端連續輸入，V2 單輪搜尋

用途：給使用者或 IT 工程師直接在終端機輸入問題。模型與 index 只載入一次，可以連續問很多題，但每一題都是獨立查詢，不保留上一題上下文。

```bash
python rag_ans_v2.py
```

畫面：

```text
問題> EL817 的 CTR 是多少？
回答：...

問題> 如果要求 5000 Vrms 又是雙通道 Photo Transistor，哪個系列符合？
回答：...

問題> exit
```

也可只問一次：

```bash
python rag_ans_v2.py --question "EL817 的 CTR 是多少？"
```

---

### 1.4 `rag_ans_v3.py`：終端連續輸入，V3 迭代搜尋

與 `rag_ans_v2.py` 相同，差別是單一問題內可以啟動 Retrieval Review 與重新搜尋。

```bash
python rag_ans_v3.py
```

除錯時可查看每輪 keyword、停止原因與 evidence：

```bash
python rag_ans_v3.py --debug
```

注意：`rag_ans_v3.py` 可以連續問很多題，但不是聊天機器人的多輪對話。上一題答案不會被送入下一題。

---

## 2. V2 與 V3 的差異

### V2

```text
Question
  ↓
Qwen Query Analyzer
  ↓
keywords + proper_nouns
  ↓
BGE-M3 Dense + Sparse
  ↓
Weighted RRF
  ↓
Document-level Exact Product Filter（問題有明確型號時）
  ↓
BAAI/bge-reranker-v2-m3
  ↓
Final Top-K
  ↓
Qwen3.5-4B Answer
```

### V3

```text
Question
  ↓
Qwen Query Analyzer
  ↓
Retrieval + Filter + Reranker
  ↓
Final Top-K
  ↓
Qwen Retrieval Reviewer
  ↓
輸出 irrelevant_sources + revised_keywords
  ↓
是否同時滿足：
1. 有明確無用 Evidence
2. 有新的 keyword
3. keyword 確實有變
4. 還沒超過最大輪數
  ├─ 否 → 直接回答
  └─ 是 → 用新 keyword 再搜尋
```

目前 V3 最多搜尋 3 輪。原始使用者問題永遠保留，只調整附加的搜尋 keyword，降低 Query Drift。

### Exact Product Filter 的文件層級規則

當問題包含明確產品型號時，系統仍先由 BGE-M3 取得 Candidate-K（預設 50）。
Exact Product Filter 先找出 Candidate-K 中哪些 chunk 明確包含該型號；一旦命中某個 `document_id`，就保留「原始 Candidate-K 中」所有同 `document_id` 的 chunks，再交給 Reranker。

因此，如果型號寫在 Description chunk，但規格位於同一產品頁的 Product Features chunk，只要兩者原本都在 BGE Candidate-K 內，規格 chunk 不會再因為沒有重複寫產品型號而被刪除。Filter 不會從 Candidate-K 之外額外加入 chunk。

---

## 3. 參數只改 `rag_app/config.py`

所有 RAG 主要參數統一放在：

```text
rag_app/config.py
```

例如：

```python
candidate_k: int = 50
top_k: int = 7

rrf_dense_weight: float = 0.40
rrf_sparse_weight: float = 0.60
rrf_k: int = 60

reranker_enabled: bool = True
reranker_model_id: str = "BAAI/bge-reranker-v2-m3"

max_search_rounds: int = 3
```

不要再到 `rag_loop_v2.py`、`rag_loop_v3.py`、`rag_ans_v2.py` 或 `rag_ans_v3.py` 修改另一份 Top-K。

| 參數 | 預設值 | 用途 |
|---|---:|---|
| `candidate_k` | 50 | 初步候選數量 |
| `top_k` | 7 | 最後交給 Qwen 的 Evidence 數量 |
| `rrf_dense_weight` | 0.40 | Dense Weighted RRF 權重 |
| `rrf_sparse_weight` | 0.60 | Sparse Weighted RRF 權重 |
| `rrf_k` | 60 | RRF rank constant |
| `reranker_enabled` | True | 是否啟用 Cross-Encoder Reranker |
| `max_search_rounds` | 3 | V3 單一問題最多搜尋輪數 |
| `qwen_max_new_tokens_html` | 1800 | HTML/TXT → MD 最大生成 token |
| `qwen_max_new_tokens_page` | 2200 | PDF page → MD 最大生成 token |
| `qwen_max_new_tokens_answer` | 1000 | 最終回答最大生成 token |
| `max_answer_images` | 7 | 最後回答最多附帶的 PDF 頁面影像 |

`rrf_dense_weight + rrf_sparse_weight` 必須等於 1.0。

---

## 4. 模型

目前只使用以下主要模型：

```text
文件轉 Markdown / Query Analyzer / Reviewer / Answer
→ Qwen/Qwen3.5-4B

Embedding / Dense + Sparse
→ BAAI/bge-m3

Cross-Encoder Reranker
→ BAAI/bge-reranker-v2-m3
```

---

## 5. 資料路徑

預設：

```python
DATA_DIR = Path("/home/skyler/Desktop/rag_system/data_photo_coupler")
```

也可用環境變數：

```bash
export RAG_DATA_DIR=/your/path/data_photo_coupler
```

來源資料預期：

```text
DATA_DIR/
├── raw/
│   ├── html/
│   └── pdf/
├── text/
├── everlight.db
└── rag_ready/
    └── documents.jsonl
```

RAG 產物：

```text
DATA_DIR/rag_v6/
├── txt/html/
├── md/html/
├── md/pdf/
├── page_images/
├── manifest.jsonl
├── chunks.jsonl
└── index/
```

---

## 6. 文件前處理

### HTML / TXT → Markdown

```bash
python rag.py prepare-html
```

流程：

```text
crawler TXT 存在
  ├─ 是 → 使用 TXT
  └─ 否 → raw HTML → BeautifulSoup 萃取文字
                ↓
          Qwen3.5-4B
                ↓
             Markdown
```

HTML 轉 Markdown 的提示詞沿用既有版本，只更換模型為 Qwen3.5-4B。

### PDF → Markdown

```bash
python rag.py prepare-pdf
```

PDF 先以 PyMuPDF render page image，再由 Qwen3.5-4B 產生 Markdown。

預設 `pdf_context_radius=1`：

```text
前一頁影像 ┐
目標頁影像 ├─> Qwen3.5-4B → 只輸出目標頁 Markdown
下一頁影像 ┘
```

---

## 7. 建立 Chunk 與 Index

```bash
python rag.py chunk
python rag.py build-index
```

Chunk 預設：

```text
chunk_target_tokens = 450
chunk_max_tokens = 650
chunk_overlap_tokens = 70
min_chunk_tokens = 35
```

Markdown Chunker 先依 heading / section 切，再在 section 內組 chunk，不跨 heading 強行合併。

BGE-M3 Index 保存：

```text
index/
├── dense.npy
├── sparse.jsonl
├── chunks.jsonl
└── index_meta.json
```

---

## 8. 批次 JSONL 格式

最小輸入：

```json
{"id": 1, "question": "EL817 的 CTR 是多少？"}
```

可以有額外評估欄位：

```json
{
  "id": 1,
  "category": "客戶可能問的問題",
  "question": "EL817 的 CTR 是多少？",
  "ground_truth": "...",
  "source_url": "..."
}
```

公平測試原則：RAG 推論只取得 `question`。Ground Truth、Reference Source 等欄位只在回答完成後複製到 output 的 `evaluation_reference`，不會進入 Query Analyzer、Retrieval、Reranker 或 Answer Model。

---

## 9. 程式結構

```text
rag_everlight-main/
├── rag.py                     # 建資料 / chunk / index 等命令
├── rag_loop.py                # 相容入口，指向批次 V2
├── rag_loop_v2.py             # 批次 JSONL V2
├── rag_loop_v3.py             # 批次 JSONL V3
├── rag_ans.py                 # 相容入口，指向互動 V2
├── rag_ans_v2.py              # 終端互動 V2
├── rag_ans_v3.py              # 終端互動 V3
│
└── rag_app/
    ├── config.py              # 唯一主要設定來源
    ├── preprocess/
    ├── chunking/
    ├── retrieval/
    ├── models/
    └── qa/
        ├── engine.py          # Qwen Answer + PDF page image
        ├── pipeline.py        # V2/V3 共用核心流程
        ├── batch.py           # JSONL 批次 runner
        ├── interactive.py     # 終端互動 runner
        ├── prompts.py         # Query / Review / Answer prompts
        ├── query_tools.py     # keyword / proper noun / review parsing
        ├── filters.py         # Exact Product Filter
        └── confidence.py      # 批次 generated_token_probability 擷取
```

---

## 10. 建議交付檢查順序

```bash
pip install -r requirements.txt
python rag.py paths
python rag.py inspect
python rag.py chunk
python rag.py build-index
python rag_ans_v2.py
python rag_ans_v3.py --debug
python rag_loop_v2.py --input questions.jsonl --output test_v2.jsonl --limit 5
python rag_loop_v3.py --input questions.jsonl --output test_v3.jsonl --limit 5
```

若 index 已經建立完成，不需要每次啟動查詢時重新 build。

## 最終回答的相鄰 Chunk 文字補充

目前 Exact Product Filter 與最終回答流程如下：

```text
BGE-M3 Candidate Top-50
↓
若 Query Analyzer 抓到產品型號：
  先找 Candidate 中直接包含該型號的 chunk
  ↓
  找出這些 chunk 的 document_id
  ↓
  Top-50 中「不屬於命中文件」的 chunk 刪除
  ↓
  同一 document 的 Candidate chunk 即使本身沒有再次出現型號，也保留
↓
Cross-Encoder Reranker
↓
Final Top-K（由 config.py 的 top_k 控制，例如 5 或 7）
↓
對每一個 Final Top-K evidence：
  補上同 document 的前一個 chunk + 後一個 chunk 作為文字 context
↓
Qwen3.5-4B 最終回答
```

相鄰 chunk 只在**最終回答文字 context**使用，不會加入 Retrieval / Reranker 排名，也不會額外加入 PDF page image。PDF 圖片仍只依實際 Final Top-K evidence 附加。

相鄰範圍統一由 `rag_app/config.py` 控制：

```python
answer_neighbor_chunk_radius = 1
```

`1` 代表每個 Top-K chunk 補前 1 個與後 1 個 chunk；設為 `0` 可關閉。
