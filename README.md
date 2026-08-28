# Everlight 本地 RAG 系統

本專案為億光電子技術文件的本地 Retrieval-Augmented Generation（RAG）系統。

資料來源可包含 HTML、TXT 與 PDF，先轉成 Markdown，再進行 Chunk、BGE-M3 Dense + Sparse 檢索、Weighted RRF、Exact Product Filter、BGE Reranker，最後由 Qwen3.5-4B 根據檢索內容回答。

## 1. 系統流程

```text
HTML / TXT / PDF
        ↓
Qwen3.5-4B → Markdown
        ↓
Markdown Chunking
        ↓
BGE-M3 Dense + Sparse
        ↓
Weighted RRF
        ↓
Candidate Top-K
        ↓
Exact Product Filter
        ↓
BGE Reranker
        ↓
Final Top-K
        ↓
補入同文件前後相鄰 Chunk
        ↓
Qwen3.5-4B Answer
```

目前提供：

- **V2**：單輪 Retrieval。
- **V3**：加入 Retrieval Reviewer，可視 Evidence 情況調整 keyword 後重新搜尋。

---

## 2. 資料準備

預設資料路徑：

```text
/home/????/data_photo_coupler
```

也可自行設定：

```bash
export RAG_DATA_DIR=/your/path/data_photo_coupler
```

轉 Markdown 前，來源資料整理成：

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

### HTML / TXT

建議同時保留：

```text
raw/html/<document_id>.html
text/<document_id>.txt
```

若 TXT 已存在，系統會優先使用 TXT；若沒有 TXT，則會從 HTML 擷取文字。

### PDF

原始 PDF 放在：

```text
raw/pdf/<document_id>.pdf
```

後續會由 PyMuPDF render 成頁面影像，再由 Qwen3.5-4B 轉成 Markdown。

### Metadata

目前可從：

```text
everlight.db
```

或：

```text
rag_ready/documents.jsonl
```

讀取文件資訊。

`documents.jsonl` 範例：

```json
{"document_id":"abc123","source_kind":"html","title":"Photo Transistor","source_url":"https://www.everlight.com/example/","raw_path":"raw/html/abc123.html","text_path":"text/abc123.txt","language":"zh-TW","page_count":null}
{"document_id":"def456","source_kind":"pdf","title":"Application Note","source_url":"https://www.everlight.com/download/example/","raw_path":"raw/pdf/def456.pdf","text_path":null,"language":"zh-TW","page_count":12}
```

`document_id` 建議使用文件內容的 SHA-256。

---

## 3. 爬蟲參考程式

專案中可放置：

```text
tools/everlight_crawler_reference.py
```

此檔案供未來新增或重新收集億光公開資料時參考，不是 RAG 啟動必要程式。

爬蟲輸出會整理為：

```text
raw/html/
raw/pdf/
text/
everlight.db
rag_ready/documents.jsonl
```

爬完後即可接續執行 RAG 前處理：

```bash
python rag.py inspect
python rag.py prepare-html
python rag.py prepare-pdf
python rag.py chunk
python rag.py build-index
```

---

## 4. 環境安裝

請先依主機 GPU / CUDA 環境自行安裝適合版本的 **PyTorch**。

其餘套件：

```bash
pip install -r requirements.txt
```

目前主要模型：

```text
Qwen/Qwen3.5-4B
BAAI/bge-m3
BAAI/bge-reranker-v2-m3
```

---

## 5. 資料前處理

先確認資料路徑與來源：

```bash
python rag.py paths
python rag.py inspect
```

HTML / TXT → Markdown：

```bash
python rag.py prepare-html
```

PDF → Markdown：

```bash
python rag.py prepare-pdf
```

建立 Chunk：

```bash
python rag.py chunk
```

建立 Index：

```bash
python rag.py build-index
```

處理後主要產物位於：

```text
DATA_DIR/rag_v6/
├── md/
│   ├── html/
│   └── pdf/
├── page_images/
├── manifest.jsonl
├── chunks.jsonl
└── index/
```

若 Markdown、Chunk 或 Embedding 沒有變更，之後問答不需要重新建立 Index。

---

## 6. Retrieval 補充

### Exact Product Filter

當問題包含明確產品型號時，系統會先從 Candidate 中找出包含該型號的 Chunk，再取得其 `document_id`。

之後保留 Candidate 中相同 `document_id` 的其他 Chunk，因此同一產品文件中的規格 Chunk 即使沒有再次出現產品型號，也不會被直接刪除。

### Neighbor Chunk

Reranker 選出 Final Top-K 後，每個 Top-K Chunk 會補入同文件的前後相鄰 Chunk：

```text
Previous Chunk
Main Chunk
Next Chunk
```

目前設定：

```python
answer_neighbor_chunk_radius = 1
```

Neighbor Chunk 只提供給最終 Answer Model，不參與 Retrieval 或 Reranker 排名。

---

## 7. 實際問答

### V2

```bash
python rag_ans_v2.py
```

單次問題：

```bash
python rag_ans_v2.py --question "EL817 的 CTR 是多少？"
```

### V3

```bash
python rag_ans_v3.py
```

查看每輪 Retrieval：

```bash
python rag_ans_v3.py --debug
```

每個問題彼此獨立，不保留上一題對話內容。

---

## 8. Batch JSONL 測試

V2：

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl
```

V3：

```bash
python rag_loop_v3.py \
  --input questions.jsonl \
  --output rag_model_outputs_v3.jsonl
```

最小輸入：

```json
{"id":1,"question":"EL817 的 CTR 是多少？"}
```

批次測試時，RAG 推論只讀取 `question`。Ground Truth 與 Reference Source 僅供推論完成後的評估紀錄使用。

---

## 9. 主要設定

主要參數集中在：

```text
rag_app/config.py
```

常用設定：

```python
candidate_k = 50
top_k = 7

rrf_dense_weight = 0.40
rrf_sparse_weight = 0.60
rrf_k = 60

reranker_enabled = True
max_search_rounds = 3

answer_neighbor_chunk_radius = 1
```

---

## 10. 建議執行順序

第一次建立資料：

```bash
python rag.py paths
python rag.py inspect

python rag.py prepare-html
python rag.py prepare-pdf

python rag.py chunk
python rag.py build-index
```

完成後直接問答：

```bash
python rag_ans_v2.py
```

或：

```bash
python rag_ans_v3.py --debug
```
