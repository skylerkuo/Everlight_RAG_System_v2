# Everlight 本地 RAG 系統交接說明

## 1. 系統用途與整體架構

本專案是一套針對億光電子技術文件建立的本地 Retrieval-Augmented Generation（RAG）系統。

系統主要用途是將網站爬蟲取得的 HTML、TXT 與 PDF 技術資料進行統一前處理，轉換成可檢索的 Markdown，再建立 Chunk 與向量索引，最後讓使用者可以透過自然語言問題查詢產品規格、應用說明與技術文件內容。

目前系統的主要資料處理與問答流程如下：

```text
HTML / TXT / PDF
        ↓
Qwen3.5-4B
        ↓
Markdown
        ↓
Markdown Chunking
        ↓
BGE-M3 Dense + Sparse Index
        ↓
Weighted RRF
        ↓
Candidate Top-K
        ↓
Exact Product Filter
        ↓
BAAI/bge-reranker-v2-m3
        ↓
Final Top-K
        ↓
補入同文件前後相鄰 Chunk
        ↓
Qwen3.5-4B
        ↓
最終回答
```

目前系統分成兩種查詢模式：

- **V2**：單輪 Retrieval，搜尋一次後直接產生答案。
- **V3**：在 Retrieval 後增加 Reviewer，可依目前 Evidence 重新調整搜尋關鍵字並再次 Retrieval，最多執行多輪搜尋。

批次評估與終端互動查詢已拆成不同入口，但共用核心邏輯集中於：

```text
rag_app/qa/
```

避免 V2 / V3 各自維護重複程式。

---

# 2. 資料來源與資料格式

## 2.1 原始資料來源

目前系統預期資料來源包含：

```text
1. 網頁 HTML
2. 爬蟲整理後的 TXT
3. PDF 技術文件
```

資料預設放在：

```python
DATA_DIR = Path("/home/skyler/Desktop/rag_system/data_photo_coupler")
```

也可以使用環境變數指定：

```bash
export RAG_DATA_DIR=/your/path/data_photo_coupler
```

原始資料建議整理為：

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

其中：

```text
raw/html/
```

保存原始 HTML。

```text
raw/pdf/
```

保存下載的 PDF。

```text
text/
```

保存爬蟲已經整理完成的文字資料。

```text
everlight.db
```

保存爬蟲與文件相關資料。

```text
rag_ready/documents.jsonl
```

作為後續 RAG 文件處理所使用的文件清單。

---

## 2.2 RAG 處理後的資料格式

前處理完成後，RAG 相關產物會放在：

```text
DATA_DIR/rag_v6/
```

預期結構如下：

```text
rag_v6/
├── txt/
│   └── html/
├── md/
│   ├── html/
│   └── pdf/
├── page_images/
├── manifest.jsonl
├── chunks.jsonl
└── index/
    ├── dense.npy
    ├── sparse.jsonl
    ├── chunks.jsonl
    └── index_meta.json
```

用途如下：

| 路徑 | 用途 |
|---|---|
| `txt/html/` | HTML 經過文字萃取後的中間結果 |
| `md/html/` | HTML / TXT 經 Qwen3.5-4B 整理後的 Markdown |
| `md/pdf/` | PDF 每頁經 Qwen3.5-4B 轉換後的 Markdown |
| `page_images/` | PDF render 成的頁面影像 |
| `manifest.jsonl` | 前處理文件索引與來源資訊 |
| `chunks.jsonl` | Markdown 切分後的 Chunk |
| `index/` | BGE-M3 Dense / Sparse Retrieval Index |

---

## 2.3 Batch Evaluation JSONL 格式

批次評估最小輸入只需要：

```json
{"id": 1, "question": "EL817 的 CTR 是多少？"}
```

也可以保留額外的 Ground Truth：

```json
{
  "id": 1,
  "category": "客戶可能問的問題",
  "question": "EL817 的 CTR 是多少？",
  "ground_truth": "...",
  "source_url": "..."
}
```

批次測試時，真正送進 RAG Pipeline 的只有：

```text
question
```

其他 Ground Truth、Reference Source、Source URL 等資訊只用於推論完成後的評估紀錄，不會提供給：

```text
Query Analyzer
Retrieval
Reranker
Reviewer
Answer Model
```

因此測試資料中的標準答案不會參與 RAG 推論。

---

# 3. 執行環境與模型

## 3.1 Python 環境

首先安裝專案相依套件：

```bash
pip install -r requirements.txt
```

建議先確認資料路徑：

```bash
python rag.py paths
```

再檢查目前來源資料：

```bash
python rag.py inspect
```

---

## 3.2 使用模型

目前系統主要使用三個模型。

### Qwen3.5-4B

```text
Qwen/Qwen3.5-4B
```

用途：

```text
HTML / TXT → Markdown
PDF page → Markdown
Query Analyzer
V3 Retrieval Reviewer
Final Answer
```

---

### BGE-M3

```text
BAAI/bge-m3
```

用途：

```text
Dense Retrieval
Sparse Retrieval
```

系統會同時建立 Dense 與 Sparse 分數，再透過 Weighted RRF 融合。

---

### BGE Reranker

```text
BAAI/bge-reranker-v2-m3
```

用途：

```text
Candidate Chunk Cross-Encoder Reranking
```

---

# 4. 資料前處理與 Index 建立

資料準備完成後，依序執行前處理。

---

## 4.1 HTML / TXT → Markdown

執行：

```bash
python rag.py prepare-html
```

流程：

```text
crawler TXT 存在？
  ├─ 是
  │   ↓
  │ 直接使用 TXT
  │
  └─ 否
      ↓
  Raw HTML
      ↓
BeautifulSoup
      ↓
文字萃取
      ↓

TXT / Extracted Text
      ↓
Qwen3.5-4B
      ↓
Markdown
```

如果爬蟲已經產生 TXT，系統會優先使用 TXT，不需要再次從 HTML 重複做相同文字抽取。

---

## 4.2 PDF → Markdown

執行：

```bash
python rag.py prepare-pdf
```

PDF 會先由 PyMuPDF render 成頁面影像，再交給 Qwen3.5-4B 轉成 Markdown。

預設：

```python
pdf_context_radius = 1
```

表示處理某一頁時，模型可以同時看到：

```text
前一頁
目標頁
下一頁
```

但只要求模型輸出：

```text
目標頁 Markdown
```

流程如下：

```text
PDF
 ↓
PyMuPDF
 ↓
Page Images
 ↓
前一頁 + 目標頁 + 下一頁
 ↓
Qwen3.5-4B
 ↓
Target Page Markdown
```

---

## 4.3 建立 Chunk

執行：

```bash
python rag.py chunk
```

目前預設參數：

```text
chunk_target_tokens = 450
chunk_max_tokens = 650
chunk_overlap_tokens = 70
min_chunk_tokens = 35
```

Chunker 會優先依照 Markdown 的：

```text
Heading
Section
```

進行結構化切分，再於同一 Section 中組成 Chunk。

不會為了湊 Token 而任意把不同 Heading 的內容強行合併。

---

## 4.4 建立 Retrieval Index

執行：

```bash
python rag.py build-index
```

建立後會得到：

```text
index/
├── dense.npy
├── sparse.jsonl
├── chunks.jsonl
└── index_meta.json
```

其中：

```text
dense.npy
```

保存 Dense Embedding。

```text
sparse.jsonl
```

保存 Sparse Representation。

兩者會在查詢時使用 Weighted RRF 融合。

如果原始 Markdown、Chunk 規則或 Embedding 資料沒有改變，日常查詢不需要重新建立 Index。

---

# 5. Retrieval 與回答流程

## 5.1 基本 Retrieval

使用者問題會先由 Qwen Query Analyzer 解析：

```text
Question
  ↓
Qwen Query Analyzer
  ↓
keywords
+
proper_nouns
```

之後進入：

```text
BGE-M3 Dense Retrieval
+
BGE-M3 Sparse Retrieval
        ↓
Weighted RRF
        ↓
Candidate-K
```

目前預設：

```python
candidate_k = 50
```

---

## 5.2 Exact Product Filter

當問題中有明確產品型號時，系統會啟用 Exact Product Filter。

先前的處理方式是：

```text
Chunk 沒有直接出現產品型號
→ 刪除
```

但實際產品頁常出現：

```text
Description Chunk
→ 有產品型號

Product Features Chunk
→ 只有規格，沒有再次寫產品型號
```

因此如果只用 Chunk 是否出現型號判斷，會把實際上屬於該產品的規格 Chunk 誤刪。

目前改成文件層級處理：

```text
BGE-M3 Candidate Top-50
        ↓
找出直接包含產品型號的 Chunk
        ↓
取得這些 Chunk 的 document_id
        ↓
保留原始 Top-50 中
所有相同 document_id 的 Chunk
        ↓
其他文件刪除
        ↓
Reranker
```

因此，同一產品頁中的規格 Chunk 即使沒有再次寫出產品型號，只要原本存在 Candidate Top-50 中，仍然會被保留。

注意：

```text
Exact Product Filter 不會從 Candidate Top-50 外額外新增 Chunk。
```

---

## 5.3 Cross-Encoder Reranker

Exact Product Filter 完成後，Candidate Chunk 會交給：

```text
BAAI/bge-reranker-v2-m3
```

進行 Chunk-level Reranking。

最後留下：

```python
top_k
```

個 Evidence。

例如：

```python
top_k = 7
```

代表最後會保留 7 個主要 Evidence。

---

## 5.4 Final Top-K 相鄰 Chunk 補充

Reranker 選出 Final Top-K 後，系統會針對每一個 Top-K Chunk，補上原始文件切分時相鄰的 Chunk。

目前：

```python
answer_neighbor_chunk_radius = 1
```

表示：

```text
Previous Chunk
      +
Main Top-K Chunk
      +
Next Chunk
```

一起提供給最終回答模型。

流程：

```text
Final Top-K
    ↓
對每一個 Top-K Chunk
    ↓
依 document_id + chunk index
    ↓
補前 1 個 Chunk
+
補後 1 個 Chunk
    ↓
Final Answer Context
```

這些 Neighbor Chunk：

```text
只用於最終回答 Context
```

不會：

```text
加入 Retrieval 排名
加入 Reranker 排名
改變原本 Top-K 名次
額外加入 PDF Page Image
```

設為：

```python
answer_neighbor_chunk_radius = 0
```

即可關閉。

---

# 6. V2 與 V3 查詢模式

## 6.1 V2：單輪 Retrieval

V2 流程：

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
Document-level Exact Product Filter
  ↓
BGE Reranker
  ↓
Final Top-K
  ↓
Neighbor Chunk Context
  ↓
Qwen3.5-4B Answer
```

V2 每一題只 Retrieval 一次。

---

## 6.2 V3：Iterative Retrieval

V3 在第一次 Retrieval 後增加：

```text
Qwen Retrieval Reviewer
```

流程：

```text
Question
  ↓
Query Analyzer
  ↓
Retrieval
  ↓
Exact Product Filter
  ↓
Reranker
  ↓
Final Top-K
  ↓
Retrieval Reviewer
```

Reviewer 輸出：

```json
{
  "irrelevant_sources": [],
  "revised_keywords": []
}
```

系統只有在以下條件同時成立時才重新搜尋：

```text
1. Reviewer 找到明確不相關 Evidence
2. Reviewer 提供 revised_keywords
3. revised_keywords 與目前 keyword 確實不同
4. 尚未超過 max_search_rounds
```

若成立：

```text
保留原始使用者問題
+
替換 / 補充搜尋 Keywords
        ↓
重新 Retrieval
```

預設：

```python
max_search_rounds = 3
```

因此單一問題最多搜尋 3 輪。

不同問題之間彼此獨立，不共享對話歷史。

---

# 7. 主要設定

主要 RAG 設定統一放在：

```text
rag_app/config.py
```

不要分別修改：

```text
rag_loop_v2.py
rag_loop_v3.py
rag_ans_v2.py
rag_ans_v3.py
```

目前主要參數：

```python
candidate_k: int = 50
top_k: int = 7

rrf_dense_weight: float = 0.40
rrf_sparse_weight: float = 0.60
rrf_k: int = 60

reranker_enabled: bool = True
reranker_model_id: str = "BAAI/bge-reranker-v2-m3"

max_search_rounds: int = 3

answer_neighbor_chunk_radius: int = 1
```

模型 Token 相關設定：

```python
qwen_max_new_tokens_html = 1800
qwen_max_new_tokens_page = 2200
qwen_max_new_tokens_answer = 1000
max_answer_images = 7
```

主要設定整理：

| 參數 | 預設值 | 用途 |
|---|---:|---|
| `candidate_k` | 50 | BGE 初步 Candidate 數量 |
| `top_k` | 7 | Reranker 最後保留 Evidence 數量 |
| `rrf_dense_weight` | 0.40 | Dense RRF 權重 |
| `rrf_sparse_weight` | 0.60 | Sparse RRF 權重 |
| `rrf_k` | 60 | RRF rank constant |
| `reranker_enabled` | True | 是否啟用 Cross-Encoder Reranker |
| `max_search_rounds` | 3 | V3 最大 Retrieval 輪數 |
| `answer_neighbor_chunk_radius` | 1 | 最終回答補入前後 Chunk 數量 |
| `qwen_max_new_tokens_html` | 1800 | HTML/TXT → Markdown 最大 token |
| `qwen_max_new_tokens_page` | 2200 | PDF → Markdown 最大 token |
| `qwen_max_new_tokens_answer` | 1000 | Answer 最大生成 token |
| `max_answer_images` | 7 | Final Answer 最大 PDF 頁面影像數 |

必須維持：

```text
rrf_dense_weight + rrf_sparse_weight = 1.0
```

---

# 8. 實際問答程式

完成 Markdown、Chunk 與 Index 後，即可執行問答。

目前有四個主要入口。

---

## 8.1 `rag_ans_v2.py`

終端互動式 V2。

執行：

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

也可以單次查詢：

```bash
python rag_ans_v2.py --question "EL817 的 CTR 是多少？"
```

模型與 Index 只載入一次，可以連續詢問多題。

但：

```text
每一題都是獨立 Query
```

上一題不會成為下一題的對話上下文。

---

## 8.2 `rag_ans_v3.py`

終端互動式 V3。

執行：

```bash
python rag_ans_v3.py
```

V3 會在單一問題內啟用 Retrieval Reviewer 與 Iterative Retrieval。

除錯時：

```bash
python rag_ans_v3.py --debug
```

可查看：

```text
Initial Keywords
每輪 Keywords
Irrelevant Evidence
Revised Keywords
Search Round
Stop Reason
Final Evidence
```

同樣不保留上一題的對話歷史。

---

## 8.3 `rag_loop_v2.py`

批次 JSONL V2。

執行：

```bash
python rag_loop_v2.py \
  --input photo_coupler_eval_400_full.jsonl \
  --output rag_model_outputs_v2.jsonl
```

用途：

```text
讀取 JSONL
↓
每行取 question
↓
逐題執行 V2 RAG
↓
輸出 JSONL
```

常用參數：

```bash
--limit 20
--resume
--fail-fast
--verbose
```

其中：

```text
--limit 20
```

只執行前 20 題。

```text
--resume
```

跳過 Output 中已經完成的 ID。

```text
--fail-fast
```

遇到第一題錯誤立即停止。

```text
--verbose
```

顯示詳細 Log。

---

## 8.4 `rag_loop_v3.py`

批次 JSONL V3。

執行：

```bash
python rag_loop_v3.py \
  --input photo_coupler_eval_400_full.jsonl \
  --output rag_model_outputs_v3.jsonl
```

用途與 V2 相同，但每一題內部可能進行最多：

```text
max_search_rounds
```

次 Retrieval。

不同 JSONL 題目仍然完全獨立。

---

# 9. 程式結構

目前主要程式結構：

```text
rag_everlight-main/
├── rag.py
├── rag_loop.py
├── rag_loop_v2.py
├── rag_loop_v3.py
├── rag_ans.py
├── rag_ans_v2.py
├── rag_ans_v3.py
│
└── rag_app/
    ├── config.py
    ├── preprocess/
    ├── chunking/
    ├── retrieval/
    ├── models/
    └── qa/
        ├── engine.py
        ├── pipeline.py
        ├── batch.py
        ├── interactive.py
        ├── prompts.py
        ├── query_tools.py
        ├── filters.py
        └── confidence.py
```

主要用途：

| 檔案 | 用途 |
|---|---|
| `rag.py` | 前處理、Chunk、Index 等 CLI |
| `rag_loop_v2.py` | Batch V2 |
| `rag_loop_v3.py` | Batch V3 |
| `rag_ans_v2.py` | Interactive V2 |
| `rag_ans_v3.py` | Interactive V3 |
| `config.py` | 主要參數唯一設定來源 |
| `engine.py` | Qwen Answer 與 PDF Image 處理 |
| `pipeline.py` | V2 / V3 RAG 共用核心 |
| `batch.py` | JSONL Batch Runner |
| `interactive.py` | 終端互動 Runner |
| `prompts.py` | Query / Review / Answer Prompt |
| `query_tools.py` | Keyword / Proper Noun / Review Parsing |
| `filters.py` | Exact Product Filter |
| `confidence.py` | generated token probability |

---

# 10. 建議的完整執行順序

第一次建立資料時：

```bash
pip install -r requirements.txt

python rag.py paths
python rag.py inspect

python rag.py prepare-html
python rag.py prepare-pdf

python rag.py chunk
python rag.py build-index
```

完成後即可使用：

```bash
python rag_ans_v2.py
```

或：

```bash
python rag_ans_v3.py --debug
```

批次測試：

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output test_v2.jsonl \
  --limit 5
```

```bash
python rag_loop_v3.py \
  --input questions.jsonl \
  --output test_v3.jsonl \
  --limit 5
```

如果：

```text
Markdown
Chunk
Embedding
Index
```

都沒有修改，之後重新啟動問答時不需要再次執行：

```bash
python rag.py build-index
```

可直接執行：

```bash
python rag_ans_v2.py
```

或：

```bash
python rag_ans_v3.py
```

---

# 11. IT 交接快速流程

如果已經取得完整資料與建好的 Index，IT 端最簡單的驗證方式如下：

```bash
pip install -r requirements.txt

python rag.py paths
python rag.py inspect

python rag_ans_v2.py
```

確認基本 Retrieval 可以正常回答後，再測試 V3：

```bash
python rag_ans_v3.py --debug
```

如果需要重新建立資料：

```bash
python rag.py prepare-html
python rag.py prepare-pdf
python rag.py chunk
python rag.py build-index
```

之後即可重新執行 Answer 或 Batch Evaluation。
