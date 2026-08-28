# Everlight 本地 RAG 系統設計文件

本文件提供給接手與維護工程師，說明資料處理、檢索、V2/V3、批次與終端查詢的責任邊界。

---

## 1. 設計原則

系統主要遵循四個原則：

1. **單一設定來源**：Top-K、Candidate-K、RRF 權重、Reranker、模型與 V3 最大搜尋輪數集中在 `rag_app/config.py`。
2. **批次與互動分離**：`rag_loop_*` 專門讀 JSONL；`rag_ans_*` 專門做終端輸入。
3. **V2/V3 共用核心**：Retrieval、Exact Product Filter、Reranker、Answer 不複製到各入口檔案。
4. **每題獨立**：無論批次或終端連續輸入，不保存對話歷史；V3 的多輪只代表同一題內部重新搜尋。

---

## 2. 入口責任

```text
rag_loop_v2.py
  → JSONL 批次
  → pipeline.ask_v2()

rag_loop_v3.py
  → JSONL 批次
  → pipeline.ask_v3()

rag_ans_v2.py
  → Terminal 多次獨立輸入
  → pipeline.ask_v2()

rag_ans_v3.py
  → Terminal 多次獨立輸入
  → pipeline.ask_v3()
```

入口檔案只負責 CLI，不保存 Retrieval 主要參數。

---

## 3. 共用 QA 模組

```text
rag_app/qa/
├── engine.py
├── pipeline.py
├── batch.py
├── interactive.py
├── prompts.py
├── query_tools.py
├── filters.py
└── confidence.py
```

### `engine.py`

負責：

- 載入 BGE-M3 index
- 載入 Qwen3.5-4B
- 組合 Final Answer Prompt
- 依 final result 對應 PDF page image
- 執行最終回答

### `pipeline.py`

負責真正的 V2/V3 RAG 邏輯。

### `batch.py`

負責 JSONL：

- 讀題目
- `--resume`
- `--limit`
- 每題立即寫 output
- 錯誤處理
- 評估 metadata 隔離
- generated token probability（環境支援時）

### `interactive.py`

負責 Terminal：

- 模型只載入一次
- 可以連續輸入很多問題
- `exit / quit / q` 離開
- 每一題獨立，不傳前一題紀錄

---

## 4. V2 核心流程

```text
Question
  ↓
Qwen Query Analyzer
  ↓
keywords / proper_nouns
  ↓
build_search_query()
  ↓
BGE-M3 Dense + Sparse Retrieval
  ↓
Weighted RRF
  ↓
Candidate-K
  ↓
Document-level Exact Product Filter
  ↓
Cross-Encoder Reranker
  ↓
Final Top-K
  ↓
Qwen3.5-4B Answer
```

### Query Analyzer

輸出：

```json
{
  "keywords": [],
  "proper_nouns": []
}
```

`proper_nouns` 只包含使用者問題明確寫出的型號或系列，不允許模型自行補產品名稱。

### Document-level Exact Product Filter

只有 `proper_nouns` 非空時才啟動。先在原始 BGE Candidate-K 中找出含 exact model name 的命中 chunk，再取得其 `document_id`；之後保留原始 Candidate-K 中所有相同 `document_id` 的 chunks，最後才交給 Reranker。這可避免型號位於 Description、規格位於 Product Features 時，規格 chunk 因本身沒重複產品型號而被刪除。

Filter **不會從 Candidate-K 之外額外擴張 chunks**。如果候選中完全沒有 exact match，仍退回原始 candidates，避免結果被全部清空。

---

## 5. V3 核心流程

V3 在每輪 final Top-K 後加入 Retrieval Reviewer。

```text
Round 1
Question
  ↓
Query Analyzer
  ↓
Retrieval / Filter / Reranker
  ↓
Top-K
  ↓
Reviewer
  ↓
{
  irrelevant_sources: [...],
  revised_keywords: [...]
}
```

只有同時符合：

```text
irrelevant_sources 非空
AND
revised_keywords 非空
AND
keyword 集合真的改變
AND
round < max_search_rounds
```

才進下一輪。

下一輪 Query 仍保留原始使用者問題：

```text
Original Question
+
Search keywords: revised_keywords
+
Exact product/model names: 原始問題中明確出現的 proper_nouns
```

因此 Reviewer 只調整搜尋詞，不取代原問題。

---

## 6. 批次與終端不等於多輪對話

### 批次 JSONL

```text
Question #1 → Answer #1
Question #2 → Answer #2
Question #3 → Answer #3
```

三題彼此獨立。

### 終端連續輸入

```text
問題> A
回答> ...

問題> B
回答> ...
```

B 不會取得 A 的 question 或 answer。

### V3 的「多輪」

只發生在同一題內：

```text
Question A
  ↓
Search Round 1
  ↓
Review
  ↓
Search Round 2（必要時）
  ↓
Review
  ↓
Search Round 3（必要時）
  ↓
Answer A
```

這不是 conversation memory。

---

## 7. Retrieval 參數

唯一來源：

```text
rag_app/config.py
```

目前：

```python
candidate_k = 50
top_k = 7

rrf_dense_weight = 0.40
rrf_sparse_weight = 0.60
rrf_k = 60

reranker_enabled = True
reranker_model_id = "BAAI/bge-reranker-v2-m3"
max_search_rounds = 3
```

BGE-M3 index 搜尋不再硬編碼另一份 Dense/Sparse 權重。

Weighted RRF：

```text
score =
DenseWeight / (rrf_k + dense_rank)
+
SparseWeight / (rrf_k + sparse_rank)
```

---

## 8. HTML/TXT → Markdown

模組：

```text
rag_app/preprocess/html_to_md.py
```

資料順序：

```text
crawler TXT 有資料
  ├─ 是 → 使用 TXT
  └─ 否 → raw HTML → BeautifulSoup → 純文字
                         ↓
                    Qwen3.5-4B
                         ↓
                      Markdown
```

HTML 轉 Markdown 的提示詞沿用原本內容，模型統一使用 Qwen3.5-4B。

---

## 9. PDF → Markdown

模組：

```text
rag_app/preprocess/pdf_to_md.py
```

```text
PDF
  ↓
PyMuPDF render page image
  ↓
前一頁 + 目標頁 + 下一頁（預設 radius=1）
  ↓
Qwen3.5-4B
  ↓
只輸出目標頁 Markdown
```

Page image path 會保留到 chunk metadata。若 PDF chunk 進入 final result，Final Answer 可再次附上原始 page image 讓 Qwen 核對表格、圖、數值與 layout。

---

## 10. Chunking

```text
Markdown
  ↓
Heading / Section 分割
  ↓
Section 內 Block 組合
  ↓
Token 控制
  ↓
Chunk
```

目前：

```text
Target  = 450 tokens
Max     = 650 tokens
Overlap = 70 tokens
Min     = 35 tokens
```

不同 heading 不為了湊 token 強制合併。

---

## 11. Index

```text
chunks.jsonl
  ↓
BGE-M3
  ├─ Dense vectors
  └─ Sparse lexical weights
  ↓
index/
├── dense.npy
├── sparse.jsonl
├── chunks.jsonl
└── index_meta.json
```

Dense 使用 normalization 後 dot product；Sparse 使用 lexical-weight dot product。兩者先各自排序，再由 Weighted RRF 融合。

---

## 12. Batch Fair-Test Boundary

JSONL 可以包含：

```text
ground_truth
source_url
source_excerpt
source_title
page_number
...
```

但推論期間只把：

```text
question
```

送進 RAG。

其他欄位只在該題推論完成後寫進：

```json
"evaluation_reference": {...}
```

因此不會洩漏答案給 Retrieval 或 Qwen。

---

## 13. 擴充原則

若未來新增 V4，不建議複製 `rag_loop_v3.py` 數百行。應在：

```text
rag_app/qa/pipeline.py
```

新增：

```python
ask_v4(...)
```

入口檔只負責呼叫。

同樣地：

- 新 Prompt → `prompts.py`
- Query processing → `query_tools.py`
- Filter → `filters.py`
- JSONL 行為 → `batch.py`
- Terminal 行為 → `interactive.py`
- 主要參數 → `config.py`

避免同一邏輯在多個入口檔案各維護一份。

## Final Top-K Neighbor Context Expansion

為避免「型號、規格、應用」因 Markdown chunking 分散到相鄰 chunk，最終回答階段增加 text-only neighbor expansion。

```text
Candidate Top-50
→ Document-level Exact Product Filter（只在 Top-50 內）
→ Reranker
→ Top-K
→ 每個 Top-K 依 document_id + chunk_index 補 previous/next chunk
→ Qwen final answer
```

重要邊界：

- Exact Product Filter 不會從 Top-50 外補候選；它只保留 Top-50 中屬於 exact-model 命中文件的 chunks。
- Neighbor expansion 發生在 Reranker **之後**，因此不改變 Top-K 排名。
- Neighbor expansion 只增加文字，不把鄰近 PDF page image 加入 multimodal input。
- `[S1]...[Sn]` 仍只對應原始 Final Top-K；相鄰 chunk 是該 evidence group 的 context，不產生新的 Source label。
