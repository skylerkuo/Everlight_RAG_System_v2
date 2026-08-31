# Everlight 本地 RAG 系統

本專案為億光電子技術文件設計的本地 Retrieval-Augmented Generation（RAG）系統，主要處理 HTML、TXT 與 PDF 技術資料，並針對產品型號、規格、表格、頁碼與工程條件進行檢索與問答。

目前系統的核心流程為：

```text
HTML / TXT / PDF
        ↓
Qwen3.5-4B → Markdown
        ↓
Markdown Chunking
        ↓
BGE-M3 Dense + Sparse Retrieval
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
Neighbor Chunk + PDF Page Image
        ↓
Qwen3.5-4B Answer（Thinking OFF）
        ↓
Evidence Verifier（Thinking OFF）
        ↓
PASS ─────────────→ Final Answer
FIX / 需要修正
        ↓
Constrained Correction（Thinking OFF）
        ↓
Final Answer
```

系統將「資料前處理」、「Retrieval」、「Reranking」、「Answer Generation」與「Answer Verification」分開處理，便於獨立測試與調整。

---

## 1. 主要特色

- 支援 HTML、TXT、PDF 技術文件。
- PDF 以 PyMuPDF render 成頁面影像，再由 Qwen3.5-4B 轉為 Markdown。
- 使用 BGE-M3 同時進行 Dense 與 Sparse Retrieval。
- 使用 Weighted RRF 合併 Dense / Sparse 排名。
- 支援 Exact Product Filter，強化產品型號查詢。
- 使用 `BAAI/bge-reranker-v2-m3` 對 Candidate 重新排序。
- Final Top-K 會補入同文件前後相鄰 Chunk。
- PDF Evidence 可同時提供原始頁面影像給 Answer Model 核對。
- Final Answer 預設不啟用 Thinking。
- Answer 完成後再執行 Evidence Verifier，檢查型號、數值、單位、條件、公式與前後矛盾。
- Confidence 功能預設關閉，需要時才手動開啟。
- 支援單題問答與 JSONL Batch Evaluation。

---

## 2. 使用模型

| 模型 | 用途 |
|---|---|
| `Qwen/Qwen3.5-4B` | HTML / PDF 資料整理、Query Analysis、Answer、Verifier、Correction |
| `BAAI/bge-m3` | Dense + Sparse Retrieval |
| `BAAI/bge-reranker-v2-m3` | Candidate Reranking |

目前不需要更換模型即可完成整個 Pipeline。

---

## 3. 資料目錄

預設資料路徑：

```text
/home/????/data_photo_coupler
```

也可以自行指定：

```bash
export RAG_DATA_DIR=/your/path/data_photo_coupler
```

建議原始資料整理如下：

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

### 3.1 HTML / TXT

建議同時保留：

```text
raw/html/<document_id>.html
text/<document_id>.txt
```

若 TXT 已存在，系統會優先使用 TXT；沒有 TXT 時才從 HTML 擷取文字。

### 3.2 PDF

原始 PDF：

```text
raw/pdf/<document_id>.pdf
```

處理方式：

```text
PDF
 ↓
PyMuPDF
 ↓
Page Image
 ↓
Qwen3.5-4B
 ↓
Markdown
```

這種方式適合包含大量表格、圖表、產品選型表與圖片式內容的技術 PDF。

### 3.3 Metadata

Metadata 可由以下來源取得：

```text
everlight.db
```

或：

```text
rag_ready/documents.jsonl
```

`documents.jsonl` 範例：

```json
{"document_id":"abc123","source_kind":"html","title":"Photo Transistor","source_url":"https://www.everlight.com/example/","raw_path":"raw/html/abc123.html","text_path":"text/abc123.txt","language":"zh-TW","page_count":null}
{"document_id":"def456","source_kind":"pdf","title":"Application Note","source_url":"https://www.everlight.com/download/example/","raw_path":"raw/pdf/def456.pdf","text_path":null,"language":"zh-TW","page_count":12}
```

`document_id` 建議使用文件內容的 SHA-256。

---

## 4. 環境安裝

請先依 GPU 與 CUDA 環境安裝適合版本的 PyTorch。

其餘套件：

```bash
pip install -r requirements.txt
```

---

## 5. 建立 RAG 資料

### 5.1 確認路徑

```bash
python rag.py paths
```

### 5.2 檢查來源資料

```bash
python rag.py inspect
```

### 5.3 HTML / TXT → Markdown

```bash
python rag.py prepare-html
```

### 5.4 PDF → Markdown

```bash
python rag.py prepare-pdf
```

### 5.5 建立 Chunk

```bash
python rag.py chunk
```

### 5.6 建立 Index

```bash
python rag.py index
```

處理完成後，主要產物位於：

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

若 Markdown、Chunk 與 Embedding 沒有變動，後續問答不需要重新建立 Index。

---

## 6. Retrieval Pipeline

目前 Retrieval 流程：

```text
User Question
      ↓
Query Analysis
      ↓
BGE-M3 Dense Retrieval
      +
BGE-M3 Sparse Retrieval
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
```

### 6.1 Dense + Sparse Retrieval

BGE-M3 同時提供：

- **Dense Retrieval**：偏重語意相似度。
- **Sparse Retrieval**：偏重 Keyword / Token Matching，對產品型號、數字與專有名詞特別重要。

目前 Weighted RRF 設定：

```python
rrf_dense_weight = 0.40
rrf_sparse_weight = 0.60
rrf_k = 60
```

即：

```text
Dense  = 40%
Sparse = 60%
```

---

## 7. Exact Product Filter

如果問題中包含明確產品型號，例如：

```text
EL817
EL3120
ELM453
```

系統會在 Candidate 中尋找包含該產品型號的 Chunk，取得對應 `document_id`，再保留 Candidate 中屬於相同產品文件的相關 Chunk。

概念如下：

```text
Question
   ↓
EL3120
   ↓
Candidate Top-K
   ↓
找到含 EL3120 的 Chunk
   ↓
取得 document_id
   ↓
保留同文件相關 Candidate
```

這可降低「規格 Chunk 沒再次寫出型號」而被錯誤排除的情況。

---

## 8. BGE Reranker

Candidate 經 Hybrid Retrieval 後，再交給：

```text
BAAI/bge-reranker-v2-m3
```

進行 Query-Document Pair 評分：

```text
Candidate Top-K
      ↓
BGE Reranker
      ↓
重新排序
      ↓
Final Top-K
```

主要設定集中於 `rag_app/config.py`，例如：

```python
candidate_k = 50
reranker_enabled = True
```

`top_k` 請以目前 `config.py` 的實際設定為準。

---

## 9. Neighbor Chunk 與 PDF Evidence

Final Top-K 決定後，系統會補入同一文件的前後相鄰 Chunk。

目前設定：

```python
answer_neighbor_chunk_radius = 1
```

因此每個主要 Chunk 最多搭配：

```text
Previous Chunk
Main Chunk
Next Chunk
```

Neighbor Chunk：

- 不參與 Dense Retrieval。
- 不參與 Sparse Retrieval。
- 不參與 Weighted RRF。
- 不參與 BGE Reranker。
- 只在最後 Answer 階段提供上下文。

若 Final Top-K 來源為 PDF，系統也可把對應頁面影像提供給 Qwen3.5-4B，協助核對表格、圖表與排版資訊。

---

## 10. Answer Generation 與 Verifier

目前 Final Answer **不啟用 Thinking**。

Answer 階段：

```text
Final Top-K
      +
Neighbor Chunk
      +
PDF Page Image
      ↓
Qwen3.5-4B
Thinking OFF
      ↓
Draft Answer
```

`rag_app/qa/engine.py` 中的 Answer 呼叫應使用：

```python
enable_thinking=False
```

### 10.1 Evidence Verifier

Draft Answer 產生後，系統使用同一個 Qwen3.5-4B 進行一次受限檢查。

Verifier 主要檢查：

1. 核心結論是否回答題目。
2. 產品型號是否正確。
3. 關鍵數值是否與 Evidence 一致。
4. 單位是否一致。
5. 測試或操作條件是否一致。
6. 公式與計算結果是否一致。
7. 回答內部是否前後矛盾。

Verifier 不應因為以下情況直接判錯：

- 額外提到其他產品類別，但沒有改變核心答案。
- 額外補充來源未直接支持的內容，但沒有與 Evidence 衝突，也沒有影響核心結論。
- 文字較長、風格不同或缺少非必要細節。

Verifier 可回傳：

```text
pass
fix
insufficient
```

典型流程：

```text
Draft Answer
     ↓
Verifier
     ↓
PASS ─────────────→ Draft Answer = Final Answer

FIX / 需要修正
     ↓
Constrained Correction
     ↓
Final Answer
```

Correction 只允許修正 Verifier 明確指出的問題，不重新自由生成整份答案，也不應加入新的事實。

---

## 11. Confidence 功能

Confidence 功能目前**預設關閉**。

一般問答或 Batch Evaluation 不需要啟用 Confidence，因此預設不額外計算 Generated Token Probability，以降低額外運算與記錄負擔。

若需要進行 Confidence 實驗，再使用對應開關，例如：

```bash
--enable-confidence
```

如果沒有加上此參數，Confidence 維持 Disabled。

---

## 12. 實際問答

### 12.1 互動式問答

```bash
python rag_ans_v2.py
```

例如：

```text
EL817 的 CTR 是多少？
```

下一題可以直接繼續輸入：

```text
EL3120 的總功耗如何計算？
```

每一題彼此獨立，不會將上一題回答帶入下一題：

```text
Question 1
   ↓
完整 Retrieval + Answer + Verification

Question 2
   ↓
重新 Retrieval + Answer + Verification
```

### 12.2 單次問題

```bash
python rag_ans_v2.py --question "EL817 的 CTR 是多少？"
```

---

## 13. Batch JSONL Evaluation

批次測試：

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl
```

最小輸入：

```json
{"id":1,"question":"EL817 的 CTR 是多少？"}
```

多題：

```json
{"id":1,"question":"EL817 的 CTR 是多少？"}
{"id":2,"question":"一般 Photo Transistor 光耦適合哪些工業場合？"}
{"id":3,"question":"EL3120 的總功耗如何計算？"}
```

RAG 推論只使用 `question`。

若輸入資料同時包含：

```text
evaluation_reference
source_url
page_number
source_excerpt
```

這些欄位只供推論完成後 Evaluation 使用，不會提供給 Answer Model 作為 Ground Truth。

### 13.1 Batch Output

Batch Output 可記錄：

```text
question
keywords
retrieval result
final_top_k
draft_answer
model_answer
verifier_verdict
verifier_issues
verifier_parse_ok
verifier_applied_fix
answer_seconds
verifier_seconds
correction_seconds
elapsed_seconds
```

其中：

- `draft_answer`：Verifier 前的原始回答。
- `model_answer`：完成 Verification / Correction 後的最終回答。
- `verifier_verdict`：`pass`、`fix` 或 `insufficient`。
- `verifier_issues`：Verifier 找出的具體問題。
- `verifier_applied_fix`：是否真的執行修正。

這些欄位可用來分析 Verifier 是否改善答案，或是否發生不必要修正。

Confidence 預設關閉，因此一般 Batch 不需依賴 `generated_token_probability`。

---

## 14. 主要設定

主要設定集中於：

```text
rag_app/config.py
```

常用 Retrieval 設定：

```python
candidate_k = 50
rrf_dense_weight = 0.40
rrf_sparse_weight = 0.60
rrf_k = 60
reranker_enabled = True
answer_neighbor_chunk_radius = 1
```

Final Top-K、Answer Token、Verifier Token 與 Correction Token 請以目前 `config.py` 為準。

Verifier 可由 Config 控制，例如：

```python
answer_verifier_enabled = True
```

Final Answer 與 Verifier 目前都使用：

```python
enable_thinking=False
```

---

## 15. 建議執行順序

第一次建立資料：

```bash
python rag.py paths
python rag.py inspect
```

準備 Markdown：

```bash
python rag.py prepare-html
python rag.py prepare-pdf
```

建立 Chunk 與 Index：

```bash
python rag.py chunk
python rag.py index
```

完成後進行問答：

```bash
python rag_ans_v2.py
```

或：

```bash
python rag_ans_v2.py --question "EL817 的 CTR 是多少？"
```

Batch Evaluation：

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl
```

---

## 16. 專案架構重點

```text
                          HTML / TXT / PDF
                                 │
                                 ▼
                              Markdown
                                 │
                                 ▼
                              Chunking
                                 │
                                 ▼
                    BGE-M3 Dense + Sparse
                                 │
                                 ▼
                           Weighted RRF
                                 │
                                 ▼
                         Candidate Top-K
                                 │
                                 ▼
                      Exact Product Filter
                                 │
                                 ▼
                          BGE Reranker
                                 │
                                 ▼
                           Final Top-K
                                 │
                                 ▼
                    Neighbor Chunk / PDF Image
                                 │
                                 ▼
                      Qwen3.5-4B Answer
                         Thinking OFF
                                 │
                                 ▼
                           Draft Answer
                                 │
                                 ▼
                      Qwen3.5-4B Verifier
                         Thinking OFF
                       ┌─────────┴─────────┐
                       │                   │
                     PASS                 FIX
                       │                   │
                       │                   ▼
                       │       Constrained Correction
                       │            Thinking OFF
                       │                   │
                       └─────────┬─────────┘
                                 ▼
                            Final Answer
```

整體設計重點：

```text
資料前處理
→ Hybrid Retrieval
→ Product-aware Filtering
→ Reranking
→ Neighbor / PDF Evidence
→ No-thinking Answer Generation
→ Evidence Verification
→ 必要時受限修正
→ Final Answer
```

此設計讓 Retrieval、Answer 與 Verification 各自獨立，方便後續以同一組測試資料比較 Retrieval 命中率、回答正確率、Verifier 修正率與整體推論時間。

---

## 17. 爬蟲參考程式

專案若包含：

```text
tools/everlight_crawler_reference.py
```

此程式僅供未來重新收集或新增億光公開資料時參考，不是 RAG 啟動的必要程式。

資料收集完成後，只要整理回：

```text
raw/html/
raw/pdf/
text/
everlight.db
rag_ready/documents.jsonl
```

即可重新執行前處理、Chunk 與 Index 建立流程。
