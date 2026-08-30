# Everlight 本地 RAG 系統

本專案為億光電子技術文件的本地 Retrieval-Augmented Generation（RAG）系統。

資料來源可包含 HTML、TXT 與 PDF。系統會先將原始資料整理為 Markdown，再進行 Chunk、BGE-M3 Dense + Sparse 混合檢索、Weighted RRF、Exact Product Filter、BGE Reranker，最後由 Qwen3.5-4B 根據檢索到的 Evidence 產生回答。

目前系統採用**單輪 Retrieval**：

- 每一個問題彼此獨立。
- 不保留上一題的對話歷史。
- Retrieval 完成後，由 Qwen3.5-4B 進行最終答案生成。
- 最終 Answer 階段可選擇是否啟用 Thinking。

---

## 1. 系統流程

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
補入同文件前後相鄰 Chunk
        ↓
Qwen3.5-4B
Final Answer
Thinking 可開 / 關
        ↓
移除 Thinking 內容
        ↓
只輸出 Final Answer
```

系統將 Retrieval 與 Answer Generation 分開處理。

前段負責找出最相關的文件內容，最後才由 Qwen3.5-4B 根據 Final Top-K Evidence 與 Neighbor Chunk 進行答案生成。

---

# 2. 資料準備

預設資料路徑：

```text
/home/????/data_photo_coupler
```

也可自行設定：

```bash
export RAG_DATA_DIR=/your/path/data_photo_coupler
```

轉 Markdown 前，來源資料建議整理為：

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

---

## 2.1 HTML / TXT

建議同時保留：

```text
raw/html/<document_id>.html
text/<document_id>.txt
```

若 TXT 已存在，系統會優先使用 TXT。

若沒有 TXT，則會從 HTML 擷取文字。

---

## 2.2 PDF

原始 PDF 放置於：

```text
raw/pdf/<document_id>.pdf
```

後續系統會使用 PyMuPDF 將 PDF render 成頁面影像，再交由 Qwen3.5-4B 轉換成 Markdown。

處理流程：

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

---

## 2.3 Metadata

文件 Metadata 可由：

```text
everlight.db
```

或：

```text
rag_ready/documents.jsonl
```

取得。

`documents.jsonl` 範例：

```json
{"document_id":"abc123","source_kind":"html","title":"Photo Transistor","source_url":"https://www.everlight.com/example/","raw_path":"raw/html/abc123.html","text_path":"text/abc123.txt","language":"zh-TW","page_count":null}
{"document_id":"def456","source_kind":"pdf","title":"Application Note","source_url":"https://www.everlight.com/download/example/","raw_path":"raw/pdf/def456.pdf","text_path":null,"language":"zh-TW","page_count":12}
```

`document_id` 建議使用文件內容的 SHA-256。

---

# 3. 爬蟲參考程式

專案中可放置：

```text
tools/everlight_crawler_reference.py
```

此程式主要提供未來重新收集或新增億光公開資料時參考。

它不是 RAG 啟動必要程式。

爬蟲輸出整理為：

```text
raw/html/
raw/pdf/
text/
everlight.db
rag_ready/documents.jsonl
```

資料收集完成後，即可進入 RAG 前處理：

```bash
python rag.py inspect
python rag.py prepare-html
python rag.py prepare-pdf
python rag.py chunk
python rag.py index
```

---

# 4. 環境安裝

請先依主機 GPU 與 CUDA 環境，自行安裝適合版本的 **PyTorch**。

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

主要用途：

| 模型 | 用途 |
|---|---|
| Qwen3.5-4B | HTML / PDF 資料整理與最終答案生成 |
| BGE-M3 | Dense + Sparse Retrieval |
| BGE Reranker v2 M3 | Candidate 重排序 |

---

# 5. 資料前處理

先確認資料路徑：

```bash
python rag.py paths
```

確認來源文件：

```bash
python rag.py inspect
```

---

## 5.1 HTML / TXT → Markdown

```bash
python rag.py prepare-html
```

---

## 5.2 PDF → Markdown

```bash
python rag.py prepare-pdf
```

PDF 會先 render 成頁面影像：

```text
PDF
 ↓
Page Image
 ↓
Qwen3.5-4B
 ↓
Markdown
```

---

## 5.3 建立 Chunk

```bash
python rag.py chunk
```

---

## 5.4 建立 Index

```bash
python rag.py index
```

---

## 5.5 處理結果

主要產物位於：

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

如果：

- Markdown 沒有變動
- Chunk 沒有變動
- Embedding 沒有變動

後續問答時不需要重新建立 Index。

---

# 6. Retrieval 流程

目前 Retrieval 主要流程：

```text
User Question
      ↓
BGE-M3
Dense Retrieval
+
Sparse Retrieval
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

---

## 6.1 BGE-M3 Dense + Sparse Retrieval

BGE-M3 同時產生：

```text
Dense Score
Sparse Score
```

Dense Retrieval 主要負責：

```text
語意相似度
```

Sparse Retrieval 則較偏向：

```text
Keyword / Token Matching
```

之後使用 Weighted RRF 合併兩組搜尋結果。

目前設定：

```python
rrf_dense_weight = 0.40
rrf_sparse_weight = 0.60
rrf_k = 60
```

也就是：

```text
Dense  = 40%
Sparse = 60%
```

---

# 7. Exact Product Filter

如果問題中包含明確產品型號，例如：

```text
EL817
EL3120
ELM453
```

系統會先檢查 Candidate 中哪些 Chunk 包含該產品型號。

例如：

```text
Question
    ↓
EL3120
    ↓
Candidate Top-K
    ↓
尋找包含 EL3120 的 Chunk
    ↓
取得對應 document_id
```

之後會保留 Candidate 中同一個 `document_id` 的其他 Chunk。

因此即使某個規格 Chunk 沒有再次出現：

```text
EL3120
```

只要它屬於相同產品文件，就不會因為沒有產品名稱而直接被刪除。

---

# 8. BGE Reranker

Candidate 經過初步 Retrieval 後，再交給：

```text
BAAI/bge-reranker-v2-m3
```

進行 Query-Document Pair 評分。

流程：

```text
Candidate Top-K
      ↓
BGE Reranker
      ↓
重新計算相關性
      ↓
Final Top-K
```

目前主要設定：

```python
candidate_k = 50
top_k = 7
reranker_enabled = True
```

例如：

```text
Candidate = 50
     ↓
Reranker
     ↓
Final Top-K = 7
```

只有 Final Top-K 會成為最終 Answer Model 的主要 Evidence。

---

# 9. Neighbor Chunk

Reranker 選出 Final Top-K 後，系統會從同一份文件補入前後相鄰 Chunk。

例如：

```text
Previous Chunk
      ↓
Main Chunk
      ↓
Next Chunk
```

目前設定：

```python
answer_neighbor_chunk_radius = 1
```

因此每一個 Final Top-K Chunk 最多補：

```text
前 1 Chunk
主 Chunk
後 1 Chunk
```

Neighbor Chunk：

- 不參與 Dense Retrieval。
- 不參與 Sparse Retrieval。
- 不參與 Weighted RRF。
- 不參與 BGE Reranker。
- 只提供給最後的 Qwen Answer Model。

用途是避免：

```text
答案剛好跨 Chunk 邊界
```

導致模型只看到部分內容。

---

# 10. Final Answer Generation

Final Top-K 與 Neighbor Context 建立完成後，會交給：

```text
Qwen/Qwen3.5-4B
```

進行最終回答。

流程：

```text
Final Top-K
      +
Neighbor Chunk
      +
PDF Page Image
      ↓
Qwen3.5-4B
      ↓
Final Answer
```

Final Answer Model 只應根據 Retrieval Evidence 回答問題。

---

# 11. Thinking 模式

目前系統支援讓 Qwen3.5-4B 在**最終 Answer Generation 階段**啟用 Thinking。

設計目的為：

```text
Retrieval
    ↓
取得 Evidence
    ↓
Qwen Thinking
    ↓
分析多份 Evidence
    ↓
比較規格 / 數值
    ↓
必要時計算
    ↓
產生 Final Answer
```

Thinking 主要適合處理：

- 多個 Evidence 的資訊整合
- 規格比較
- 公式計算
- 條件判斷
- 技術文件交叉判讀
- 從多個 Chunk 中整理出最終答案

---

## 11.1 Thinking 開關位置

Thinking 的核心控制位於：

```text
rag_app/models/qwen35_vl.py
```

`generate()` 提供：

```python
enable_thinking: bool = False
```

例如：

```python
def generate(
    self,
    prompt: str,
    image_paths=None,
    image_labels=None,
    system=None,
    max_new_tokens: int = 1024,
    enable_thinking: bool = False,
) -> str:
```

因此預設狀態：

```text
Thinking = OFF
```

只有呼叫端明確指定：

```python
enable_thinking=True
```

才會進入 Thinking。

---

# 12. 最終回答開啟 Thinking

最終 Answer Generation 位於：

```text
rag_app/qa/engine.py
```

例如：

```python
answer = self.answer_model.generate(
    build_answer_prompt(question, context_text),
    image_paths=images,
    system=ANSWER_SYSTEM,
    max_new_tokens=self.settings.qwen_max_new_tokens_answer,
    enable_thinking=True,
)
```

其中：

```python
enable_thinking=True
```

代表：

```text
Final Answer Thinking = ON
```

因此模型會先進行推理，再產生答案。

---

# 13. 關閉 Final Answer Thinking

如果希望完全關閉 Thinking，只需要修改：

```text
rag_app/qa/engine.py
```

將：

```python
enable_thinking=True,
```

改成：

```python
enable_thinking=False,
```

完整範例：

```python
answer = self.answer_model.generate(
    build_answer_prompt(question, context_text),
    image_paths=images,
    system=ANSWER_SYSTEM,
    max_new_tokens=self.settings.qwen_max_new_tokens_answer,
    enable_thinking=False,
)
```

此時流程變成：

```text
Final Evidence
      ↓
Qwen3.5-4B
Thinking OFF
      ↓
直接產生答案
```

---

# 14. Thinking 與 Final Answer 分離

Qwen 開啟 Thinking 後，原始生成文字可能類似：

```text
先檢查 S1...
比較 S2...
計算規格...
確認結果...
</think>

EL3120 的總功耗為 122.6 mW。
```

如果直接 Decode，Thinking 與最終答案可能會一起出現在：

```json
{
  "model_answer": "思考內容...</think>最終答案..."
}
```

因此目前：

```text
rag_app/models/qwen35_vl.py
```

會在 Decode 後自動移除 Thinking。

核心處理：

```python
decoded_text = self.processor.decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
).strip()

if enable_thinking and "</think>" in decoded_text:
    decoded_text = decoded_text.rsplit("</think>", 1)[-1].strip()

return decoded_text
```

也就是：

```text
Qwen 原始輸出
      ↓
Thinking
      ↓
</think>
      ↓
Final Answer
      ↓
rsplit("</think>", 1)
      ↓
刪除 Thinking
      ↓
只保留 Final Answer
```

例如原始結果：

```text
我需要先比較 EL3120 的功耗參數……
PEmitter = ...
PInternal = ...
POutput = ...
</think>

EL3120 的總功耗為 122.6 mW。
```

最後系統只回傳：

```text
EL3120 的總功耗為 122.6 mW。
```

因此：

**模型可以進行 Thinking，但使用者與 Batch JSONL 不會看到 Thinking 內容。**

---

# 15. Thinking 設定總結

目前建議設定：

```text
資料前處理
Qwen3.5-4B
Thinking OFF

        ↓

Retrieval
BGE-M3
不適用

        ↓

Weighted RRF
不適用

        ↓

Exact Product Filter
不適用

        ↓

BGE Reranker
不適用

        ↓

Final Answer
Qwen3.5-4B
Thinking ON

        ↓

移除 </think> 前內容

        ↓

只輸出 Final Answer
```

如果要改 Final Answer Thinking：

### 開啟

```python
enable_thinking=True
```

### 關閉

```python
enable_thinking=False
```

修改位置：

```text
rag_app/qa/engine.py
```

---

# 16. 實際問答

啟動互動式問答：

```bash
python rag_ans_v2.py
```

進入後可連續輸入問題。

例如：

```text
EL817 的 CTR 是多少？
```

下一題：

```text
EL3120 的總功耗如何計算？
```

每一題彼此獨立：

```text
Question 1
   ↓
完整 Retrieval + Answer

Question 2
   ↓
重新 Retrieval + Answer
```

不會將上一題的回答內容帶入下一題。

---

## 16.1 單次問題

也可以直接：

```bash
python rag_ans_v2.py --question "EL817 的 CTR 是多少？"
```

執行完成後直接輸出答案。

---

# 17. Batch JSONL 測試

批次測試：

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl
```

最小輸入格式：

```json
{"id":1,"question":"EL817 的 CTR 是多少？"}
```

多題：

```json
{"id":1,"question":"EL817 的 CTR 是多少？"}
{"id":2,"question":"一般 Photo Transistor 光耦適合哪些工業場合？"}
{"id":3,"question":"EL3120 的總功耗如何計算？"}
```

Batch 推論時 RAG 只會讀取：

```text
question
```

Ground Truth：

```text
evaluation_reference
```

以及 Reference Source：

```text
source_url
page_number
source_excerpt
```

只供推論完成後進行 Evaluation，不會提供給 RAG Answer Model。

---

# 18. Batch Output

Batch Output 可包含：

```text
question
keywords
retrieval result
final_top_k
model_answer
generated_token_probability
elapsed_seconds
```

其中：

```text
model_answer
```

只保存最終答案。

即使：

```python
enable_thinking=True
```

Thinking 內容也會在：

```text
qwen35_vl.py
```

中移除，因此不會寫入：

```text
model_answer
```

---

# 19. 主要設定

主要設定集中於：

```text
rag_app/config.py
```

常用參數：

```python
candidate_k = 50
top_k = 7

rrf_dense_weight = 0.40
rrf_sparse_weight = 0.60
rrf_k = 60

reranker_enabled = True

answer_neighbor_chunk_radius = 1
```

Final Answer 最大生成 Token 也可在 Config 中設定，例如：

```python
qwen_max_new_tokens_answer = 1600
```

如果開啟 Thinking，需要注意 Thinking 本身也會使用 Generation Token，因此可依 GPU、延遲與答案完整度調整：

```text
1000
1500
1600
2048
```

---

# 20. 建議執行順序

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

建立 Chunk：

```bash
python rag.py chunk
```

建立 Index：

```bash
python rag.py index
```

完成後即可進行問答：

```bash
python rag_ans_v2.py
```

或直接輸入單一問題：

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

# 21. 整體架構摘要

```text
                         ┌────────────────────┐
                         │ HTML / TXT / PDF   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Markdown           │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Chunking           │
                         └─────────┬──────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │ BGE-M3 Dense + Sparse Retrieval    │
              └──────────────────┬─────────────────┘
                                 │
                                 ▼
                         ┌───────────────────┐
                         │ Weighted RRF      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Candidate Top-K   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Exact Product     │
                         │ Filter            │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ BGE Reranker      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Final Top-K       │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Neighbor Chunk    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Qwen3.5-4B        │
                         │ Thinking          │
                         │ ON / OFF          │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Remove Thinking   │
                         │ </think>          │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Final Answer      │
                         └───────────────────┘
```

整體設計重點為：

```text
資料前處理
→ Hybrid Retrieval
→ Product-aware Filtering
→ Reranking
→ Neighbor Context
→ Thinking-based Answer Generation
→ Final Answer Only
```

使 Retrieval、Reasoning 與最終答案輸出彼此分離，同時保留工程文件檢索與技術問題推理能力。