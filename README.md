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

# 2. 轉 Markdown 前，來源資料要先整理成什麼格式

這一節是交接時最重要的資料規格。

`prepare-html` 與 `prepare-pdf` **不是直接掃描任意資料夾中的所有檔案**。  
程式會先讀取「文件 Metadata」，取得每一份文件的：

```text
document_id
source_kind
title
source_url
raw_path
text_path
language
page_count
```

再依 Metadata 指向的實際 HTML / TXT / PDF 檔案進行處理。

因此，IT 若要加入新的網站資料或 PDF，必須先把資料整理成下列格式，再執行 Markdown 轉換。

---

## 2.1 DATA_DIR 最少需要的來源資料結構

預設資料根目錄：

```text
/home/skyler/Desktop/rag_system/data_photo_coupler
```

也就是程式中的：

```python
DATA_DIR
```

在進行 Markdown 轉換以前，建議整理成：

```text
DATA_DIR/
├── raw/
│   ├── html/
│   │   ├── <document_id_1>.html
│   │   ├── <document_id_2>.html
│   │   └── ...
│   │
│   └── pdf/
│       ├── <document_id_3>.pdf
│       ├── <document_id_4>.pdf
│       └── ...
│
├── text/
│   ├── <document_id_1>.txt
│   ├── <document_id_2>.txt
│   └── ...
│
├── everlight.db
│
└── rag_ready/
    └── documents.jsonl
```

其中：

```text
raw/html/
```

放爬蟲下載的原始 HTML。

```text
raw/pdf/
```

放爬蟲下載的原始 PDF。

```text
text/
```

放 HTML 已經由爬蟲抽取出的純文字 TXT。

```text
everlight.db
```

是原爬蟲使用的 Metadata Database。

```text
rag_ready/documents.jsonl
```

是沒有使用 `everlight.db` 時，可以提供給 RAG 的替代 Metadata 格式。

---

## 2.2 document_id 的用途

每一份來源文件都必須有一個唯一的：

```text
document_id
```

目前原爬蟲使用：

```text
SHA-256
```

作為 `document_id`。

例如：

```text
8a3c0a9d8f7e.......
```

建議檔名也直接使用相同的 `document_id`：

```text
raw/html/8a3c0a9d8f7e....html
text/8a3c0a9d8f7e....txt
```

HTML 與其對應的 TXT 必須使用同一個 Metadata record，兩者代表的是**同一份網頁文件**。

PDF 則例如：

```text
raw/pdf/f19b3225cac3c9ac....pdf
```

對應：

```text
document_id = f19b3225cac3c9ac...
```

`document_id` 不一定技術上必須是 SHA-256，但必須：

```text
1. 每份文件唯一
2. 不要重複
3. 同一份文件在後續 Markdown、Chunk、Index 階段保持不變
```

為了與目前系統一致，建議繼續使用 SHA-256。

---

# 2.3 Metadata 有兩種提供方式

目前程式支援兩種 Metadata 來源：

```text
方法 A：everlight.db
方法 B：rag_ready/documents.jsonl
```

讀取優先順序是：

```text
everlight.db 存在且可讀到資料
        ↓
優先使用 everlight.db

如果 DB 不存在、無法讀取或沒有有效資料
        ↓
改讀 rag_ready/documents.jsonl
```

因此：

> 若 `everlight.db` 中已經存在有效資料，修改 `documents.jsonl` 不會取代 DB 內容。

IT 在匯入新資料時必須清楚目前部署環境是使用哪一種 Metadata 來源。

---

# 2.4 方法 A：維持原爬蟲 `everlight.db` 格式

目前 RAG 會從 SQLite 執行類似以下查詢：

```sql
SELECT
    dv.sha256,
    dv.title,
    dv.raw_path,
    dv.text_path,
    dv.page_count,
    u.url,
    u.kind,
    u.language
FROM document_versions dv
JOIN urls u ON u.id = dv.url_id
WHERE dv.is_current = 1
```

因此原爬蟲 DB 至少需要能提供下列欄位。

### `urls`

```text
id
url
kind
language
```

其中：

```text
kind = "html"
```

代表 HTML 文件。

```text
kind = "pdf"
```

代表 PDF 文件。

### `document_versions`

```text
url_id
sha256
title
raw_path
text_path
page_count
is_current
```

其中：

```text
sha256
```

會被當成：

```text
document_id
```

而：

```text
is_current = 1
```

才會被 RAG 當成目前有效版本。

---

## 2.4.1 HTML 在 DB 中應該對應成什麼樣子

例如：

```text
document_id:
abc123...

kind:
html

title:
Photo Transistor | LED產業的領導廠商 | 億光電子

url:
https://www.everlight.com/...

raw_path:
raw/html/abc123....html

text_path:
text/abc123....txt

language:
zh-TW

page_count:
NULL
```

對應實體檔案：

```text
DATA_DIR/
├── raw/html/abc123....html
└── text/abc123....txt
```

---

## 2.4.2 PDF 在 DB 中應該對應成什麼樣子

例如：

```text
document_id:
def456...

kind:
pdf

title:
Photo Coupler Selection Guide

url:
https://www.everlight.com/download/...

raw_path:
raw/pdf/def456....pdf

text_path:
NULL

language:
zh-TW

page_count:
35
```

對應實體檔案：

```text
DATA_DIR/
└── raw/pdf/def456....pdf
```

PDF 不需要先轉 TXT。

PDF 後續會直接：

```text
PDF
↓
PyMuPDF Render
↓
Page Image
↓
Qwen3.5-4B
↓
Markdown
```

---

# 2.5 方法 B：使用 `rag_ready/documents.jsonl`

如果 IT 不需要沿用原本的爬蟲 DB，也可以直接建立：

```text
DATA_DIR/rag_ready/documents.jsonl
```

這是最容易交接、也最容易人工產生的 Metadata 格式。

格式是：

```text
一行 = 一份來源文件
```

不是一整個 JSON Array。

也就是：

```json
{"document_id":"doc1", ...}
{"document_id":"doc2", ...}
{"document_id":"doc3", ...}
```

而不是：

```json
[
  {"document_id":"doc1"},
  {"document_id":"doc2"}
]
```

---

## 2.5.1 HTML 的 `documents.jsonl` 範例

假設有：

```text
raw/html/abc123.html
text/abc123.txt
```

則 `documents.jsonl` 可寫：

```json
{"document_id":"abc123","source_kind":"html","title":"Photo Transistor | LED產業的領導廠商 | 億光電子","source_url":"https://www.everlight.com/photo_coupler_igbt_ssr/category-photo_transistor/","raw_path":"raw/html/abc123.html","text_path":"text/abc123.txt","language":"zh-TW","page_count":null}
```

欄位意義：

| 欄位 | HTML 是否需要 | 說明 |
|---|---|---|
| `document_id` | 必要 | 唯一文件 ID |
| `source_kind` | 必要 | 必須填 `html` |
| `title` | 強烈建議 | 原始網頁標題 |
| `source_url` | 強烈建議 | 原始網頁 URL |
| `raw_path` | 必要 | 相對於 `DATA_DIR` 的 HTML 路徑 |
| `text_path` | 建議 | 爬蟲已抽出的 TXT；若無可填 `null` |
| `language` | 選填 | 例如 `zh-TW`、`en` |
| `page_count` | 不需要 | HTML 通常填 `null` |

---

## 2.5.2 PDF 的 `documents.jsonl` 範例

假設：

```text
raw/pdf/def456.pdf
```

則：

```json
{"document_id":"def456","source_kind":"pdf","title":"Photo Coupler Selection Guide","source_url":"https://www.everlight.com/download/photo-coupler-selection-guide/","raw_path":"raw/pdf/def456.pdf","text_path":null,"language":"zh-TW","page_count":35}
```

欄位：

| 欄位 | PDF 是否需要 | 說明 |
|---|---|---|
| `document_id` | 必要 | 唯一文件 ID |
| `source_kind` | 必要 | 必須填 `pdf` |
| `title` | 強烈建議 | PDF 文件名稱 |
| `source_url` | 強烈建議 | PDF 原始下載網址 |
| `raw_path` | 必要 | 相對於 `DATA_DIR` 的 PDF 路徑 |
| `text_path` | 不使用 | 建議 `null` |
| `language` | 選填 | 文件主要語言 |
| `page_count` | 建議 | PDF 頁數；實際處理仍會直接讀 PDF 頁數 |

---

# 2.6 `raw_path` 與 `text_path` 要怎麼填

建議全部使用：

```text
相對於 DATA_DIR 的路徑
```

例如：

```text
raw/html/abc123.html
text/abc123.txt
raw/pdf/def456.pdf
```

不要在 Metadata 中寫死：

```text
/home/skyler/Desktop/...
```

否則交接到其他 IT 主機時會失去可攜性。

程式實際會用：

```python
DATA_DIR / raw_path
```

或：

```python
DATA_DIR / text_path
```

找到檔案。

---

# 2.7 HTML 為什麼同時有 HTML 與 TXT

HTML 文件的處理邏輯是：

```text
Metadata
↓
有 text_path？
├─ 有，而且 TXT 檔存在
│      ↓
│   直接使用 TXT
│
└─ 沒有 / TXT 不存在
       ↓
    使用 raw_path 的 HTML
       ↓
    BeautifulSoup
       ↓
    自動抽取純文字
```

因此最建議的爬蟲輸出是：

```text
原始 HTML
+
已抽取 TXT
+
Metadata
```

例如：

```text
raw/html/abc123.html
text/abc123.txt
```

這樣 RAG 會直接使用：

```text
text/abc123.txt
```

而不需要重新從 HTML 抽文字。

如果只有：

```text
raw/html/abc123.html
```

也可以處理。

這時 `text_path` 可設：

```json
"text_path": null
```

程式會使用 BeautifulSoup 抽取文字，並把中間 TXT 寫到：

```text
rag_v6/txt/html/<document_id>.txt
```

---

# 2.8 TXT 在進 Qwen 以前建議保留哪些內容

TXT 的目的是保存網頁中的**可檢索技術文字**。

應盡量保留：

```text
產品名稱
產品型號
產品系列
Section Heading
產品描述
產品特性
技術規格
數值
單位
封裝
CTR
Isolation Voltage
CMTI
工作溫度
應用場合
表格文字
下載文件名稱
```

例如好的 TXT：

```text
8Pin Wide Body

The ELW3120 consists of an infrared light emitting diodes and integrated
high gain, high-speed photo detectors.

The device is housed in an 8 pin DIP wide body package and available
in SMD package option.

產品特性
輸出電流最大 2.5A
保證性能從 -40°C 至 110°C
輸入和輸出之間的隔離電壓 5000Vrms
外部爬電距離大於 10 mm

產品應用
隔離驅動 IGBT/Power MOSFET
不間斷電源供應
變頻器
```

不需要特別自己先整理成 Markdown。

因為：

```text
TXT
↓
Qwen3.5-4B
↓
Markdown
```

Markdown 的 Heading、Bullet、Table 會由 Qwen 再整理。

---

## 2.8.1 不要只留下產品型號

不建議整理成：

```text
ELW3120
2.5A
5000Vrms
```

這樣會失去：

```text
數值與欄位的關係
產品描述
產品應用
上下文
```

應盡量保留原始文字結構。

---

## 2.8.2 可以移除的 HTML 雜訊

爬蟲若能先移除以下內容會比較乾淨：

```text
Cookie banner
JavaScript
CSS
重複 Navigation
Footer 導航
社群按鈕
完全無內容的下載 Placeholder
重複 Header
```

即使沒有完全移除，後續 Qwen Markdown Cleaning Prompt 仍會再做一次整理。

但：

> 不要因為看起來像網站模板，就把產品名稱、產品分類、規格標題或技術內容刪掉。

---

# 2.9 PDF 不要先拆成 TXT

PDF 進入本專案前，只需要：

```text
原始 PDF
+
Metadata
```

例如：

```text
raw/pdf/def456.pdf
```

後續系統會自己產生：

```text
page_images/def456/page_0001.png
page_images/def456/page_0002.png
...
```

再逐頁轉成：

```text
md/pdf/def456/page_0001.md
md/pdf/def456/page_0002.md
...
```

因此不要自行把 PDF 全部轉成一個 TXT 再交給 `prepare-pdf`。

---

# 2.10 一份完整的「轉 Markdown 前」資料範例

假設目前有：

```text
1 個產品網頁
1 份 PDF Application Note
```

建議整理：

```text
data_photo_coupler/
├── raw/
│   ├── html/
│   │   └── abc123.html
│   │
│   └── pdf/
│       └── def456.pdf
│
├── text/
│   └── abc123.txt
│
└── rag_ready/
    └── documents.jsonl
```

`documents.jsonl`：

```json
{"document_id":"abc123","source_kind":"html","title":"8Pin Wide Body | 億光電子","source_url":"https://www.everlight.com/example-product/","raw_path":"raw/html/abc123.html","text_path":"text/abc123.txt","language":"zh-TW","page_count":null}
{"document_id":"def456","source_kind":"pdf","title":"IGBT Gate Driver Application Note","source_url":"https://www.everlight.com/download/example.pdf","raw_path":"raw/pdf/def456.pdf","text_path":null,"language":"zh-TW","page_count":12}
```

確認這些檔案真的存在：

```text
DATA_DIR/raw/html/abc123.html
DATA_DIR/text/abc123.txt
DATA_DIR/raw/pdf/def456.pdf
DATA_DIR/rag_ready/documents.jsonl
```

之後才執行：

```bash
python rag.py inspect
```

再進行：

```bash
python rag.py prepare-html
python rag.py prepare-pdf
```

---

# 2.11 `python rag.py inspect` 應該先確認什麼

完成來源整理後，先執行：

```bash
python rag.py inspect
```

主要確認：

```text
raw_html_files
raw_pdf_files
crawler_txt_files
db_html_documents
db_pdf_documents
db_pdf_pages
```

如果：

```text
raw_html_files > 0
```

但：

```text
db_html_documents = 0
```

通常代表：

```text
HTML 檔案雖然存在
但 Metadata 沒有正確建立
```

同樣地，如果 PDF 已經放到：

```text
raw/pdf/
```

但 Metadata 沒有：

```text
source_kind = "pdf"
raw_path = "raw/pdf/xxx.pdf"
```

`prepare-pdf` 也不會處理它。

所以：

> 「檔案存在」與「Metadata 有登記」兩件事都必須成立。

---

# 2.12 Markdown 轉換之後會自動建立什麼

來源格式正確後：

```bash
python rag.py prepare-html
```

會產生：

```text
rag_v6/md/html/<document_id>.md
```

Markdown Front Matter 會保存：

```text
document_id
source_kind
title
source_url
source_raw_path
source_txt_path
language
```

PDF：

```bash
python rag.py prepare-pdf
```

則會產生：

```text
rag_v6/md/pdf/<document_id>/page_0001.md
rag_v6/md/pdf/<document_id>/page_0002.md
...
```

並保存：

```text
document_id
source_kind
title
source_url
source_raw_path
page_number
page_count
page_image
vlm_context_pages
vlm_context_radius
language
```

另外會更新：

```text
rag_v6/manifest.jsonl
```

因此：

```text
原始來源 Metadata
        ↓
Markdown Front Matter
        ↓
Chunk Metadata
        ↓
Retrieval Evidence
```

來源 URL 與 Document ID 會一路被保留下來。

---

# 2.13 交付時建議 IT 遵守的資料新增流程

未來若要加入新資料，建議固定使用以下順序：

```text
① 下載原始 HTML / PDF
        ↓
② HTML 額外抽取 TXT（可選，但建議）
        ↓
③ 產生唯一 document_id
        ↓
④ 將檔案放到：
   raw/html/
   raw/pdf/
   text/
        ↓
⑤ 更新 everlight.db
   或 rag_ready/documents.jsonl
        ↓
⑥ python rag.py inspect
        ↓
⑦ prepare-html / prepare-pdf
        ↓
⑧ chunk
        ↓
⑨ index
        ↓
⑩ 問答測試
```

這樣可以確保新文件的：

```text
原始檔案
來源 URL
標題
document_id
Markdown
Chunk
Index
```

可以完整追溯。

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
