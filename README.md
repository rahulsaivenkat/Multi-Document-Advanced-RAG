# PaperMind AI — Multi-Format RAG Application

PaperMind AI is an enterprise-grade, evidence-grounded Retrieval-Augmented Generation (RAG) backend engineered in Python and FastAPI. It enables users to upload heterogeneous document formats, ask natural language questions across their content, and receive accurate answers with verifiable, format-specific source citations.

---

## 1. System Architecture & RAG Pipeline

```text
[ Document Upload ] (.pdf, .txt, .md, .docx, .csv, .html, .json, .py)
         │
         ▼
[ Format-Specific Parsers & AST Extractors ]
         │
         ▼
[ Format-Aware Chunking (Preserving Exact Location Metadata) ]
         │
         ├────────────────────────────────────────┐
         ▼                                        ▼
[ ChromaDB Persistent Vector Store ]    [ Persistent BM25 Keyword Index ]
  (Dense SentenceTransformer Embeddings)   (Sparse Tokenizer & Matcher)
         │                                        │
         └───────────────────┬────────────────────┘
                             │
                             ▼
               [ Reciprocal Rank Fusion (RRF) ]
                             │
                             ▼
              [ CrossEncoder Neural Reranking ]
           (cross-encoder/ms-marco-MiniLM-L-6-v2)
                             │
                             ▼
        [ Evidence Sufficiency & Grounding Gate ]
                             │
                             ▼
          [ Groq LLM (llama-3.3-70b-versatile) ]
                             │
                             ▼
      [ Citation Verification & Source Attribution ]
                             │
                             ▼
               [ Structured JSON Response ]
```

---

## 2. Supported Formats & Source Citation Types

PaperMind AI eliminates hallucinated source citations by strictly constructing citation tags from real parsed document metadata:

| Extension | Format | Extraction Strategy | Verifiable Source Citation |
| :--- | :--- | :--- | :--- |
| **`.pdf`** | Adobe PDF | Page-level extraction using PyMuPDF | 1-based Page Number (`Page 4`) |
| **`.txt`** | Plain Text | Encoding detection (UTF-8, Latin-1, CP1252) | 1-based Line Range (`Lines 12-35`) |
| **`.md`** | Markdown | Heading hierarchy tracking + code blocks | Section Heading Path + Lines (`Setup > Windows [Lines 45-60]`) |
| **`.docx`** | Microsoft Word | Paragraph ordering & table cell coordinates | Paragraph Index (`Para #14`) or Table Coordinates (`Table #1, Row 2`) |
| **`.csv`** | CSV Records | Structured records retaining column names | Row Number accounting for Header + Columns (`Row 15 [Cols: ID, Price]`) |
| **`.html`** | HTML Webpages | DOM sanitization, script removal, tables | Section Heading Path + Tag (`Overview > Features [Block #8]`) |
| **`.json`** | JSON Data | RFC 6901 JSON Pointer path traversal | Exact JSON Pointer Path (`Path: /servers/0/host`) |
| **`.py`** | Python Source | AST syntax analysis (classes, functions) | Symbol Name + Exact Lines (`Symbol: def calculate() [Lines 40-55]`) |

---

## 3. Prerequisites on Windows

- **Operating System:** Windows 10 or Windows 11
- **Python:** Python 3.10 to 3.13 (64-bit) installed and added to `PATH`
- **PowerShell:** Windows PowerShell 5.1+ or PowerShell 7+
- **Groq API Key:** Free account from [https://console.groq.com](https://console.groq.com)

---

## 4. Windows Setup & Installation

Open Windows PowerShell and navigate to the project directory:

```powershell
cd c:\Users\ASUS\MYPROJECTS\New_RAG_System
```

### Step 1: Create and Activate Virtual Environment

```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> [!NOTE]
> If you encounter an execution policy error (`running scripts is disabled on this system`), run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

### Step 2: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Safely Configuring Your Groq API Key

Your API key must remain strictly confidential and never be committed to git or exposed to assistants.

1. In the project root, make a copy of `.env.example` named `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` in Notepad or your preferred editor:
   ```powershell
   notepad .env
   ```
3. Locate the `GROQ_API_KEY=` line and paste your actual secret key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```
4. Save and close the file.
5. The `.env` file is already listed in `.gitignore` to prevent accidental commits.

---

## 6. Running the Application

PaperMind AI uses **FastAPI** as the backend RAG engine and **Streamlit** as the interactive frontend interface.

### Step 1: Start the FastAPI Backend

Open a PowerShell terminal and run:

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Step 2: Start the Streamlit Frontend

Open a second PowerShell terminal in the project directory and launch Streamlit:

```powershell
.\.venv\Scripts\streamlit run streamlit_app.py
```

Streamlit will automatically open in your default browser at:
- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)

From the Streamlit UI, you can:
- Drag-and-drop any of the 8 supported document formats.
- View real-time system status and indexed document inventory.
- Ask questions and explore verifiable citations with exact page/line/row coordinates.
- Inspect diagnostic telemetry (dense vs BM25 keyword matches, RRF fusion, CrossEncoder reranking).

---

## 7. Example API Usage via PowerShell

### A. Check System Health

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET | ConvertTo-Json
```

### B. Upload Documents

Upload one or more files (.pdf, .txt, .md, .docx, .csv, .html, .json, .py):

```powershell
$form = @{
    files = Get-Item -Path ".\sample.pdf"
}
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/upload" -Method POST -Form $form
$response | ConvertTo-Json -Depth 5
```

### C. Ask a Grounded Question

```powershell
$body = @{
    question = "What are the primary performance metrics reported in the evaluation?"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method POST -ContentType "application/json" -Body $body
$response | ConvertTo-Json -Depth 6
```

### D. List Indexed Documents

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents" -Method GET | ConvertTo-Json -Depth 4
```

### E. Delete a Document

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/doc_12345" -Method DELETE
```

---

## 8. Persistent Storage Details

- **ChromaDB (`data/storage/chroma`):** Stores precomputed dense vector embeddings for indexed chunks. Persisted to disk with SQLite and Parquet.
- **BM25 Inverted Index (`data/storage/bm25`):** Stores tokenized chunk corpus and lexical mappings for exact identifier, function, and keyword lookup.
- **Document Inventory (`data/storage/documents.json`):** Tracks file hashes for duplicate prevention, upload timestamps, and active chunk mappings.
- **Upload Staging (`data/uploads`):** Staged uploaded files for reference.

When a document is deleted via `DELETE /documents/{id}`, its chunks are completely purged from both ChromaDB, BM25, and disk.

---

## 9. Running Automated Tests

Run the full automated test suite using pytest:

```powershell
.\.venv\Scripts\pytest -v
```

To run individual test modules:

```powershell
# Test parsers for all 8 formats
.\.venv\Scripts\pytest tests/test_parsers.py -v

# Test hybrid search (BM25 + ChromaDB + RRF)
.\.venv\Scripts\pytest tests/test_hybrid_search.py -v

# Run the RAG evaluation benchmark
.\.venv\Scripts\pytest tests/test_evaluation.py -v -s
```

---

## 10. Known Limitations & OCR Disclosure

> [!IMPORTANT]
> **Scanned / Image-Only PDFs:**
> Standard text extraction tools (PyMuPDF) extract digital text streams. Scanned documents or image-only PDFs do not contain embedded text and require an Optical Character Recognition (OCR) engine (such as Tesseract). PaperMind AI detects scanned PDFs with insufficient text and reports a clear error rather than falsely claiming successful ingestion.

---

## 11. Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Script execution disabled` | Windows PowerShell execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError: No module named 'app'` | Running command from incorrect folder | Ensure you are in `c:\Users\ASUS\MYPROJECTS\New_RAG_System` |
| `Groq API key not configured` | Missing or default `.env` file | Ensure `.env` exists with a valid `GROQ_API_KEY` |
| `Torch / C++ runtime error` | Missing MSVC runtime on Windows | Install the official [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `ChromaDB SQLite error` | Corrupt temporary database | Stop Uvicorn, delete `data/storage/chroma`, and restart |
