# Ticket Similarity RAG (POC)

A local Python-based POC to find **similar ServiceNow tickets** using:

- **multilingual embeddings**
- **Qdrant** as the vector database
- structured ticket preprocessing
- metadata filtering by fields like **Area** and **Sub Area**

The goal is to help support teams quickly find historically similar incidents and identify recurring patterns such as common APIs or operations.

---

## Current Status

Implemented so far:

- Excel ingestion from ServiceNow extract
- column header normalization
- safe text normalization
- high-precision PII masking
- API / Operation endpoint sanitization
- generation of:
  - `base_text`
  - `enrichment_text`
  - `embedding_text`
- export to cleaned CSV
- local Qdrant setup with Docker
- embeddings generation
- vector indexing into Qdrant
- similarity search with optional metadata filtering

---

## Project Structure

```text
ticket-similarity-rag/
├── data/
│   └── tickets.xlsx
├── extract/
│   ├── cleaned_tickets.csv
│   └── data_quality_report.json
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── normalize.py
│   │   ├── normalize_lite.py
│   │   ├── pii.py
│   │   └── run_ingestion.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embeddings.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── qdrant_store.py
│   │   └── index_tickets.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── search_tickets.py
│   ├── api/
│   │   └── __init__.py
│   └── evaluation/
│       └── __init__.py
├── tests/
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
├── .gitignore
└── README.md
```

---

## Architecture Overview

```text
Excel Tickets
    ↓
Ingestion + Cleaning
    ↓
PII Masking + Endpoint Sanitization
    ↓
base_text / enrichment_text / embedding_text
    ↓
Embedding Model
    ↓
Qdrant Vector DB
    ↓
Similarity Search
    ↓
Top Similar Tickets
```

### Text fields used

#### `base_text`
Core ticket meaning only:

- Area
- SubArea
- Short Description
- Description

#### `enrichment_text`
Extra L2 / technical context only:

- Operation
- API
- Response
- Request

#### `embedding_text`
Compact retrieval text used for embeddings:

- full `base_text`
- `operation`
- `api`
- request snippet
- response snippet

This avoids duplicating the entire enrichment payload while keeping important similarity signals.

---

## Input Data

Place your Excel file here:

```text
data/tickets.xlsx
```

Expected columns:

### Required
- Number
- Short Description
- Description
- Area
- Sub Area

### Optional
- State
- Created
- Parent Case
- Request
- Response
- API Name
- Operation Name
- Timestamp

Column names are normalized automatically, so headers with extra spaces or quotes are handled safely.

---

## Local Installation

### 1. Install prerequisites

- Python **3.11**
- Poetry
- Docker Desktop

Check versions:

```bash
python3.11 --version
poetry --version
docker --version
```

---

### 2. Clone the project

```bash
git clone https://github.com/MrShourya/ticket-similarity-rag.git
cd ticket-similarity-rag
```

---

### 3. Install Python dependencies

```bash
poetry install
```

If you are using the `src/` layout with application-style modules, run commands with:

```bash
PYTHONPATH=src
```

---

### 4. Start Qdrant locally

```bash
docker compose up -d
```

Verify:

```bash
curl http://localhost:6333/healthz
```

Expected response:

```json
{"status":"ok"}
```

Optional dashboard:

```text
http://localhost:6333/dashboard
```

---

## Running the Project Locally

### Step 1 — Put your Excel file in `data/`

```text
data/tickets.xlsx
```

---

### Step 2 — Run ingestion

This reads the Excel file, cleans the text, masks PII, sanitizes API fields, and generates the cleaned dataset.

```bash
PYTHONPATH=src poetry run python -m ingestion.run_ingestion
```

Outputs:

```text
extract/cleaned_tickets.csv
extract/data_quality_report.json
```

---

### Step 3 — Index tickets into Qdrant

This generates embeddings from `embedding_text` and stores vectors + metadata payload into Qdrant.

```bash
PYTHONPATH=src poetry run python -m vectorstore.index_tickets
```

---

### Step 4 — Run similarity search

```bash
PYTHONPATH=src poetry run python -m retrieval.search_tickets
```

This runs a sample query and returns top matching tickets.

---

## What Gets Stored in Qdrant

Each indexed ticket is stored as:

- **vector** → embedding of `embedding_text`
- **payload** → metadata fields

### Stored payload example

```json
{
  "ticket_id": "CD0374901",
  "state": "Resolved",
  "area": "ADDC",
  "sub_area": "Water and Electricity Bill Payment",
  "operation_name": "/paymentDoPayment",
  "api": "/gateway/SMARTHUB_API_SERVICES/1.0/requestSitePlan/paymentDoPayment",
  "base_text": "...",
  "enrichment_text": "...",
  "embedding_text": "..."
}
```

---

## Search Behavior

Current search flow:

1. Convert user query into an embedding
2. Search nearest vectors in Qdrant
3. Apply optional metadata filters:
   - `area`
   - `sub_area`
4. Return top matching tickets

---

## Preprocessing Rules

### Safe text normalization
- trims whitespace
- normalizes newlines
- removes invisible unicode characters
- lightly normalizes punctuation

### High-precision PII masking
Only masks strong-confidence PII such as:

- email addresses
- phone numbers
- Emirates ID
- IP addresses
- `eID=` query parameter values

Intentionally **not masked**:

- application IDs
- case numbers
- invoice/reference numbers
- amounts
- dates

This preserves retrieval quality.

### Endpoint sanitization
For API and Operation fields:

- removes query strings
- keeps useful endpoint paths
- preserves multi-line API/operation content

---

## Example Commands

### Ingestion
```bash
PYTHONPATH=src poetry run python -m ingestion.run_ingestion
```

### Indexing
```bash
PYTHONPATH=src poetry run python -m vectorstore.index_tickets
```

### Search
```bash
PYTHONPATH=src poetry run python -m retrieval.search_tickets
```

---

## Development Notes

### Current embedding model
Current local model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This was chosen because it:

- supports Arabic + English
- runs locally
- is lighter than larger multilingual models like `BAAI/bge-m3`

### Why not use full raw payloads directly?
Large request/response payloads can:

- waste memory
- reduce embedding quality
- slow down indexing

So the pipeline keeps compact snippets for embedding, while preserving fuller text in payload fields.

---

## Git Notes

Recommended `.gitignore` should exclude:

- `data/`
- `extract/`
- `.venv/`
- local model caches
- local Qdrant storage

Do **not** push customer data or processed ticket extracts to GitHub.

---

## Next Planned Enhancements

- retrieval pipeline module
- reranker using a cross-encoder
- enrichment-aware clustering / bubbling
- FastAPI endpoint
- evaluation dataset and metrics
- optional LangChain integration
- optional local translation support for analyst workflows

---

## Troubleshooting

### `ModuleNotFoundError`
Use:

```bash
PYTHONPATH=src poetry run python -m ingestion.run_ingestion
```

instead of running Python files directly by path.

### Qdrant not reachable
Make sure Docker Desktop is running and Qdrant is up:

```bash
docker compose up -d
curl http://localhost:6333/healthz
```

### Search or indexing model download warning
A Hugging Face download warning only means the model is being downloaded without authentication. It does **not** upload your ticket data anywhere. Embeddings are generated locally after the model is downloaded.

---

## License / Usage

This project is currently a personal POC intended for local experimentation and client-facing demonstrations.
