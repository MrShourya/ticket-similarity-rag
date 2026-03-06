# Ticket Similarity RAG (POC)

This project is a **local Proof of Concept (POC)** for identifying
**similar ServiceNow tickets** using a Retrieval‑Augmented Generation
(RAG) style architecture.

The goal is to help support teams quickly find **historically similar
incidents**, understand likely causes, and accelerate resolution.

The system ingests historical tickets from Excel, cleans and structures
the data, and prepares it for similarity search using **vector
embeddings and a vector database (Qdrant)**.

------------------------------------------------------------------------

# Project Status

### Completed

Phase 1 --- Data ingestion from Excel\
Phase 2 --- Data cleaning and preprocessing

Features implemented:

-   Excel ingestion
-   Column normalization
-   Text normalization
-   Safe normalization (normalization‑lite)
-   High precision PII masking
-   API / operation endpoint sanitization
-   Structured ticket document generation
-   Export to clean dataset

Outputs:

    extract/
      cleaned_tickets.csv
      data_quality_report.json

------------------------------------------------------------------------

# Planned Next Phases

Phase 3 --- Ticket document builder\
Phase 4 --- Multilingual embeddings generation\
Phase 5 --- Vector database indexing (Qdrant)\
Phase 6 --- Similarity search pipeline\
Phase 7 --- Cross‑encoder reranker\
Phase 8 --- Root cause clustering and explanations

------------------------------------------------------------------------

# Tech Stack

  Component                 Technology
  ------------------------- -------------
  Language                  Python 3.11
  Package manager           Poetry
  Data processing           Pandas
  Excel reading             OpenPyXL
  Vector DB (next phase)    Qdrant
  Embeddings (next phase)   BGE‑M3

Everything currently runs **fully locally**.

------------------------------------------------------------------------

# Project Structure

    ticket-similarity-rag/

    data/
      tickets.xlsx

    extract/
      cleaned_tickets.csv
      data_quality_report.json

    src/
      ticket_similarity/

        config.py
        ingest.py
        normalize.py
        normalize_lite.py
        pii.py
        run_ingestion.py

    docker-compose.yml

    pyproject.toml
    poetry.lock
    README.md

------------------------------------------------------------------------

# Setup

## 1 Install Python

Python 3.11 is recommended.

    python3.11 --version

------------------------------------------------------------------------

## 2 Install Poetry

    pip install poetry

Verify:

    poetry --version

------------------------------------------------------------------------

## 3 Install dependencies

    poetry install

------------------------------------------------------------------------

# Input Data

Place your Excel extract here:

    data/tickets.xlsx

Expected columns (ServiceNow export):

Core fields

-   Number
-   Short Description
-   Description
-   Area
-   Sub Area

Optional enrichment fields (may be empty)

-   Request
-   Response
-   API Name
-   Operation Name
-   Timestamp

Excel column headers are automatically normalized so columns such as:

    "Operation Name  "

are internally converted to:

    Operation Name

------------------------------------------------------------------------

# Run Data Ingestion

Run:

    poetry run python -m ticket_similarity.run_ingestion

Outputs will be generated in:

    extract/

Files created:

    cleaned_tickets.csv
    data_quality_report.json

------------------------------------------------------------------------

# Output Dataset Structure

Each processed ticket contains:

    ticket_id
    area
    sub_area
    base_text
    enriched_text
    operation_name
    api

Example base_text:

    Area: ADDC
    SubArea: Water and Electricity Bill Payment

    Short Description:
    Payment Issue

    Description:
    My credit card ending 3699 was debited but bill not cleared

Example enriched_text:

    Area: ADDC
    SubArea: Water and Electricity Bill Payment

    Short Description:
    Payment Issue

    Description:
    ...

    Operation:
    /getStepInfo
    /gateway/.../getStepInfo

    API:
    /gateway/.../getStepInfo

------------------------------------------------------------------------

# Text Processing Pipeline

Current pipeline:

    normalize_text
          ↓
    normalize_lite
          ↓
    mask_pii

------------------------------------------------------------------------

# normalize_text

Basic cleaning.

Handles:

-   whitespace normalization
-   trimming
-   NA / NULL values

------------------------------------------------------------------------

# normalize_lite

Safe normalization without changing meaning.

Actions:

-   remove invisible unicode characters
-   normalize whitespace
-   normalize Arabic punctuation
-   sanitize URLs in text
-   preserve numeric values, dates, and references

------------------------------------------------------------------------

# PII Masking

Only **high confidence PII** is masked.

Masked:

  Type                   Replacement
  ---------------------- -----------------
  email                  `[EMAIL]`
  phone numbers          `[PHONE]`
  Emirates ID            `[EMIRATES_ID]`
  IP address             `[IP]`
  eID query parameters   `[ID]`

Not masked:

-   application IDs
-   case numbers
-   invoice numbers
-   amounts
-   dates

This ensures **semantic meaning is preserved** for similarity search.

------------------------------------------------------------------------

# Local Vector Database (Next Phase)

The next phase will introduce **Qdrant** as the vector database.

Start Qdrant using Docker:

    docker compose up -d

Verify:

    http://localhost:6333/dashboard
    http://localhost:6333/healthz

Stop Qdrant:

    docker compose down

------------------------------------------------------------------------

# Future Architecture

    Excel Tickets
          ↓
    Data Cleaning Pipeline
          ↓
    Ticket Document Builder
          ↓
    Embeddings (multilingual)
          ↓
    Vector Database (Qdrant)
          ↓
    Similarity Search
          ↓
    Cross Encoder Reranker
          ↓
    Similar Ticket Results

------------------------------------------------------------------------

# Goal of the System

For a **new incoming ticket**, the system will:

1.  Generate an embedding
2.  Search the vector database
3.  Retrieve similar historical tickets
4.  Rerank results
5.  Provide insights such as:

-   likely root cause
-   related API failures
-   similar incident patterns

This will help **L2 support teams triage incidents faster and reuse
historical knowledge**.

------------------------------------------------------------------------

# Notes

This POC is designed to run **fully locally**.

After validation, the architecture can be migrated to:

-   cloud vector databases
-   scalable embedding services
-   automated ticket pipelines
