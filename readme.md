
# Ticket Similarity RAG (Local POC)

A local Retrieval-Augmented Generation (RAG) proof-of-concept that helps triage ServiceNow tickets by:

- Finding similar historical tickets
- Predicting the most likely Area / SubArea
- Allowing user confirmation via CLI
- Retrieving final similar tickets filtered by the confirmed pair
- Running entirely locally with Qdrant + embeddings

---

# Architecture

User Ticket
↓
Text Preprocessing
↓
Embedding Generation
↓
Vector Search (Qdrant)
↓
Area / SubArea Inference
↓
Candidate Pair Ranking
↓
User Confirmation (CLI)
↓
Filtered Similar Ticket Retrieval

---

# Key Features

## Ticket Similarity Search

Finds the most similar historical tickets using embeddings.

Fields used:

- Short Description
- Description
- API
- Operation
- Request (snippet)
- Response (snippet)

Output includes:

similarity_score

---

## Area / SubArea Prediction

The system predicts the most likely Area/SubArea using weighted similarity voting.

Example:

Predicted Area: ADDC (confidence 0.74)

Predicted SubArea: Water and Electricity Bill Payment (confidence 0.69)

---

## Candidate Pair Ranking

Top 3 candidate pairs are produced:

1. ADDC → Water and Electricity Bill Payment
2. ADDC → New Connection
3. DPM → Issue a Site Plan

Each pair contains:

- similarity support score
- input alignment boost
- final confidence

---

## Confidence Boosting

If the user already provides Area/SubArea the system boosts confidence.

Boost rules:

Area match → +0.10

Area + SubArea match → +0.25

---

# CLI Guided Triage

Workflow:

1. Enter ticket details OR select demo case
2. System predicts candidate Area/SubArea pairs
3. User selects best pair
4. System retrieves similar tickets filtered by the selected pair

---

# Demo Scenarios

## Demo 1 — ADDC Payment Issue

Area: ADDC
SubArea: Water and Electricity Bill Payment

Short Description:
Payment Issue

Description:
My credit card ending 1234 was debited on 20Feb2026 for 1,333.62. But bills not cleared in application.

---

## Demo 2 — DPM Site Plan Issue

Area: DPM
SubArea: Issue a Site Plan

Short Description:
Site Plan Payment Redirection Issue

Description:
Payment summary page is shown for issuing a site plan, but the system does not redirect to the payment page.

---

## Demo 3 — Unknown Area Example

Short Description:
User cannot proceed with subsidy request

Description:
User is unable to proceed after selecting fodder details.

---

# Project Structure

ticket-similarity-rag
│
├── data
│   └── tickets.xlsx
│
├── extract
│   └── cleaned_tickets.csv
│
├── src
│   └── ticket_similarity
│       ├── cli
│       │   ├── triage_cli.py
│       │   └── demo_inputs.py
│       │
│       ├── config
│       │
│       ├── ingestion
│       │   ├── ingest.py
│       │   ├── normalize.py
│       │   ├── normalize_lite.py
│       │   └── pii.py
│       │
│       ├── embeddings
│       │   └── embeddings.py
│       │
│       ├── vectorstore
│       │   ├── qdrant_store.py
│       │   └── index_tickets.py
│       │
│       └── retrieval
│           ├── search_tickets.py
│           ├── pipeline.py
│           ├── area_inference.py
│           ├── subarea_inference.py
│           ├── pair_inference.py
│           └── confidence.py

---

# Installation

Requirements:

Python 3.11
Poetry
Docker

---

Clone repo:

git clone https://github.com/MrShourya/ticket-similarity-rag.git

cd ticket-similarity-rag

---

Install dependencies:

poetry install

---

Start Qdrant:

docker compose up -d

Dashboard:

http://localhost:6333/dashboard

---

# Data Preparation

Place your ticket extract in:

data/tickets.xlsx

---

# Ingestion

poetry run python -m ticket_similarity.ingestion.run_ingestion

Output:

extract/cleaned_tickets.csv

---

# Index Tickets

poetry run python -m ticket_similarity.vectorstore.index_tickets

---

# Run CLI Triage

poetry run python -m ticket_similarity.cli.triage_cli

---

# Similar Ticket Output Example

[1] Ticket ID: CD0374901
Similarity Score: 0.91
Area: ADDC
SubArea: Water and Electricity Bill Payment

---

# Security

The system includes:

PII masking
Endpoint sanitization
Local embeddings
No external ticket data transfer

---

# Future Enhancements

Cross‑encoder reranker

API clustering

Langfuse observability

FastAPI endpoint

Web / chatbot UI

Evaluation metrics

---

# License

Local POC for ticket triage experimentation.


---

# Detailed Retrieval Pipeline

The system does **not rely on a single vector search**. Instead it uses a multi‑stage retrieval pipeline.

1. **Global Retrieval**
   - Query built from:
     - Short Description
     - Description
   - Vector search runs across **all tickets**

2. **Area Inference**
   - Top results are analyzed
   - Areas are ranked using **weighted similarity scores**

3. **SubArea Inference**
   - SubAreas are predicted **only inside the predicted Area**
   - Prevents invalid Area/SubArea combinations

4. **Candidate Pair Ranking**
   - Top 3 `(Area, SubArea)` pairs generated
   - Each pair contains:
     - support score
     - alignment boost
     - final confidence

5. **User Confirmation**
   - CLI presents top pairs
   - User confirms or corrects the classification

6. **Filtered Retrieval**
   - Final vector search runs with filters:

```
area = selected_area
sub_area = selected_sub_area
```

This significantly improves relevance of similar tickets.

---

# Text Processing Pipeline

Before indexing tickets, the system performs preprocessing.

### Normalization

Removes noisy formatting:

- extra whitespace
- broken line breaks
- malformed characters

### PII Masking

Sensitive values are masked:

- phone numbers
- emails
- Emirates IDs
- IP addresses
- credit card numbers

Example:

```
credit card ending 1234
```

remains readable while protecting sensitive data.

### Endpoint Sanitization

API and operation endpoints are normalized.

Example:

```
/gateway/service/paymentDoPayment?args=XYZ
```

becomes

```
/gateway/service/paymentDoPayment
```

---

# Embedding Strategy

Each ticket is converted into an **embedding text representation**.

Embedding text contains:

- Area
- SubArea
- Short Description
- Description
- Operation
- API
- Request snippet
- Response snippet

Large payloads are truncated to maintain performance.

---

# Vector Database

The system uses **Qdrant** as the vector database.

Each stored record contains:

### Vector

Embedding of ticket content.

### Payload

Metadata fields:

```
ticket_id
area
sub_area
operation_name
api
base_text
request
response
```

Payload fields allow filtering during retrieval.

---

# Running the Retrieval Pipeline (Developer Mode)

You can run the retrieval pipeline directly for testing.

```
poetry run python -m ticket_similarity.retrieval.pipeline
```

This will:

1. run global similarity search
2. infer Area/SubArea
3. show candidate pairs
4. perform final filtered retrieval

Useful for debugging retrieval behaviour.

---

# CLI Demo Mode

Run the CLI:

```
poetry run python -m ticket_similarity.cli.triage_cli
```

The CLI supports **demo inputs**.

You can choose:

```
0 → manual input
1 → ADDC payment issue
2 → DPM site plan issue
3 → unknown area example
```

This allows easy demonstrations without typing ticket details.

---

# Example CLI Session

Example interaction:

```
Select demo (0/1/2/3)
```

Then:

```
Top predicted Area/SubArea pairs
```

User selects a pair:

```
1 → ADDC / Water and Electricity Bill Payment
```

System returns:

```
Final Similar Tickets
```

---

# Security & Data Handling

The system runs **fully locally**.

No ticket data is sent to external services.

Security measures:

- PII masking
- endpoint sanitization
- local embedding models
- local vector database

---
