import json
from pathlib import Path

import pandas as pd

from ticket_similarity.config.settings import ColumnMap, IngestConfig
from ticket_similarity.ingestion.normalize import normalize_text
from ticket_similarity.ingestion.normalize_lite import normalize_lite
from ticket_similarity.ingestion.pii import mask_pii, sanitize_endpoints_text


def _limit_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[TRUNCATED]"


def ingest_excel(config: IngestConfig, columns: ColumnMap):
    df = pd.read_excel(config.excel_path)

    # --- Normalize Excel headers so trailing spaces/quotes don't break mapping ---
    def _norm_col(c) -> str:
        c = str(c)
        c = c.strip()
        c = c.strip('"').strip("'")
        c = " ".join(c.split())  # collapse multiple spaces
        return c

    df.columns = [_norm_col(c) for c in df.columns]

    # --- Validate expected columns (required vs optional) ---
    required_cols = [
        columns.ticket_id,
        columns.short_description,
        columns.description,
        columns.area,
        columns.sub_area,
    ]
    optional_cols = [
        columns.request,
        columns.response,
        columns.api_name,
        columns.operation_name,
        columns.state,
        columns.created,
        columns.parent_case,
        columns.timestamp,
    ]

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in Excel after normalization: {missing_required}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    missing_optional = [c for c in optional_cols if c not in df.columns]
    if missing_optional:
        print(f"Warning: Optional columns missing after normalization: {missing_optional}")

    tickets = []

    for _, row in df.iterrows():
        ticket_id = normalize_text(row.get(columns.ticket_id))
        if not ticket_id:
            continue

        short_desc = normalize_text(row.get(columns.short_description))
        description = normalize_text(row.get(columns.description))
        area = normalize_text(row.get(columns.area))
        sub_area = normalize_text(row.get(columns.sub_area))

        state = normalize_text(row.get(columns.state)) if columns.state in df.columns else ""
        created = normalize_text(row.get(columns.created)) if columns.created in df.columns else ""
        parent_case = normalize_text(row.get(columns.parent_case)) if columns.parent_case in df.columns else ""
        failure_timestamp = normalize_text(row.get(columns.timestamp)) if columns.timestamp in df.columns else ""

        request = normalize_text(row.get(columns.request)) if columns.request in df.columns else ""
        response = normalize_text(row.get(columns.response)) if columns.response in df.columns else ""
        api = normalize_text(row.get(columns.api_name)) if columns.api_name in df.columns else ""
        operation = normalize_text(row.get(columns.operation_name)) if columns.operation_name in df.columns else ""

        # --- Normalization-lite only for natural language / payload text ---
        short_desc = normalize_lite(short_desc)
        description = normalize_lite(description)
        request = normalize_lite(request)
        response = normalize_lite(response)

        # --- Endpoint sanitization only for endpoint fields (multi-line safe) ---
        operation = sanitize_endpoints_text(operation)
        api = sanitize_endpoints_text(api)

        # --- PII masking (high precision) ---
        short_desc = mask_pii(short_desc)
        description = mask_pii(description)
        request = mask_pii(request)
        response = mask_pii(response)
        operation = mask_pii(operation)
        api = mask_pii(api)

        # --- Limit very large payload fields ---
        request = _limit_text(request, config.max_request_chars)
        response = _limit_text(response, config.max_response_chars)

        # --- Base text: core semantic ticket text only ---
        base_text = f"""
Area: {area}
SubArea: {sub_area}

Short Description:
{short_desc}

Description:
{description}
""".strip()

        # --- Enrichment text: extra L2/log-derived information only (NO base duplication) ---
        enrichment_parts = []

        if operation:
            enrichment_parts.append(f"Operation:\n{operation}")

        if api:
            enrichment_parts.append(f"API:\n{api}")

        if response:
            enrichment_parts.append(f"Response:\n{response}")

        if request:
            enrichment_parts.append(f"Request:\n{request}")

        enrichment_text = "\n\n".join(enrichment_parts).strip()

        # --- Embedding text: controlled text for vectorization ---
        embedding_parts = [base_text]

        if operation:
            embedding_parts.append(f"Operation:\n{operation}")

        if api:
            embedding_parts.append(f"API:\n{api}")

        if request:
            embedding_parts.append(
                f"Request Snippet:\n{_limit_text(request, 1000)}"
            )

        if response:
            embedding_parts.append(
                f"Response Snippet:\n{_limit_text(response, 1000)}"
            )

        embedding_text = "\n\n".join([p for p in embedding_parts if p]).strip()

        # hard safety cap
        embedding_text = _limit_text(embedding_text, 3000)

        ticket = {
            "ticket_id": ticket_id,
            "state": state,
            "created": created,
            "parent_case": parent_case,
            "failure_timestamp": failure_timestamp,
            "area": area,
            "sub_area": sub_area,
            "base_text": base_text,
            "enrichment_text": enrichment_text,
            "embedding_text": embedding_text,
            "operation_name": operation,
            "api": api,
            "request": request,
            "response": response,
            "has_enrichment": bool(operation or api or response or request),
        }

        tickets.append(ticket)

    return tickets


def export_results(tickets):
    Path("extract").mkdir(exist_ok=True)

    pd.DataFrame(tickets).to_csv(
        "extract/cleaned_tickets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "total_tickets": len(tickets),
        "tickets_with_enrichment": sum(1 for t in tickets if t.get("has_enrichment")),
    }

    with open("extract/data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saved cleaned_tickets.csv")
    print("Saved data_quality_report.json")