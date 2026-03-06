import pandas as pd
from qdrant_client.models import PointStruct

from embeddings.embeddings import EmbeddingService
from vectorstore.qdrant_store import (
    COLLECTION_NAME,
    get_qdrant_client,
    recreate_collection,
)


def build_embedding_text(row) -> str:
    """
    Prefer the precomputed embedding_text from ingestion.
    Fallback only if it is missing.
    """
    embedding_text = str(row.get("embedding_text", "") or "").strip()
    if embedding_text:
        return embedding_text

    base_text = str(row.get("base_text", "") or "").strip()
    operation = str(row.get("operation_name", "") or "").strip()
    api = str(row.get("api", "") or "").strip()
    request = str(row.get("request", "") or "").strip()
    response = str(row.get("response", "") or "").strip()

    parts = [base_text]

    if operation:
        parts.append(f"Operation:\n{operation}")

    if api:
        parts.append(f"API:\n{api}")

    if request:
        parts.append(f"Request Snippet:\n{request[:1000]}")

    if response:
        parts.append(f"Response Snippet:\n{response[:1000]}")

    text = "\n\n".join([p for p in parts if p]).strip()
    return text[:3000]


def main():
    df = pd.read_csv("extract/cleaned_tickets.csv")

    if df.empty:
        raise ValueError("cleaned_tickets.csv is empty")

    embedder = EmbeddingService()
    client = get_qdrant_client()

    vector_size = embedder.vector_size()
    recreate_collection(client, vector_size)

    embedding_texts = [build_embedding_text(row) for _, row in df.iterrows()]
    vectors = embedder.embed_texts(embedding_texts, batch_size=8)

    points = []
    for idx, row in df.iterrows():
        point = PointStruct(
            id=idx + 1,
            vector=vectors[idx],
            payload={
                "ticket_id": str(row.get("ticket_id", "")),
                "state": str(row.get("state", "")),
                "created": str(row.get("created", "")),
                "parent_case": str(row.get("parent_case", "")),
                "failure_timestamp": str(row.get("failure_timestamp", "")),
                "area": str(row.get("area", "")),
                "sub_area": str(row.get("sub_area", "")),
                "has_enrichment": bool(row.get("has_enrichment", False)),
                "operation_name": str(row.get("operation_name", "")),
                "api": str(row.get("api", "")),
                "base_text": str(row.get("base_text", "")),
                "enrichment_text": str(row.get("enrichment_text", "")),
                "embedding_text": str(row.get("embedding_text", "")),
                "request": str(row.get("request", "")),
                "response": str(row.get("response", "")),
            },
        )
        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Indexed {len(points)} tickets into Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()