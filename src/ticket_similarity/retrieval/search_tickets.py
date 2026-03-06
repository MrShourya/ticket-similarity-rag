from qdrant_client.models import Filter, FieldCondition, MatchValue

from ticket_similarity.embeddings.embeddings import EmbeddingService
from ticket_similarity.vectorstore.qdrant_store import COLLECTION_NAME, get_qdrant_client


def build_filter(area: str | None = None, sub_area: str | None = None):
    conditions = []

    if area:
        conditions.append(FieldCondition(key="area", match=MatchValue(value=area)))

    if sub_area:
        conditions.append(FieldCondition(key="sub_area", match=MatchValue(value=sub_area)))

    if not conditions:
        return None

    return Filter(must=conditions)


def search_similar_tickets(
    query: str,
    area: str | None = None,
    sub_area: str | None = None,
    limit: int = 5,
):
    embedder = EmbeddingService()
    client = get_qdrant_client()

    query_vector = embedder.embed_query(query)
    query_filter = build_filter(area=area, sub_area=sub_area)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    output = []
    for point in response.points:
        payload = point.payload or {}
        output.append(
            {
                "score": point.score,
                "ticket_id": payload.get("ticket_id"),
                "state": payload.get("state"),
                "area": payload.get("area"),
                "sub_area": payload.get("sub_area"),
                "operation_name": payload.get("operation_name"),
                "api": payload.get("api"),
                "base_text": payload.get("base_text"),
                "enrichment_text": payload.get("enrichment_text"),
                "has_enrichment": payload.get("has_enrichment"),
            }
        )

    return output


if __name__ == "__main__":
    query = "My credit card ending 3699 was debited on 20Feb2026 for 1,796.62. But bills not cleared in TAMM."
    results = search_similar_tickets(
        query=query,
        area="ADDC",
        sub_area=None,
        limit=5,
    )

    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {r['score']:.4f}")
        print(f"Ticket ID: {r['ticket_id']}")
        print(f"State: {r['state']}")
        print(f"Area: {r['area']}")
        print(f"Sub Area: {r['sub_area']}")
        print(f"Has Enrichment: {r['has_enrichment']}")
        print(f"Operation: {r['operation_name']}")
        print(f"API: {r['api']}")
        print(f"Base Text:\n{str(r['base_text'])[:500]}")