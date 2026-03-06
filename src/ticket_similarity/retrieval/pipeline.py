from ticket_similarity.retrieval.search_tickets import search_similar_tickets
from ticket_similarity.retrieval.area_inference import infer_area
from ticket_similarity.retrieval.subarea_inference import infer_sub_area


def run_retrieval_pipeline(
    short_description: str,
    description: str,
    top_k: int = 10,
    candidate_k: int = 20,
) -> dict:
    """
    Phase 1 pipeline:
    - build query from short_description + description
    - run global retrieval (no area/sub_area filter)
    - infer Area
    - infer Sub Area within Area
    - return top results + predicted labels
    """

    query = f"""
Short Description:
{short_description}

Description:
{description}
""".strip()

    # Global retrieval, no filters yet
    results = search_similar_tickets(
        query=query,
        area=None,
        sub_area=None,
        limit=candidate_k,
    )

    area_prediction = infer_area(results)
    sub_area_prediction = infer_sub_area(results, area_prediction["label"])

    return {
        "query": query,
        "predicted_area": area_prediction,
        "predicted_sub_area": sub_area_prediction,
        "similar_tickets": results[:top_k],
    }


if __name__ == "__main__":
    short_description = "Payment Issue"
    description = "My credit card ending 3699 was debited on 20Feb2026 for 1,796.62. But bills not cleared in TAMM."

    output = run_retrieval_pipeline(
        short_description=short_description,
        description=description,
        top_k=5,
        candidate_k=20,
    )

    print("\nPredicted Area:")
    print(output["predicted_area"])

    print("\nPredicted Sub Area:")
    print(output["predicted_sub_area"])

    print("\nTop Similar Tickets:")
    for i, r in enumerate(output["similar_tickets"], start=1):
        print(f"\nResult {i}")
        print(f"Score: {r['score']:.4f}")
        print(f"Ticket ID: {r['ticket_id']}")
        print(f"Area: {r['area']}")
        print(f"Sub Area: {r['sub_area']}")
        print(f"Operation: {r['operation_name']}")
        print(f"API: {r['api']}")
        print(f"Base Text: {str(r['base_text'])[:300]}")