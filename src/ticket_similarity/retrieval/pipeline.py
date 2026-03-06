from ticket_similarity.retrieval.search_tickets import search_similar_tickets
from ticket_similarity.retrieval.area_inference import infer_area
from ticket_similarity.retrieval.subarea_inference import infer_sub_area


def build_query(short_description: str, description: str) -> str:
    return f"""
Short Description:
{short_description}

Description:
{description}
""".strip()


def run_retrieval_pipeline(
    short_description: str,
    description: str,
    top_k: int = 10,
    candidate_k: int = 20,
    refine_with_area: bool = True,
) -> dict:
    """
    Pipeline:
    1. Global retrieval
    2. Infer Area
    3. Infer Sub Area within predicted Area
    4. Optional second retrieval filtered by predicted Area
    """

    query = build_query(short_description=short_description, description=description)

    # Pass 1: global retrieval
    initial_results = search_similar_tickets(
        query=query,
        area=None,
        sub_area=None,
        limit=candidate_k,
    )

    area_prediction = infer_area(initial_results)
    sub_area_prediction = infer_sub_area(initial_results, area_prediction["label"])

    final_results = initial_results[:top_k]

    # Pass 2: refine retrieval using predicted Area only
    if refine_with_area and area_prediction["label"]:
        refined_results = search_similar_tickets(
            query=query,
            area=area_prediction["label"],
            sub_area=None,
            limit=top_k,
        )
        final_results = refined_results

    return {
        "query": query,
        "predicted_area": area_prediction,
        "predicted_sub_area": sub_area_prediction,
        "similar_tickets": final_results,
        "initial_similar_tickets": initial_results[:top_k],
    }


if __name__ == "__main__":
    short_description = "Payment Issue"
    description = "My credit card was debited but bill is still not cleared in TAMM."

    output = run_retrieval_pipeline(
        short_description=short_description,
        description=description,
        top_k=5,
        candidate_k=20,
        refine_with_area=True,
    )

    print("\nPredicted Area:")
    print(output["predicted_area"])

    print("\nPredicted Sub Area:")
    print(output["predicted_sub_area"])

    print("\nFinal Similar Tickets:")
    for i, r in enumerate(output["similar_tickets"], start=1):
        print(f"\nResult {i}")
        print(f"Similarity Score: {r['score']:.4f}")
        print(f"Ticket ID: {r['ticket_id']}")
        print(f"Area: {r['area']}")
        print(f"Sub Area: {r['sub_area']}")
        print(f"Operation: {r['operation_name']}")
        print(f"API: {r['api']}")
        print(f"Base Text: {str(r['base_text'])[:300]}")