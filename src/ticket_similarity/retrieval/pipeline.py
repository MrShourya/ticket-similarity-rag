from ticket_similarity.retrieval.search_tickets import search_similar_tickets
from ticket_similarity.retrieval.area_inference import infer_area
from ticket_similarity.retrieval.subarea_inference import infer_sub_area
from ticket_similarity.retrieval.pair_inference import infer_top_area_subarea_pairs
from ticket_similarity.retrieval.confidence import apply_input_alignment_boost


def build_query(short_description: str, description: str) -> str:
    return f"""
Short Description:
{short_description}

Description:
{description}
""".strip()


def run_global_inference(
    short_description: str,
    description: str,
    input_area: str | None = None,
    input_sub_area: str | None = None,
    candidate_k: int = 20,
    top_pairs: int = 3,
) -> dict:
    query = build_query(short_description=short_description, description=description)

    initial_results = search_similar_tickets(
        query=query,
        area=None,
        sub_area=None,
        limit=candidate_k,
    )

    area_prediction = infer_area(initial_results)
    sub_area_prediction = infer_sub_area(initial_results, area_prediction["label"])

    candidate_pairs = infer_top_area_subarea_pairs(initial_results, top_n=top_pairs)
    candidate_pairs = apply_input_alignment_boost(
        candidate_pairs,
        input_area=input_area,
        input_sub_area=input_sub_area,
    )

    return {
        "query": query,
        "predicted_area": area_prediction,
        "predicted_sub_area": sub_area_prediction,
        "candidate_pairs": candidate_pairs,
        "initial_similar_tickets": initial_results,
    }


def run_final_similarity_search(
    query: str,
    selected_area: str,
    selected_sub_area: str | None = None,
    top_k: int = 5,
):
    return search_similar_tickets(
        query=query,
        area=selected_area,
        sub_area=selected_sub_area,
        limit=top_k,
    )


def print_prediction_block(output: dict):
    print("\n" + "=" * 80)
    print("GLOBAL INFERENCE SUMMARY")
    print("=" * 80)

    print("\nPredicted Area")
    print("-" * 80)
    print(
        f"Best Area       : {output['predicted_area']['label']}"
        f" (confidence={output['predicted_area']['confidence']:.4f})"
    )

    print("\nPredicted Sub Area")
    print("-" * 80)
    print(
        f"Best Sub Area   : {output['predicted_sub_area']['label']}"
        f" (confidence={output['predicted_sub_area']['confidence']:.4f})"
    )

    print("\nTop Candidate Area/Sub Area Pairs")
    print("-" * 80)
    for idx, pair in enumerate(output["candidate_pairs"], start=1):
        print(
            f"{idx}. Area='{pair['area']}' | "
            f"Sub Area='{pair['sub_area']}' | "
            f"support={pair['support_score']:.4f} | "
            f"boost={pair['input_alignment_boost']:.4f} | "
            f"final_confidence={pair['final_confidence']:.4f}"
        )


def print_similar_tickets(results: list[dict], title: str = "SIMILAR TICKETS"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No similar tickets found.")
        return

    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] Ticket ID: {r['ticket_id']}")
        print(f"    Similarity Score : {r['similarity_score']:.4f}")
        print(f"    Area             : {r['area']}")
        print(f"    Sub Area         : {r['sub_area']}")
        print(f"    State            : {r['state']}")
        print(f"    Operation        : {r['operation_name']}")
        print(f"    API              : {r['api']}")
        print(f"    Base Text        : {str(r['base_text'])[:300]}")


if __name__ == "__main__":
    short_description = "Payment Issue"
    description = "My credit card ending 1234 was debited on 20Feb2026 for 1,333.62. But bills not cleared in application."

    output = run_global_inference(
        short_description=short_description,
        description=description,
        input_area="ADDC",
        input_sub_area="Water and Electricity Bill Payment",
        candidate_k=20,
        top_pairs=3,
    )

    print_prediction_block(output)

    final_results = run_final_similarity_search(
        query=output["query"],
        selected_area=output["candidate_pairs"][0]["area"],
        selected_sub_area=output["candidate_pairs"][0]["sub_area"],
        top_k=5,
    )

    print_similar_tickets(final_results, title="FINAL FILTERED SIMILAR TICKETS")