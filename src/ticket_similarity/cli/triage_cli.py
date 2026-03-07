from ticket_similarity.cli.demo_inputs import DEMO_INPUTS
from ticket_similarity.retrieval.pipeline import (
    run_global_inference,
    run_final_similarity_search,
    print_prediction_block,
    print_similar_tickets,
    print_ranked_results,
    print_rank_changes,
)


def prompt_optional(label: str) -> str | None:
    value = input(label).strip()
    return value if value else None


def choose_demo_input():
    print("\nAvailable demo scenarios:")
    for key, demo in DEMO_INPUTS.items():
        print(f"{key}. {demo['name']}")
    print("0. Enter manually")

    choice = input("\nSelect a demo (0/1/2/3): ").strip()

    if choice in DEMO_INPUTS:
        return DEMO_INPUTS[choice]

    return None


def choose_pair(candidate_pairs: list[dict]) -> tuple[str, str | None]:
    while True:
        choice = input(
            "\nSelect a pair number (1/2/3), or type 'm' for manual entry: "
        ).strip().lower()

        if choice == "m":
            area = input("Enter Area: ").strip()
            sub_area = prompt_optional("Enter Sub Area (optional): ")
            if area:
                return area, sub_area
            print("Area is required for manual selection.")
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidate_pairs):
                selected = candidate_pairs[idx]
                return selected["area"], selected["sub_area"] or None

        print("Invalid selection. Please try again.")

def print_final_ticket_details(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("FINAL TICKET DETAILS (RERANKED RESULTS)")
    print("=" * 80)

    if not results:
        print("No final tickets found.")
        return

    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] Ticket ID: {r['ticket_id']}")
        print(f"    Similarity Score : {r['similarity_score']:.4f}")

        if "rerank_score" in r:
            print(f"    Rerank Score     : {r['rerank_score']:.4f}")

        print(f"    Area             : {r['area']}")
        print(f"    Sub Area         : {r['sub_area']}")
        print(f"    State            : {r['state']}")
        print(f"    Operation        : {r['operation_name']}")
        print(f"    API              : {r['api']}")

        print("\n    Base Text")
        print("    " + "-" * 68)
        for line in str(r.get("base_text", "")).splitlines():
            print(f"    {line}")

        enrichment_text = str(r.get("enrichment_text", "") or "").strip()
        if enrichment_text:
            print("\n    Enrichment Text")
            print("    " + "-" * 68)
            for line in enrichment_text[:1200].splitlines():
                print(f"    {line}")

def main():
    print("\n" + "=" * 80)
    print("TICKET SIMILARITY TRIAGE CLI")
    print("=" * 80)

    demo = choose_demo_input()

    if demo:
        short_description = demo["short_description"]
        description = demo["description"]
        input_area = demo["input_area"]
        input_sub_area = demo["input_sub_area"]

        print("\nUsing demo input:")
        print("-" * 80)
        print(f"Short Description : {short_description}")
        print(f"Description       : {description}")
        print(f"Input Area        : {input_area}")
        print(f"Input Sub Area    : {input_sub_area}")
    else:
        short_description = input("\nEnter Short Description: ").strip()
        description = input("Enter Description: ").strip()
        input_area = prompt_optional("Enter existing Area if available (optional): ")
        input_sub_area = prompt_optional("Enter existing Sub Area if available (optional): ")

    inference_output = run_global_inference(
        short_description=short_description,
        description=description,
        input_area=input_area,
        input_sub_area=input_sub_area,
        candidate_k=20,
        top_pairs=3,
    )

    print_prediction_block(inference_output)

    candidate_pairs = inference_output["candidate_pairs"]
    if not candidate_pairs:
        print("\nNo candidate pairs found from global similarity search.")
        return

    selected_area, selected_sub_area = choose_pair(candidate_pairs)

    print("\n" + "=" * 80)
    print("USER SELECTED PAIR")
    print("=" * 80)
    print(f"Area     : {selected_area}")
    print(f"Sub Area : {selected_sub_area}")

    comparison = run_final_similarity_search(
    query=inference_output["query"],
    selected_area=selected_area,
    selected_sub_area=selected_sub_area,
    top_k=5,
    rerank_top_n=15,
    use_reranker=True,
    return_comparison=True,
)

    before_results = comparison["before_rerank"]
    after_results = comparison["after_rerank"]

    print("\nRunning cross-encoder reranking on top candidates...")

    print_ranked_results(
        before_results,
        title="SIMILAR TICKETS (VECTOR SEARCH ONLY)"
    )

    print_ranked_results(
        after_results,
        title="SIMILAR TICKETS (AFTER RERANKING)"
    )

    print_rank_changes(
        before_results,
        after_results
    )

    print_final_ticket_details(after_results)


if __name__ == "__main__":
    main()