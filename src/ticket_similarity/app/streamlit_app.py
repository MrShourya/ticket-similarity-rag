import streamlit as st

from ticket_similarity.cli.demo_inputs import DEMO_INPUTS
from ticket_similarity.retrieval.pipeline import (
    run_global_inference,
    run_final_similarity_search,
)


st.set_page_config(
    page_title="Ticket Similarity Triage",
    page_icon="🧠",
    layout="wide",
)


def format_pair_label(pair: dict) -> str:
    area = pair.get("area") or ""
    sub_area = pair.get("sub_area") or "[No Sub Area]"
    support = pair.get("support_score", 0.0)
    boost = pair.get("input_alignment_boost", 0.0)
    final_conf = pair.get("final_confidence", 0.0)

    return (
        f"{area} → {sub_area} | "
        f"support={support:.4f} | "
        f"boost={boost:.4f} | "
        f"final_confidence={final_conf:.4f}"
    )


def render_prediction_summary(inference_output: dict):
    predicted_area = inference_output["predicted_area"]
    predicted_sub_area = inference_output["predicted_sub_area"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Area")
        st.metric(
            label="Best Area",
            value=predicted_area.get("label") or "N/A",
            delta=f"confidence={predicted_area.get('confidence', 0.0):.4f}",
        )

        with st.expander("Area Candidates", expanded=False):
            for c in predicted_area.get("candidates", []):
                st.write(
                    f"- {c['label']} | score={c['score']:.4f} | confidence={c['confidence']:.4f}"
                )

    with col2:
        st.subheader("Predicted Sub Area")
        st.metric(
            label="Best Sub Area",
            value=predicted_sub_area.get("label") or "N/A",
            delta=f"confidence={predicted_sub_area.get('confidence', 0.0):.4f}",
        )

        with st.expander("Sub Area Candidates", expanded=False):
            for c in predicted_sub_area.get("candidates", []):
                st.write(
                    f"- {c['label']} | score={c['score']:.4f} | confidence={c['confidence']:.4f}"
                )


def results_to_dataframe(results: list[dict], include_rerank: bool = False):
    rows = []
    for idx, r in enumerate(results, start=1):
        row = {
            "rank": idx,
            "ticket_id": r.get("ticket_id"),
            "similarity_score": round(float(r.get("similarity_score", 0.0)), 4),
            "area": r.get("area"),
            "sub_area": r.get("sub_area"),
            "state": r.get("state"),
            "operation_name": r.get("operation_name"),
            "api": r.get("api"),
        }

        if include_rerank:
            row["rerank_score"] = round(float(r.get("rerank_score", 0.0)), 4)

        rows.append(row)

    return rows


def render_rank_movement(before_results: list[dict], after_results: list[dict]):
    st.subheader("Rank Movement")

    before_map = {r["ticket_id"]: idx + 1 for idx, r in enumerate(before_results)}
    after_map = {r["ticket_id"]: idx + 1 for idx, r in enumerate(after_results)}

    movement_rows = []
    for ticket_id, before_rank in before_map.items():
        if ticket_id in after_map:
            movement_rows.append(
                {
                    "ticket_id": ticket_id,
                    "before_rank": before_rank,
                    "after_rank": after_map[ticket_id],
                    "movement": before_rank - after_map[ticket_id],
                }
            )

    if movement_rows:
        st.dataframe(movement_rows, use_container_width=True)
    else:
        st.info("No overlapping ticket IDs found to compare.")


def render_final_details(results: list[dict]):
    st.subheader("Final Ticket Details")

    if not results:
        st.info("No final tickets found.")
        return

    for i, r in enumerate(results, start=1):
        title = (
            f"[{i}] {r.get('ticket_id')} | "
            f"similarity={float(r.get('similarity_score', 0.0)):.4f}"
        )
        if "rerank_score" in r:
            title += f" | rerank={float(r.get('rerank_score', 0.0)):.4f}"

        with st.expander(title, expanded=(i == 1)):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Area:** {r.get('area')}")
                st.write(f"**Sub Area:** {r.get('sub_area')}")
                st.write(f"**State:** {r.get('state')}")
            with c2:
                st.write(f"**Operation:** {r.get('operation_name')}")
                st.write(f"**API:** {r.get('api')}")

            st.markdown("**Base Text**")
            st.code(str(r.get("base_text", "")), language="text")

            enrichment_text = str(r.get("enrichment_text", "") or "").strip()
            if enrichment_text:
                st.markdown("**Enrichment Text**")
                st.code(enrichment_text[:1500], language="text")


def get_demo_defaults(demo_key: str):
    if demo_key in DEMO_INPUTS:
        return DEMO_INPUTS[demo_key]
    return {
        "name": "Manual",
        "short_description": "",
        "description": "",
        "input_area": None,
        "input_sub_area": None,
    }


def main():
    st.title("🧠 Auto Ops Local Demo")
    st.caption("Local ticket triage using embeddings, Qdrant, hierarchical classification, and reranking.")

    with st.sidebar:
        st.header("Input Mode")

        demo_options = {"0": "Manual Input"}
        for key, demo in DEMO_INPUTS.items():
            demo_options[key] = demo["name"]

        selected_demo_key = st.selectbox(
            "Choose demo scenario",
            options=list(demo_options.keys()),
            format_func=lambda k: f"{k} - {demo_options[k]}",
            index=1 if "1" in demo_options else 0,
        )

        demo_data = get_demo_defaults(selected_demo_key)

        st.divider()
        st.markdown("**About**")
        st.write(
            "This app runs locally. "
            "Ticket data stays on local machine. "
        )
        

    st.subheader("Ticket Input")

    col1, col2 = st.columns(2)

    with col1:
        input_area = st.text_input(
            "Existing Area (optional)",
            value=demo_data.get("input_area") or "",
        )
        short_description = st.text_input(
            "Short Description",
            value=demo_data.get("short_description", ""),
        )

    with col2:
        input_sub_area = st.text_input(
            "Existing Sub Area (optional)",
            value=demo_data.get("input_sub_area") or "",
        )

    description = st.text_area(
        "Description",
        value=demo_data.get("description", ""),
        height=180,
    )

    st.divider()

    inference_col1, inference_col2 = st.columns([1, 1])

    with inference_col1:
        candidate_k = st.slider("Number of tickets to fetch in similarity search", 5, 50, 20, 5)

    with inference_col2:
        top_pairs = st.slider("Number of Area / Sub Area Pairs", 1, 5, 3, 1)

    run_btn = st.button("Fetch Similar Tickets", type="primary", use_container_width=True)

    if run_btn:
        if not short_description.strip() or not description.strip():
            st.error("Short Description and Description are required.")
            return

        with st.spinner("Running global similarity search and classification..."):
            inference_output = run_global_inference(
                short_description=short_description.strip(),
                description=description.strip(),
                input_area=input_area.strip() or None,
                input_sub_area=input_sub_area.strip() or None,
                candidate_k=candidate_k,
                top_pairs=top_pairs,
            )

        st.session_state["inference_output"] = inference_output

    inference_output = st.session_state.get("inference_output")

    if inference_output:
        st.success("Global inference completed.")
        render_prediction_summary(inference_output)

        st.subheader("Top Candidate Area / Sub Area Pairs")

        candidate_pairs = inference_output.get("candidate_pairs", [])
        if not candidate_pairs:
            st.warning("No candidate pairs found.")
            return

        pair_options = list(range(len(candidate_pairs)))
        selected_pair_idx = st.radio(
            "Select the best Area / Sub Area pair",
            options=pair_options,
            format_func=lambda idx: format_pair_label(candidate_pairs[idx]),
        )

        selected_pair = candidate_pairs[selected_pair_idx]
        selected_area = selected_pair["area"]
        selected_sub_area = selected_pair["sub_area"] or None

        st.info(
            f"Selected Pair → Area: {selected_area} | Sub Area: {selected_sub_area or '[No Sub Area]'}"
        )

        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1:
            top_k = st.slider("No of matching tickets to be extracted", 1, 10, 5, 1)
        with c2:
            rerank_top_n = st.slider("No of tickets to be used as Rerank Candidate Pool", 5, 30, 15, 1)
        with c3:
            use_reranker = st.checkbox("Use Cross-Encoder Reranker", value=True)

        final_btn = st.button("Run Final Similarity Search", use_container_width=True)

        if final_btn:
            with st.spinner("Running filtered similarity search..."):
                comparison = run_final_similarity_search(
                    query=inference_output["query"],
                    selected_area=selected_area,
                    selected_sub_area=selected_sub_area,
                    top_k=top_k,
                    rerank_top_n=rerank_top_n,
                    use_reranker=use_reranker,
                    return_comparison=True,
                )

            before_results = comparison["before_rerank"]
            after_results = comparison["after_rerank"]

            st.session_state["comparison"] = comparison

            st.subheader("Comparison")

            tab1, tab2, tab3 = st.tabs(
                [
                    "Before Rerank",
                    "After Rerank",
                    "Rank Movement",
                ]
            )

            with tab1:
                st.dataframe(
                    results_to_dataframe(before_results, include_rerank=False),
                    use_container_width=True,
                )

            with tab2:
                st.dataframe(
                    results_to_dataframe(
                        after_results,
                        include_rerank=use_reranker,
                    ),
                    use_container_width=True,
                )

            with tab3:
                render_rank_movement(before_results, after_results)

            render_final_details(after_results)


if __name__ == "__main__":
    main()