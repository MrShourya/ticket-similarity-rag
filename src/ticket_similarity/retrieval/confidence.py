def apply_input_alignment_boost(
    candidate_pairs: list[dict],
    input_area: str | None = None,
    input_sub_area: str | None = None,
) -> list[dict]:
    """
    Boost confidence if input area/sub_area aligns with predicted pairs.

    Rules:
    - exact area + sub_area match -> strong boost
    - area match only -> moderate boost
    - no match -> no boost
    """
    input_area = (input_area or "").strip()
    input_sub_area = (input_sub_area or "").strip()

    boosted = []

    for pair in candidate_pairs:
        area = (pair.get("area") or "").strip()
        sub_area = (pair.get("sub_area") or "").strip()

        boost = 0.0

        if input_area and area == input_area:
            boost += 0.10

        if input_area and input_sub_area and area == input_area and sub_area == input_sub_area:
            boost += 0.15

        final_confidence = min(1.0, round(pair["confidence"] + boost, 4))

        updated = dict(pair)
        updated["input_alignment_boost"] = round(boost, 4)
        updated["final_confidence"] = final_confidence
        boosted.append(updated)

    boosted.sort(key=lambda x: x["final_confidence"], reverse=True)
    return boosted