from collections import defaultdict


def infer_top_area_subarea_pairs(results: list[dict], top_n: int = 3) -> list[dict]:
    """
    Build top candidate (area, sub_area) pairs from retrieved tickets using weighted voting.
    Weight = similarity_score.
    """
    pair_scores = defaultdict(float)

    for r in results:
        area = (r.get("area") or "").strip()
        sub_area = (r.get("sub_area") or "").strip()
        score = float(r.get("similarity_score", r.get("score", 0.0)) or 0.0)

        if not area:
            continue

        key = (area, sub_area)
        pair_scores[key] += score

    if not pair_scores:
        return []

    total_score = sum(pair_scores.values())

    candidates = []
    for (area, sub_area), score in pair_scores.items():
        candidates.append(
            {
                "area": area,
                "sub_area": sub_area,
                "support_score": round(score, 6),
                "confidence": round(score / total_score, 4) if total_score else 0.0,
            }
        )

    candidates.sort(key=lambda x: x["support_score"], reverse=True)
    return candidates[:top_n]