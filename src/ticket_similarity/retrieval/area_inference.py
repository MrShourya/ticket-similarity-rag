from collections import defaultdict


def infer_area(results: list[dict]) -> dict:
    """
    Infer the most likely Area from retrieved tickets using weighted voting.

    Weight = similarity score
    """
    area_scores = defaultdict(float)

    for r in results:
        area = (r.get("area") or "").strip()
        score = float(r.get("score") or 0.0)

        if not area:
            continue

        area_scores[area] += score

    if not area_scores:
        return {
            "label": None,
            "confidence": 0.0,
            "candidates": [],
        }

    total_score = sum(area_scores.values())

    candidates = [
        {
            "label": area,
            "score": score,
            "confidence": round(score / total_score, 4) if total_score else 0.0,
        }
        for area, score in area_scores.items()
    ]

    candidates.sort(key=lambda x: x["score"], reverse=True)

    best = candidates[0]

    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "candidates": candidates,
    }