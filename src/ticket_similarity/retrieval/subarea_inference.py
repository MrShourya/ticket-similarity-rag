from collections import defaultdict


def infer_sub_area(results: list[dict], predicted_area: str | None) -> dict:
    """
    Infer the most likely Sub Area, but only within the predicted Area.
    """
    if not predicted_area:
        return {
            "label": None,
            "confidence": 0.0,
            "candidates": [],
        }

    sub_area_scores = defaultdict(float)

    for r in results:
        area = (r.get("area") or "").strip()
        sub_area = (r.get("sub_area") or "").strip()
        score = float(r.get("score") or 0.0)

        if area != predicted_area:
            continue

        if not sub_area:
            continue

        sub_area_scores[sub_area] += score

    if not sub_area_scores:
        return {
            "label": None,
            "confidence": 0.0,
            "candidates": [],
        }

    total_score = sum(sub_area_scores.values())

    candidates = [
        {
            "label": sub_area,
            "score": score,
            "confidence": round(score / total_score, 4) if total_score else 0.0,
        }
        for sub_area, score in sub_area_scores.items()
    ]

    candidates.sort(key=lambda x: x["score"], reverse=True)

    best = candidates[0]

    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "candidates": candidates,
    }