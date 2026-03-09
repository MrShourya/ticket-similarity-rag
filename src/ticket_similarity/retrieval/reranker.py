from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from ticket_similarity.observability.langfuse_support import (
    observe,
    update_current_observation,
)

RERANKER_MODEL = "BAAI/bge-reranker-base"
from transformers import logging
logging.set_verbosity_error()

class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @observe(name="cross_encoder_rerank")
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Rerank candidate tickets using a cross-encoder.

        Returns candidates sorted by rerank_score descending.
        """
        if not candidates:
            update_current_observation(
                input={
                    "query": query[:500],
                    "candidate_count": 0,
                    "top_k": top_k,
                },
                output={"reranked_ticket_ids": []},
                metadata={"model": RERANKER_MODEL},
            )
            return []

        pair_texts = []
        for c in candidates:
            candidate_text = build_reranker_candidate_text(c)
            pair_texts.append((query, candidate_text))

        with torch.no_grad():
            inputs = self.tokenizer(
                pair_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            outputs = self.model(**inputs)
            scores = outputs.logits.view(-1).tolist()

        reranked = []
        for candidate, score in zip(candidates, scores):
            updated = dict(candidate)
            updated["rerank_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_results = reranked[:top_k]

        update_current_observation(
            input={
                "query": query[:500],
                "candidate_count": len(candidates),
                "top_k": top_k,
            },
            output={
                "reranked_ticket_ids": [r["ticket_id"] for r in final_results],
            },
            metadata={
                "model": RERANKER_MODEL,
            },
        )

        return final_results


def build_reranker_candidate_text(candidate: dict) -> str:
    """
    Build a field-prioritized text for reranking.

    Priority:
    1. Short Description / Description (via base_text)
    2. Response snippet
    3. Operation
    4. API
    5. Request snippet (small)
    """
    base_text = str(candidate.get("base_text", "") or "").strip()
    response = str(candidate.get("response", "") or "").strip()
    operation = str(candidate.get("operation_name", "") or "").strip()
    api = str(candidate.get("api", "") or "").strip()
    request = str(candidate.get("request", "") or "").strip()

    parts = [base_text]

    if response:
        parts.append(f"Response Snippet:\n{response[:800]}")

    if operation:
        parts.append(f"Operation:\n{operation}")

    if api:
        parts.append(f"API:\n{api}")

    if request:
        parts.append(f"Request Snippet:\n{request[:300]}")

    text = "\n\n".join([p for p in parts if p]).strip()
    return text[:2500]