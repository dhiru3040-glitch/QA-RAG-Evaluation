from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    """Load the cross-encoder once and reuse it across questions."""
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)

    return _reranker


def rerank_documents(question, candidates, top_k=4):
    """Rerank vector candidates using a question/content cross-encoder."""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = []
    for candidate in candidates:
        section_title = candidate.get("metadata", {}).get("section_title")
        reranker_text = candidate["content"]
        if section_title:
            reranker_text = f"{section_title}\n{reranker_text}"
        pairs.append([question, reranker_text])
    reranker_scores = reranker.predict(pairs).tolist()

    ranked = []
    for candidate, reranker_score in zip(candidates, reranker_scores):
        result = dict(candidate)
        result["reranker_score"] = float(reranker_score)
        ranked.append(result)

    reranker_values = [document["reranker_score"] for document in ranked]
    fusion_values = [
        document.get("fusion_score", 0.0)
        for document in ranked
    ]

    reranker_min = min(reranker_values)
    reranker_max = max(reranker_values)
    fusion_min = min(fusion_values)
    fusion_max = max(fusion_values)

    reranker_range = reranker_max - reranker_min
    fusion_range = fusion_max - fusion_min

    ranked.sort(
        key=lambda document: document["reranker_score"],
        reverse=True
    )

    for document in ranked:
        reranker_normalized = (
            (document["reranker_score"] - reranker_min) / reranker_range
            if reranker_range
            else 1.0
        )
        fusion_normalized = (
            (document.get("fusion_score", 0.0) - fusion_min) / fusion_range
            if fusion_range
            else 1.0
        )
        document["reranker_normalized_score"] = float(reranker_normalized)
        document["fusion_normalized_score"] = float(fusion_normalized)
        document["combined_score"] = (
            0.65 * reranker_normalized
            + 0.35 * fusion_normalized
        )

    ranked.sort(
        key=lambda document: document["combined_score"],
        reverse=True
    )

    selected = ranked[:top_k]
    for reranker_rank, document in enumerate(selected, start=1):
        document["reranker_rank"] = reranker_rank

    return selected
