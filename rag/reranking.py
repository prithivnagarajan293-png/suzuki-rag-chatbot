from sentence_transformers import CrossEncoder


# ============================================================
# CONFIG
# ============================================================

RERANKER_MODEL = "BAAI/bge-reranker-base"


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded.")


# ============================================================
# RERANK
# ============================================================

def rerank_results(
    query,
    results,
    final_k=3
):

    pairs = [
        [
            query,
            result["document"]
        ]
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    reranked = []

    for result, score in zip(
        results,
        scores
    ):

        item = result.copy()

        item["reranker_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked[:final_k]