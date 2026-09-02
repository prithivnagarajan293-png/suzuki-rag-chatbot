import os

from google import genai

from rag.retrieval import (
    chroma_search,
    bm25_search,
    fuse_rrf,
)

from rag.reranking import (
    rerank_results,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

RRF_TOP_K = 10
FINAL_CONTEXT_K = 3


# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(
    api_key=api_key
)


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are a Suzuki automotive information assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not infer a value that is not explicitly present.
4. For numerical specifications, prices, dimensions, dates,
   and measurements, copy the value exactly from the context.
5. If the context contains conflicting values, explicitly say
   that the supplied sources contain conflicting information.
6. Answer only what the user asked.
7. Keep the answer concise.
8. If the context does not contain enough information, say:
   "I don't have enough information in the provided data."
"""


# ============================================================
# CONTEXT
# ============================================================

def build_context(results):

    parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        entity = metadata.get(
            "entity",
            metadata.get(
                "target_entity",
                "unknown"
            )
        )

        topic = metadata.get(
            "topic",
            "unknown"
        )

        parts.append(
            f"""
SOURCE {i}
Entity: {entity}
Topic: {topic}

{result["document"]}
"""
        )

    return "\n".join(parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query):

    # 1. Semantic retrieval
    chroma_results = chroma_search(
        query,
        RRF_TOP_K
    )

    # 2. Keyword retrieval
    bm25_results = bm25_search(
        query,
        RRF_TOP_K
    )

    # 3. RRF
    rrf_results = fuse_rrf(
        chroma_results,
        bm25_results,
        RRF_TOP_K
    )

    # 4. Reranking
    reranked_results = rerank_results(
        query,
        rrf_results,
        FINAL_CONTEXT_K
    )

    # 5. Context
    context = build_context(
        reranked_results
    )

    # 6. Prompt
    prompt = f"""
CONTEXT
=======
{context}

QUESTION
========
{query}
"""

    # 7. Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0,
        },
    )

    return response.text, reranked_results