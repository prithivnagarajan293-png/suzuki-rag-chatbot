import pickle
import re

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CHROMA_PATH = "data/chroma_v2"
BM25_PATH = "data/bm25_v1.pkl"

COLLECTION_NAME = "suzuki_knowledge"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

RRF_K = 60


# ============================================================
# LOAD MODELS / DATABASE
# ============================================================

print("Loading BGE model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading Chroma...")
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)

print(f"Chroma count: {collection.count()}")


print("Loading BM25...")

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
chunk_ids = bm25_data["chunk_ids"]

print(f"BM25 count: {len(chunk_ids)}")


# ============================================================
# CHUNK LOOKUP
# ============================================================

chunk_lookup = {}

BATCH_SIZE = 1000

for start in range(0, len(chunk_ids), BATCH_SIZE):

    batch_ids = chunk_ids[start:start + BATCH_SIZE]

    result = collection.get(
        ids=batch_ids,
        include=["documents", "metadatas"]
    )

    for chunk_id, document, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"]
    ):
        chunk_lookup[chunk_id] = {
            "document": document,
            "metadata": metadata,
        }

print(f"Lookup loaded: {len(chunk_lookup)}")


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# CHROMA SEARCH
# ============================================================

def chroma_search(query, top_k=10):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ]
    )

    results = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        ),
        start=1
    ):

        results.append({
            "id": chunk_id,
            "document": document,
            "metadata": metadata,
            "rank": rank,
            "distance": distance,
        })

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(query, top_k=10):

    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        chunk_id = chunk_ids[index]

        item = chunk_lookup[chunk_id]

        results.append({
            "id": chunk_id,
            "document": item["document"],
            "metadata": item["metadata"],
            "rank": rank,
            "score": float(scores[index]),
        })

    return results


# ============================================================
# RRF
# ============================================================

def rrf_score(rank):
    return 1.0 / (RRF_K + rank)


def fuse_rrf(
    chroma_results,
    bm25_results,
    final_k=10
):

    fused = {}

    # --------------------------------------------------------
    # Chroma contribution
    # --------------------------------------------------------

    for result in chroma_results:

        chunk_id = result["id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "chroma_rank": None,
                "bm25_rank": None,
            }

        fused[chunk_id]["rrf_score"] += rrf_score(
            result["rank"]
        )

        fused[chunk_id]["chroma_rank"] = result["rank"]

    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    for result in bm25_results:

        chunk_id = result["id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "chroma_rank": None,
                "bm25_rank": None,
            }

        fused[chunk_id]["rrf_score"] += rrf_score(
            result["rank"]
        )

        fused[chunk_id]["bm25_rank"] = result["rank"]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked[:final_k]