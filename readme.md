# 🚗 Suzuki Automotive RAG Chatbot

> A hybrid Retrieval-Augmented Generation (RAG) system for answering Suzuki automotive questions using semantic retrieval, keyword search, Reciprocal Rank Fusion, cross-encoder reranking, and grounded Gemini generation.

<p align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://suzuki-rag-chatbot-mbxyxhcxvxwj4qmnxtnm5q.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/Retrieval-BM25-blue)](https://github.com/dorianbrown/rank_bm25)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</p>

<p align="center">
  <a href="https://suzuki-rag-chatbot-mbxyxhcxvxwj4qmnxtnm5q.streamlit.app/">
    <strong>🚀 Try the Live Chatbot</strong>
  </a>
</p>
### 📸 Screenshot 1

<p align="center">
  <img src="screenshots/1.png" alt="Suzuki RAG Chatbot demo" width="850">
</p>

Screenshot 2

<p align="center">
  <img src="screenshots/1.png" alt="Suzuki RAG Chatbot demo" width="850">
</p>
## 📌 Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) application designed to answer Suzuki automotive questions using a domain-specific knowledge base.

Instead of relying solely on an LLM's pretrained knowledge, the system first retrieves relevant information using two complementary retrieval strategies:

- **Dense semantic retrieval** using BGE embeddings and ChromaDB
- **Sparse keyword retrieval** using BM25

The two retrieval rankings are combined using **Reciprocal Rank Fusion (RRF)**, followed by **cross-encoder reranking** to select the most relevant context for the final answer.

Gemini is then used as the generation layer, with explicit grounding instructions that require the model to answer from the retrieved context and surface conflicting source information when it exists.

The application is deployed publicly using **Streamlit Community Cloud**.

### What this project demonstrates

- End-to-end RAG architecture
- Hybrid semantic + keyword retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Grounded LLM generation
- Source-aware answers
- Persistent vector and keyword indexes
- Git LFS for large ML/retrieval artifacts
- Streamlit cloud deployment
## 🏗️ Architecture

The application follows a retrieval-first RAG architecture in which retrieval, ranking, and generation are separate stages.

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │     BGE Query Encoder    │
                     │  bge-base-en-v1.5 (768D) │
                     └─────────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │    ChromaDB      │          │       BM25       │
          │ Semantic Search  │          │ Keyword Search   │
          └────────┬─────────┘          └────────┬─────────┘
                   │                             │
                   └─────────────┬───────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Reciprocal Rank Fusion  │
                    │          (RRF)           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   BGE Cross-Encoder     │
                    │       Reranker           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Top Relevant Context   │
                    │      (Top 3 Chunks)      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Gemini Generation    │
                    │     Temperature = 0     │
                    │    Grounded Prompt      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Answer + Source Links │
                    └─────────────────────────┘

## 🔧 Key Engineering Decisions

### 1. Hybrid Retrieval: ChromaDB + BM25

A single retrieval method can miss relevant information.

The system therefore combines:

- **Dense retrieval** for semantic similarity
- **BM25** for exact keyword and terminology matching

This is particularly useful for automotive queries containing model names, measurements, engine sizes, prices, and technical terminology.

---

### 2. Reciprocal Rank Fusion (RRF)

The outputs from ChromaDB and BM25 are ranked independently.

Instead of directly comparing their raw scores, the system combines their rankings using **Reciprocal Rank Fusion**.

```text
Chroma ranking ──┐
                 ├──► RRF ──► Unified candidate ranking
BM25 ranking ────┘

## 🧪 Evaluation & Results

The retrieval pipeline was evaluated progressively to understand the contribution of each retrieval strategy.

### Retrieval Evaluation

A representative evaluation set was used to compare semantic retrieval, keyword retrieval, and hybrid retrieval.

| Retrieval Strategy | Recall@5 |
|---|---:|
| ChromaDB (Dense) | 80% |
| BM25 (Sparse) | 100% |
| RRF (Hybrid) | 100% |

The initial evaluation showed that BM25 recovered relevant keyword-heavy results that dense retrieval occasionally missed. Combining both retrieval signals with RRF improved the candidate set.

> **Note:** This was a small sanity-check evaluation rather than a statistically significant benchmark. The purpose was to validate the retrieval pipeline and compare the behavior of the different retrieval strategies.

---

### End-to-End Generation Evaluation

The complete pipeline was then tested with representative automotive questions covering:

- engine information
- vehicle pricing
- ground clearance
- platform relationships
- vehicle dimensions

The evaluation highlighted an important RAG behavior: **retrieval quality directly influences generation quality**.

When retrieved sources contained conflicting specifications, the final Gemini-based generation layer was instructed to surface the conflict rather than silently selecting an unsupported value.

For example, the Fronx ground-clearance query produced multiple retrieved values from different sources. The chatbot explicitly reported the conflicting values instead of presenting one as unquestionably correct.

Similarly, the Jimny dimensions query surfaced different dimension sets from different sources, and the answer identified the disagreement.

---

### What the Evaluation Demonstrated

```text
Dense Retrieval
      +
Sparse Retrieval
      ↓
     RRF
      ↓
Cross-Encoder Reranking
      ↓
Focused Context
      ↓
Grounded Generation

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python | Application and RAG pipeline |
| Embeddings | BAAI/bge-base-en-v1.5 | Semantic query/document representation |
| Vector Database | ChromaDB | Dense vector retrieval |
| Keyword Retrieval | BM25 | Lexical and exact-term retrieval |
| Fusion | Reciprocal Rank Fusion | Combines dense + sparse rankings |
| Reranking | BAAI/bge-reranker-base | Query-document relevance reranking |
| LLM | Gemini | Grounded answer generation |
| UI | Streamlit | Interactive chatbot interface |
| Version Control | Git + GitHub | Source control and collaboration |
| Large Artifacts | Git LFS | Storage of retrieval indexes |
| Deployment | Streamlit Community Cloud | Public application hosting |

---

## 📁 Project Structure

```text
suzuki-rag-chatbot/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── .gitattributes
│
├── rag/
│   ├── __init__.py
│   ├── retrieval.py
│   ├── reranking.py
│   └── generation.py
│
└── data/
    ├── bm25_v1.pkl
    └── chroma_v2/