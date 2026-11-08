
from pathlib import Path
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from rag.generator import generate_answer
from rag.reranker import rerank_documents
from rag.bm25_retriever import BM25Retriever


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "vector_db"


# Load embedding model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")


# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(
    name="qa_knowledge_base"
)


corpus = collection.get(include=["documents", "metadatas"])
bm25_retriever = BM25Retriever(
    documents=corpus["documents"],
    metadatas=corpus["metadatas"],
    ids=corpus["ids"]
)


def refresh_retrieval_runtime():
    """Reload ChromaDB and BM25 state after an explicit ingestion rebuild."""
    global client, collection, corpus, bm25_retriever

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name="qa_knowledge_base")
    corpus = collection.get(include=["documents", "metadatas"])
    bm25_retriever = BM25Retriever(
        documents=corpus["documents"],
        metadatas=corpus["metadatas"],
        ids=corpus["ids"]
    )
    return collection.count()


def merge_candidates(vector_candidates, bm25_candidates, limit=20):
    """Merge candidates with reciprocal rank fusion and deduplicate by document ID."""
    fusion_scores = {}
    candidates_by_id = {}
    rrf_k = 60.0

    for candidate in vector_candidates:
        document_id = candidate["document_id"]
        candidates_by_id.setdefault(document_id, dict(candidate))
        fusion_scores[document_id] = (
            fusion_scores.get(document_id, 0.0)
            + 1.0 / (rrf_k + candidate["vector_rank"])
        )

    for candidate in bm25_candidates:
        document_id = candidate["document_id"]
        existing = candidates_by_id.setdefault(document_id, dict(candidate))
        existing["bm25_rank"] = candidate["bm25_rank"]
        existing["bm25_score"] = candidate["bm25_score"]
        fusion_scores[document_id] = (
            fusion_scores.get(document_id, 0.0)
            + 1.0 / (rrf_k + candidate["bm25_rank"])
        )

    merged = []
    for document_id, candidate in candidates_by_id.items():
        result = dict(candidate)
        result["document_id"] = document_id
        result["fusion_score"] = fusion_scores[document_id]
        merged.append(result)

    merged.sort(
        key=lambda candidate: candidate["fusion_score"],
        reverse=True
    )
    return merged[:limit]


def retrieve_documents(question, candidate_k=20, merged_k=40, top_k=4):
    """Retrieve vector and BM25 candidates, merge them, and rerank them."""

    query_text = "search_query: " + question

    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k
    )

    retrieved_documents = []
    document_ids = results["ids"][0]

    for vector_rank, document in enumerate(
        results["documents"][0],
        start=1
    ):
        distance = results["distances"][0][vector_rank - 1]

        retrieved_documents.append({
            "content": document,
            "metadata": results["metadatas"][0][vector_rank - 1],
            "document_id": document_ids[vector_rank - 1],
            "vector_rank": vector_rank,
            "vector_distance": distance,
            "cosine_similarity": 1.0 - distance
        })

    bm25_candidates = bm25_retriever.retrieve(
        question,
        top_k=candidate_k
    )

    merged_candidates = merge_candidates(
        retrieved_documents,
        bm25_candidates,
        limit=merged_k
    )

    reranked_documents = rerank_documents(
        question,
        merged_candidates,
        top_k=top_k
    )

    return {
        "vector_candidates": retrieved_documents,
        "bm25_candidates": bm25_candidates,
        "merged_candidates": merged_candidates,
        "reranked_candidates": reranked_documents
    }


def query_documents(question, candidate_k=20, merged_k=40, top_k=4, include_usage=False):
    """Run retrieval and answer generation and return structured API/CLI data."""
    retrieval_results = retrieve_documents(
        question=question,
        candidate_k=candidate_k,
        merged_k=merged_k,
        top_k=top_k
    )
    reranked_documents = retrieval_results["reranked_candidates"]
    if include_usage:
        answer, generation_usage = generate_answer(
            question, reranked_documents, include_usage=True
        )
    else:
        answer = generate_answer(question, reranked_documents)

    result = {
        "question": question,
        "answer": answer,
        "vector_candidates": retrieval_results["vector_candidates"],
        "bm25_candidates": retrieval_results["bm25_candidates"],
        "merged_candidates": retrieval_results["merged_candidates"],
        "reranked_candidates": reranked_documents
    }
    if include_usage:
        result["generation_usage"] = generation_usage
    return result


def ask_question(question):
    """Retrieve context and generate an answer using the LLM."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    retrieval_results = retrieve_documents(
        question,
        candidate_k=20,
        merged_k=40,
        top_k=4
    )

    print("\n========== VECTOR RETRIEVAL CANDIDATES ==========")
    for candidate in retrieval_results["vector_candidates"]:
        print(
            f"\nVector Rank {candidate['vector_rank']} | "
            f"Document ID: {candidate['document_id']}"
        )
        print("Cosine Similarity:", candidate["cosine_similarity"])
        print("Vector Distance:", candidate["vector_distance"])
        print("Content:", candidate["content"])
        print("Metadata:", candidate["metadata"])

    print("\n========== BM25 RETRIEVAL CANDIDATES ==========")
    for candidate in retrieval_results["bm25_candidates"]:
        print(
            f"\nBM25 Rank {candidate['bm25_rank']} | "
            f"Document ID: {candidate['document_id']}"
        )
        print("BM25 Score:", candidate["bm25_score"])
        print("Content:", candidate["content"])
        print("Metadata:", candidate["metadata"])

    print("\n========== MERGED RETRIEVAL CANDIDATES ==========")
    for index, candidate in enumerate(
        retrieval_results["merged_candidates"],
        start=1
    ):
        print(
            f"\nMerged Rank {index} | "
            f"Document ID: {candidate['document_id']}"
        )
        print("Fusion Score:", candidate["fusion_score"])
        print("Vector Rank:", candidate.get("vector_rank"))
        print("BM25 Rank:", candidate.get("bm25_rank"))
        print("Content:", candidate["content"])
        print("Metadata:", candidate["metadata"])

    retrieved_documents = retrieval_results["reranked_candidates"]

    print("\n========== FINAL RERANKED CONTEXT ==========")
    for document in retrieved_documents:
        print(f"\nRESULT {document['reranker_rank']}")
        print("Vector Rank:", document.get("vector_rank"))
        print("BM25 Rank:", document.get("bm25_rank"))
        print("Reranker Rank:", document["reranker_rank"])
        print("Content:", document["content"])
        print("Metadata:", document["metadata"])
        print("Cosine Similarity:", document.get("cosine_similarity"))
        print("Vector Distance:", document.get("vector_distance"))
        print("Fusion Score:", document["fusion_score"])
        print("Reranker Score:", document["reranker_score"])
        print("Combined Score:", document["combined_score"])
        print(
            "Reranker Normalized Score:",
            document["reranker_normalized_score"]
        )
        print(
            "Fusion Normalized Score:",
            document["fusion_normalized_score"]
        )

    # Generate answer using retrieved context
    answer = generate_answer(
        question,
        retrieved_documents
    )

    # Display final answer
    print("\n========== GENERATED ANSWER ==========")
    print(answer)


if __name__ == "__main__":

    question = input("\nEnter your question: ")

    ask_question(question)
