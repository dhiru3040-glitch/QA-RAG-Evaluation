
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_prd, chunk_test_cases

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "vector_db"

# Load Nomic embedding model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")


def store_documents():

    # Load chunks
    prd_chunks = chunk_prd()
    tc_chunks = chunk_test_cases()

    all_chunks = prd_chunks + tc_chunks

    print("Total chunks:", len(all_chunks))

    # Prepare document text and metadata
    documents = [chunk["content"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    # Generate unique IDs
    ids = []

    for index, chunk in enumerate(all_chunks):
        source_type = chunk["metadata"]["source_type"].lower()
        ids.append(f"{source_type}_{index + 1}")

    # Add Nomic document prefix
    embedding_inputs = [
        "search_document: " + document
        for document in documents
    ]

    # Generate embeddings
    embeddings = model.encode(
        embedding_inputs,
        convert_to_numpy=True,
        show_progress_bar=True
    ).tolist()

    print("Embeddings generated:", len(embeddings))

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild collection on each ingestion run
    collection_name = "qa_knowledge_base"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Store everything in ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\nStorage completed successfully!")
    print("Stored documents:", collection.count())
    return {
        "prd_chunks": len(prd_chunks),
        "test_case_chunks": len(tc_chunks),
        "total_chunks": len(all_chunks),
        "collection_count": collection.count(),
        "collection_name": collection_name
    }


if __name__ == "__main__":
    store_documents()
