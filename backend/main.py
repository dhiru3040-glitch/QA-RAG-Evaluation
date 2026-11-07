import sys
import threading
from pathlib import Path

import chromadb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag.retrieval import query_documents, refresh_retrieval_runtime
from rag.vector_store import store_documents


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


app = FastAPI(title="VWO QA RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ingest_lock = threading.Lock()
ingest_state = {
    "last_ingestion": None,
    "prd_chunks": None,
    "test_case_chunks": None,
    "total_chunks": None,
}


def collection_status():
    db_dir = BASE_DIR / "vector_db"
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        collection = client.get_collection("qa_knowledge_base")
    except Exception:
        return {
            "collection_name": "qa_knowledge_base",
            "collection_exists": False,
            "chunk_count": 0,
            "prd_chunks": 0,
            "test_case_chunks": 0,
            "sources": [],
        }

    records = collection.get(include=["metadatas"])
    metadatas = records.get("metadatas", [])
    prd_chunks = sum(
        1 for metadata in metadatas
        if metadata and metadata.get("source_type") == "PRD"
    )
    test_case_chunks = sum(
        1 for metadata in metadatas
        if metadata and metadata.get("test_case_id")
    )

    sources = []
    for source_dir, source_type in (
        (BASE_DIR / "knowledge_base" / "prds", "PRD"),
        (BASE_DIR / "knowledge_base" / "test_cases", "TEST_CASE"),
    ):
        for source in sorted(source_dir.glob("*")):
            if not source.is_file():
                continue
            item = {
                "name": source.name,
                "source_type": source_type,
                "size_bytes": source.stat().st_size,
                "document_count": 1,
            }
            if source.suffix.lower() == ".csv":
                try:
                    item["document_count"] = len(pd.read_csv(source))
                except Exception:
                    item["document_count"] = 0
            sources.append(item)

    return {
        "collection_name": "qa_knowledge_base",
        "collection_exists": True,
        "chunk_count": collection.count(),
        "prd_chunks": prd_chunks,
        "test_case_chunks": test_case_chunks,
        "sources": sources,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "vwo-qa-rag"}


@app.get("/knowledge-base/status")
def knowledge_base_status():
    return {
        **collection_status(),
        "last_ingestion": ingest_state["last_ingestion"],
    }


@app.post("/ingest")
def ingest():
    if not ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Knowledge-base ingestion is already running.")

    try:
        result = store_documents()
        refresh_retrieval_runtime()
        from datetime import datetime, timezone

        ingest_state.update({
            "last_ingestion": datetime.now(timezone.utc).isoformat(),
            "prd_chunks": result["prd_chunks"],
            "test_case_chunks": result["test_case_chunks"],
            "total_chunks": result["total_chunks"],
        })
        return {
            "status": "completed",
            **result,
            "last_ingestion": ingest_state["last_ingestion"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        ingest_lock.release()


@app.post("/query")
def query(request: QueryRequest):
    try:
        return query_documents(request.question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
