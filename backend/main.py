import sys
import threading
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

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
from evaluation.evaluate_rag import evaluate_test_case, golden_dataset


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

EVALUATION_DIR = BASE_DIR / "evaluation"
DATASET_PATH = EVALUATION_DIR / "datasets" / "golden_dataset.json"
REPORT_DIR = EVALUATION_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_REPORT_PATH = REPORT_DIR / "evaluation_results.json"

evaluation_lock = threading.Lock()
evaluation_state = {
    "status": "Not evaluated",
    "mode": None,
    "test_case_id": None,
    "started_at": None,
    "completed_at": None,
    "current": 0,
    "total": 0,
    "error": None,
    "results": [],
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


METRIC_NAMES = {
    "FaithfulnessMetric": "faithfulness",
    "AnswerRelevancyMetric": "answer_relevancy",
    "ContextualPrecisionMetric": "contextual_precision",
    "ContextualRecallMetric": "contextual_recall",
}


def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def normalize_report(report):
    if not report:
        return None
    results = report.get("results", [])
    # Older reports may contain only the DeepEval summary fields. Do not invent
    # supporting details; preserve the report and mark evidence unavailable.
    for result in results:
        for metric_name, metric in (result.get("metrics") or {}).items():
            metric["success"] = metric.get(
                "success",
                metric.get("passed"),
            )
            metric["details"] = metric.get("details")
            if metric["details"] is None:
                metric["details"] = {
                    "truths": None,
                    "claims": None,
                    "statements": None,
                    "verdicts": None,
                    "verbose_logs": None,
                    "evaluation_model": None,
                    "availability": "not_provided_by_deepeval",
                }
    rag_usage = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "source": None,
    }
    judge_usage = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "source": None,
    }
    rag_input_values = []
    rag_output_values = []
    judge_input_values = []
    judge_output_values = []
    rag_source_available = False
    judge_source_available = False
    for result in results:
        usage = result.get("token_usage") or {}
        rag = usage.get("rag_generation") or {}
        judge = usage.get("deep_eval") or {}
        if rag.get("input_tokens") is not None:
            rag_input_values.append(rag["input_tokens"])
            rag_source_available = True
        if rag.get("output_tokens") is not None:
            rag_output_values.append(rag["output_tokens"])
            rag_source_available = True
        if judge.get("input_tokens") is not None:
            judge_input_values.append(judge["input_tokens"])
            judge_source_available = True
        if judge.get("output_tokens") is not None:
            judge_output_values.append(judge["output_tokens"])
            judge_source_available = True
    if rag_source_available:
        rag_usage["input_tokens"] = sum(rag_input_values) if rag_input_values else None
        rag_usage["output_tokens"] = sum(rag_output_values) if rag_output_values else None
        rag_usage["total_tokens"] = (
            rag_usage["input_tokens"] + rag_usage["output_tokens"]
            if rag_usage["input_tokens"] is not None and rag_usage["output_tokens"] is not None
            else None
        )
        rag_usage["source"] = "ollama_response"
    if judge_source_available:
        judge_usage["input_tokens"] = sum(judge_input_values) if judge_input_values else None
        judge_usage["output_tokens"] = sum(judge_output_values) if judge_output_values else None
        judge_usage["total_tokens"] = (
            judge_usage["input_tokens"] + judge_usage["output_tokens"]
            if judge_usage["input_tokens"] is not None and judge_usage["output_tokens"] is not None
            else None
        )
        judge_usage["source"] = "ollama_response"
    overall_parts = [
        usage["total_tokens"]
        for usage in (rag_usage, judge_usage)
        if usage["total_tokens"] is not None
    ]
    passed = sum(
        1 for result in results
        if result.get("overall_result") == "passed"
        or (
            result.get("metrics")
            and all(metric.get("passed") for metric in result["metrics"].values())
        )
    )
    average_metrics = {}
    for metric_name, public_name in METRIC_NAMES.items():
        scores = [
            result["metrics"][metric_name]["score"]
            for result in results
            if result.get("metrics") and metric_name in result["metrics"]
            and result["metrics"][metric_name].get("score") is not None
        ]
        threshold_values = [
            result["metrics"][metric_name]["threshold"]
            for result in results
            if result.get("metrics") and metric_name in result["metrics"]
            and result["metrics"][metric_name].get("threshold") is not None
        ]
        if scores:
            average_metrics[public_name] = {
                "score": sum(scores) / len(scores),
                "passed": sum(scores) / len(scores) >= (
                    threshold_values[0] if threshold_values else 0.7
                ),
                "threshold": threshold_values[0] if threshold_values else 0.7,
            }
        else:
            average_metrics[public_name] = None
    return {
        **report,
        "run_id": report.get("run_id") or report.get("evaluation_id"),
        "average_metrics": average_metrics,
        "passed_test_cases": passed,
        "failed_test_cases": len(results) - passed,
        "token_usage": {
            "rag_generation": rag_usage,
            "deep_eval": judge_usage,
            "overall_run": sum(overall_parts) if overall_parts else None,
            "scope": "completed test cases only",
        },
    }


def save_report(results, total, mode, test_case_id=None):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
    report = normalize_report({
        "run_id": run_id,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "test_case_id": test_case_id,
        "total_test_cases": total,
        "completed_test_cases": sum(1 for result in results if result.get("metrics")),
        "results": results,
    })

    timestamped_path = REPORT_DIR / f"{run_id}.json"
    with open(timestamped_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)
    return report, timestamped_path


def merge_latest_report(report):
    existing = load_json(LATEST_REPORT_PATH) or {
        "results": [],
        "total_test_cases": len(golden_dataset),
    }
    existing_results = {
        result["id"]: result
        for result in existing.get("results", [])
        if result.get("id")
    }
    for result in report["results"]:
        if result.get("id"):
            existing_results[result["id"]] = result

    merged = normalize_report({
        "evaluation_date": report["evaluation_date"],
        "run_id": report["run_id"],
        "mode": "mixed",
        "total_test_cases": len(golden_dataset),
        "completed_test_cases": len([
            result for result in existing_results.values() if result.get("metrics")
        ]),
        "results": [
            existing_results[item["id"]]
            for item in golden_dataset
            if item["id"] in existing_results
        ],
    })
    with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(merged, file, indent=4, ensure_ascii=False)
    return merged


def evaluation_worker(selected_items, mode, test_case_id=None):
    evaluation_state.update({
        "status": "Running",
        "mode": mode,
        "test_case_id": test_case_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "current": 0,
        "total": len(selected_items),
        "error": None,
        "results": [],
    })

    results = []
    try:
        for index, item in enumerate(selected_items, start=1):
            evaluation_state["current"] = index
            try:
                result = evaluate_test_case(item)
            except Exception as exc:
                result = {
                    "id": item["id"],
                    "category": item["category"],
                    "input": item["input"],
                    "expected_output": item["expected_output"],
                    "expected_retrieval_context": item["expected_retrieval_context"],
                    "error": str(exc),
                }
            results.append(result)
            evaluation_state["results"] = results

        report, _ = save_report(results, len(selected_items), mode, test_case_id)
        if mode == "individual":
            report = merge_latest_report(report)

        if mode == "all":
            with open(LATEST_REPORT_PATH, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=4, ensure_ascii=False)

        evaluation_state["status"] = (
            "Passed"
            if report["passed_test_cases"] == len(results)
            and report["failed_test_cases"] == 0
            else "Failed"
        )
    except Exception as exc:
        evaluation_state["status"] = "Failed"
        evaluation_state["error"] = str(exc)
    finally:
        evaluation_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        evaluation_lock.release()


def start_evaluation(selected_items, mode, test_case_id=None):
    if not evaluation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="An evaluation is already running.",
        )
    thread = threading.Thread(
        target=evaluation_worker,
        args=(selected_items, mode, test_case_id),
        daemon=True,
    )
    thread.start()
    return {
        "accepted": True,
        "status": "Running",
        "mode": mode,
        "test_case_id": test_case_id,
        "total": len(selected_items),
    }


@app.get("/evaluation/dataset")
def get_evaluation_dataset():
    categories = sorted({item["category"] for item in golden_dataset})
    return {"test_cases": golden_dataset, "categories": categories, "total": len(golden_dataset)}


@app.post("/evaluation/run-all")
def run_all_evaluations():
    return start_evaluation(golden_dataset, "all")


@app.post("/evaluation/run/{test_case_id}")
def run_individual_evaluation(test_case_id: str):
    item = next((case for case in golden_dataset if case["id"] == test_case_id.upper()), None)
    if not item:
        raise HTTPException(status_code=404, detail="Evaluation test case not found.")
    return start_evaluation([item], "individual", item["id"])


@app.get("/evaluation/status")
def get_evaluation_status():
    if evaluation_state["status"] == "Not evaluated":
        report = load_json(LATEST_REPORT_PATH)
        if report and report.get("results"):
            normalized = normalize_report(report)
            state = dict(evaluation_state)
            state["status"] = (
                "Passed"
                if normalized["failed_test_cases"] == 0
                else "Failed"
            )
            return state
    return evaluation_state


@app.get("/evaluation/results")
def get_evaluation_results():
    report = load_json(LATEST_REPORT_PATH)
    if not report:
        return {
            "evaluation_date": None,
            "run_id": None,
            "total_test_cases": len(golden_dataset),
            "completed_test_cases": 0,
            "passed_test_cases": 0,
            "failed_test_cases": 0,
            "average_metrics": {
                "faithfulness": None,
                "answer_relevancy": None,
                "contextual_precision": None,
                "contextual_recall": None,
            },
            "token_usage": {
                "rag_generation": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "source": None,
                },
                "deep_eval": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "source": None,
                },
                "overall_run": None,
                "scope": "completed test cases only",
            },
            "results": [],
        }
    return normalize_report(report)


@app.get("/evaluation/history")
def get_evaluation_history():
    reports = []
    for path in sorted(REPORT_DIR.glob("*.json"), reverse=True):
        if path.name == LATEST_REPORT_PATH.name:
            continue
        report = load_json(path)
        if report:
            reports.append(normalize_report(report))
    return {"history": reports}
