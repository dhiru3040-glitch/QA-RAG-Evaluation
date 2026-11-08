
import json
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

from rag.retrieval import query_documents


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "evaluation" / "datasets" / "golden_dataset.json"
REPORT_DIR = BASE_DIR / "evaluation" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Load Golden Dataset
with open(DATASET_PATH, "r", encoding="utf-8") as file:
    golden_dataset = json.load(file)

class InstrumentedOllamaModel(OllamaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_calls = []

    def reset_usage(self):
        self.usage_calls = []

    def generate(self, prompt, schema=None, audio=None):
        chat_model = self.load_model()
        messages = self._prompt_messages(prompt, audio)
        response = chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                "temperature": self.temperature,
                **self.generation_kwargs,
            },
        )
        self._record_usage(response)
        content = (
            schema.model_validate_json(response.message.content)
            if schema
            else response.message.content
        )
        return content, 0

    async def a_generate(self, prompt, schema=None, audio=None):
        chat_model = self.load_model(async_mode=True)
        messages = self._prompt_messages(prompt, audio)
        response = await chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                "temperature": self.temperature,
                **self.generation_kwargs,
            },
        )
        self._record_usage(response)
        content = (
            schema.model_validate_json(response.message.content)
            if schema
            else response.message.content
        )
        return content, 0

    def _record_usage(self, response):
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        output_tokens = getattr(response, "eval_count", None)
        if prompt_tokens is None and isinstance(response, dict):
            prompt_tokens = response.get("prompt_eval_count")
        if output_tokens is None and isinstance(response, dict):
            output_tokens = response.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None and output_tokens is not None:
            total_tokens = prompt_tokens + output_tokens
        self.usage_calls.append({
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "source": "ollama_response",
        })

    def get_usage_summary(self):
        calls_with_numbers = [
            call for call in self.usage_calls
            if call["input_tokens"] is not None or call["output_tokens"] is not None
        ]
        input_tokens = sum(
            call["input_tokens"] for call in calls_with_numbers
            if call["input_tokens"] is not None
        ) if calls_with_numbers else None
        output_tokens = sum(
            call["output_tokens"] for call in calls_with_numbers
            if call["output_tokens"] is not None
        ) if calls_with_numbers else None
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "source": "ollama_response" if calls_with_numbers else None,
            "calls": self.usage_calls,
        }


#judge model
judge_model = InstrumentedOllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0
)
# Initialize DeepEval metrics
metrics = [
    FaithfulnessMetric(threshold=0.7, model=judge_model),
    AnswerRelevancyMetric(threshold=0.7, model=judge_model),
    ContextualPrecisionMetric(threshold=0.7, model=judge_model),
    ContextualRecallMetric(threshold=0.7, model=judge_model)
]


def serialize_metric_detail(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [serialize_metric_detail(item) for item in value]
    if isinstance(value, dict):
        return {
            key: serialize_metric_detail(item)
            for key, item in value.items()
        }
    if hasattr(value, "model_dump"):
        return serialize_metric_detail(value.model_dump())
    if hasattr(value, "dict"):
        return serialize_metric_detail(value.dict())
    if hasattr(value, "__dict__"):
        return serialize_metric_detail({
            key: item for key, item in vars(value).items()
            if not key.startswith("_")
        })
    return str(value)


def metric_details(metric):
    return {
        "truths": serialize_metric_detail(getattr(metric, "truths", None)),
        "claims": serialize_metric_detail(getattr(metric, "claims", None)),
        "statements": serialize_metric_detail(getattr(metric, "statements", None)),
        "verdicts": serialize_metric_detail(getattr(metric, "verdicts", None)),
        "verbose_logs": serialize_metric_detail(getattr(metric, "verbose_logs", None)),
        "evaluation_model": serialize_metric_detail(
            getattr(metric, "evaluation_model", None)
        ),
    }


def evaluate_test_case(item):
    """Evaluate one golden-dataset case using the existing RAG and DeepEval setup."""
    question = item["input"]
    rag_result = query_documents(question, include_usage=True)
    actual_answer = rag_result["answer"]
    retrieval_context = [
        document["content"]
        for document in rag_result["reranked_candidates"]
    ]

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=item["expected_output"],
        retrieval_context=retrieval_context,
        expected_retrieval_context=item["expected_retrieval_context"],
    )

    case_results = {
        "id": item["id"],
        "category": item["category"],
        "input": question,
        "expected_output": item["expected_output"],
        "expected_retrieval_context": item["expected_retrieval_context"],
        "actual_output": actual_answer,
        "actual_retrieval_context": retrieval_context,
        "actual_retrieved_chunks": [
            {
                "content": document["content"],
                "reranker_rank": document.get("reranker_rank"),
                "vector_rank": document.get("vector_rank"),
                "bm25_rank": document.get("bm25_rank"),
                "source": (
                    document.get("metadata", {}).get("source")
                    or document.get("metadata", {}).get("source_file")
                    or document.get("metadata", {}).get("file_name")
                    or document.get("metadata", {}).get("document_id")
                    or document.get("document_id")
                ),
                "metadata": document.get("metadata", {}),
            }
            for document in rag_result["reranked_candidates"]
        ],
        "token_usage": {
            "rag_generation": rag_result.get("generation_usage"),
            "deep_eval": None,
            "per_metric": {},
            "overall_test_case": None,
        },
        "metrics": {},
    }

    fresh_metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
        ContextualPrecisionMetric(threshold=0.7, model=judge_model),
        ContextualRecallMetric(threshold=0.7, model=judge_model),
    ]

    for metric in fresh_metrics:
        judge_model.reset_usage()
        metric.measure(test_case)
        metric_usage = judge_model.get_usage_summary()
        case_results["metrics"][metric.__class__.__name__] = {
            "score": metric.score,
            "threshold": metric.threshold,
            "passed": metric.is_successful(),
            "success": metric.is_successful(),
            "reason": metric.reason,
            "details": metric_details(metric),
            "token_usage": metric_usage,
        }
        case_results["token_usage"]["per_metric"][metric.__class__.__name__] = metric_usage

    judge_call_usage = []
    for metric_usage in case_results["token_usage"]["per_metric"].values():
        judge_call_usage.extend(metric_usage.get("calls") or [])

    calls_with_numbers = [
        call for call in judge_call_usage
        if call.get("input_tokens") is not None or call.get("output_tokens") is not None
    ]
    judge_input = sum(
        call["input_tokens"] for call in calls_with_numbers
        if call.get("input_tokens") is not None
    ) if calls_with_numbers else None
    judge_output = sum(
        call["output_tokens"] for call in calls_with_numbers
        if call.get("output_tokens") is not None
    ) if calls_with_numbers else None
    judge_total = (
        judge_input + judge_output
        if judge_input is not None and judge_output is not None
        else None
    )
    case_results["token_usage"]["deep_eval"] = {
        "input_tokens": judge_input,
        "output_tokens": judge_output,
        "total_tokens": judge_total,
        "source": "ollama_response" if calls_with_numbers else None,
        "calls": judge_call_usage,
    }

    rag_usage = case_results["token_usage"]["rag_generation"]
    overall_parts = [
        usage.get("total_tokens")
        for usage in (rag_usage, case_results["token_usage"]["deep_eval"])
        if usage and usage.get("total_tokens") is not None
    ]
    case_results["token_usage"]["overall_test_case"] = (
        sum(overall_parts) if overall_parts else None
    )

    metric_values = case_results["metrics"].values()
    case_results["overall_result"] = (
        "passed" if all(metric["passed"] for metric in metric_values)
        else "failed"
    )
    return case_results


def run_evaluation():

    evaluation_results = []

    print("\n========== RAG EVALUATION STARTED ==========")
    print("Total Golden Dataset cases:", len(golden_dataset))

    for index, item in enumerate(golden_dataset, start=1):

        question = item["input"]

        print(f"\n\n========== TEST CASE {index}/{len(golden_dataset)} ==========")
        print("Test ID:", item["id"])
        print("Category:", item["category"])
        print("Question:", question)

        try:
            case_results = evaluate_test_case(item)
            print("\nGenerated Answer:")
            print(case_results["actual_output"])
            print("\nEvaluating metrics...")
            for metric_name, metric_result in case_results["metrics"].items():
                print(
                    f"{metric_name}: "
                    f"{metric_result['score']:.2f} | "
                    f"{'PASS' if metric_result['passed'] else 'FAIL'}"
                )

            evaluation_results.append(case_results)

        except Exception as error:

            print("Evaluation failed for:", item["id"])
            print("Error:", str(error))

            evaluation_results.append({
                "id": item["id"],
                "category": item["category"],
                "input": question,
                "error": str(error)
            })

    # Save evaluation report
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
    passed_test_cases = sum(
        1 for result in evaluation_results
        if result.get("overall_result") == "passed"
    )
    failed_test_cases = sum(
        1 for result in evaluation_results
        if result.get("metrics") and result.get("overall_result") != "passed"
    )
    execution_errors = sum(
        1 for result in evaluation_results
        if result.get("error")
    )
    report = {
        "run_id": run_id,
        "evaluation_date": datetime.now().isoformat(),
        "total_test_cases": len(golden_dataset),
        "completed_test_cases": sum(
            1 for result in evaluation_results
            if "metrics" in result
        ),
        "passed_test_cases": passed_test_cases,
        "failed_test_cases": failed_test_cases,
        "execution_errors": execution_errors,
        "results": evaluation_results
    }

    report_path = REPORT_DIR / "evaluation_results.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    timestamped_path = REPORT_DIR / f"{run_id}.json"
    with open(timestamped_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n\n========== EVALUATION COMPLETED ==========")
    print("Total cases:", len(golden_dataset))
    print("Completed cases:", report["completed_test_cases"])
    print("Passed test cases:", passed_test_cases)
    print("Quality failures:", failed_test_cases)
    print("Execution errors:", execution_errors)
    print("Report saved at:", report_path)

    if execution_errors:
        print(
            f"\nEVALUATION EXECUTION FAILED: {execution_errors} case(s) failed with errors.",
            file=sys.stderr,
        )
        return 1
    if failed_test_cases:
        print(
            f"\nQUALITY THRESHOLD FAILURE: {failed_test_cases} completed case(s) failed one or more metric thresholds.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
