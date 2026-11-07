
import sys

import ollama

MODEL_NAME = "llama3.2:3b"


def generate_answer(question, retrieved_documents):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    context_blocks = []

    for reranker_rank, document in enumerate(retrieved_documents, start=1):
        metadata = document["metadata"]
        context_blocks.append(
            f"[Reranker Rank {reranker_rank} | "
            f"Vector Rank {document.get('vector_rank')} | "
            f"BM25 Rank {document.get('bm25_rank')}]\n"
            f"{document['content']}\n"
            f"Metadata: {metadata}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a QA assistant.

Read the following retrieved documents and answer the question.

Documents:
{context}

Question:
{question}

Instructions:
- Find the answer directly in the document.
- If a URL is present, return the exact URL.
- Mention the Test Case ID when the source is a test case.
- Use all provided documents only where relevant.
- Preserve distinctions between PRD requirements and test-case observations.
- Do not invent information.

Answer:
"""

    print("\n========== PROMPT SENT TO LLM ==========")
    print(prompt)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]
