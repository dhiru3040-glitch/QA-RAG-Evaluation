import math
import re
from collections import Counter


class BM25Retriever:
    """Small in-memory BM25 index over the existing ChromaDB documents."""

    def __init__(self, documents, metadatas=None, ids=None):
        self.documents = list(documents)
        self.metadatas = list(metadatas or [{} for _ in self.documents])
        self.ids = list(ids or [str(index) for index in range(len(self.documents))])
        self.tokenized_documents = [
            self.tokenize(document)
            for document in self.documents
        ]
        self.document_frequency = Counter()

        for tokens in self.tokenized_documents:
            self.document_frequency.update(set(tokens))

        self.document_count = len(self.documents)
        self.average_document_length = (
            sum(len(tokens) for tokens in self.tokenized_documents)
            / self.document_count
            if self.document_count
            else 0.0
        )

    @staticmethod
    def tokenize(text):
        return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text.lower())

    def score(self, query):
        query_tokens = self.tokenize(query)
        scores = []
        k1 = 1.5
        b = 0.75

        for index, document_tokens in enumerate(self.tokenized_documents):
            term_frequency = Counter(document_tokens)
            document_length = len(document_tokens)
            score = 0.0

            for token in set(query_tokens):
                frequency = term_frequency.get(token, 0)
                if frequency == 0:
                    continue

                document_frequency = self.document_frequency.get(token, 0)
                idf = math.log(
                    1.0
                    + (
                        self.document_count - document_frequency + 0.5
                    )
                    / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b
                    + b * document_length / max(self.average_document_length, 1.0)
                )
                score += idf * frequency * (k1 + 1.0) / denominator

            scores.append((index, score))

        return sorted(scores, key=lambda item: item[1], reverse=True)

    def retrieve(self, query, top_k=10):
        results = []

        for bm25_rank, (index, score) in enumerate(
            self.score(query)[:top_k],
            start=1
        ):
            if score <= 0.0:
                continue

            results.append({
                "content": self.documents[index],
                "metadata": self.metadatas[index],
                "document_id": self.ids[index],
                "bm25_rank": bm25_rank,
                "bm25_score": float(score)
            })

        return results
