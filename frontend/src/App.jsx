import { useEffect, useState } from "react";

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function Candidates({ title, candidates, rankKey, scoreKeys }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {!candidates?.length && <p className="muted">No candidates returned.</p>}
      {candidates?.map((candidate, index) => (
        <article className="candidate" key={`${candidate.document_id}-${index}`}>
          <div className="candidate-head">
            <strong>
              {candidate.document_id}
              {rankKey && candidate[rankKey] ? ` · Rank ${candidate[rankKey]}` : ""}
            </strong>
            {scoreKeys?.map((key) =>
              candidate[key] !== undefined && candidate[key] !== null ? (
                <span key={key}>
                  {key}: {Number(candidate[key]).toFixed(6)}
                </span>
              ) : null,
            )}
          </div>
          <div className="metadata">{JSON.stringify(candidate.metadata || {})}</div>
          <p>{candidate.content}</p>
        </article>
      ))}
    </section>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState("");

  async function refreshStatus() {
    try {
      setStatus(await api("/knowledge-base/status"));
      setStatusError("");
    } catch (err) {
      setStatusError(err.message);
    }
  }

  useEffect(() => {
    refreshStatus();
  }, []);

  async function ingest() {
    setIngesting(true);
    setError("");
    try {
      await api("/ingest", { method: "POST" });
      await refreshStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await api("/query", {
        method: "POST",
        body: JSON.stringify({ question: question.trim() }),
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">VWO QA KNOWLEDGE BASE</p>
          <h1>RAG Retrieval Dashboard</h1>
        </div>
        <button className="ingest-button" onClick={ingest} disabled={ingesting}>
          {ingesting ? "Building index..." : "Rebuild knowledge base"}
        </button>
      </header>

      {statusError && <div className="error">{statusError}</div>}
      {error && <div className="error">{error}</div>}

      <section className="stats">
        <article><span>Chunks</span><strong>{status?.chunk_count ?? "-"}</strong></article>
        <article><span>PRD chunks</span><strong>{status?.prd_chunks ?? "-"}</strong></article>
        <article><span>Test-case chunks</span><strong>{status?.test_case_chunks ?? "-"}</strong></article>
        <article><span>Sources</span><strong>{status?.sources?.length ?? "-"}</strong></article>
      </section>

      <section className="panel">
        <h2>Knowledge-base sources</h2>
        {status?.sources?.map((source) => (
          <div className="source-row" key={source.name}>
            <strong>{source.name}</strong>
            <span>{source.source_type}</span>
            <span>{source.document_count} document(s)</span>
          </div>
        ))}
      </section>

      <section className="panel query-panel">
        <h2>Ask the VWO knowledge base</h2>
        <form onSubmit={submit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about the PRD or login test cases..."
            rows={3}
          />
          <button type="submit" disabled={!question.trim() || busy}>
            {busy ? "Retrieving..." : "Query"}
          </button>
        </form>
      </section>

      {result && (
        <>
          <section className="answer panel">
            <p className="eyebrow">GENERATED ANSWER</p>
            <h2>{result.question}</h2>
            <p>{result.answer}</p>
          </section>

          <Candidates
            title="Stage 1: Vector retrieval"
            candidates={result.vector_candidates}
            rankKey="vector_rank"
            scoreKeys={["cosine_similarity", "vector_distance"]}
          />
          <Candidates
            title="Stage 1: BM25 retrieval"
            candidates={result.bm25_candidates}
            rankKey="bm25_rank"
            scoreKeys={["bm25_score"]}
          />
          <Candidates
            title="Merged: Reciprocal-rank fusion"
            candidates={result.merged_candidates}
            rankKey={null}
            scoreKeys={["fusion_score"]}
          />
          <Candidates
            title="Final: Cross-encoder reranking"
            candidates={result.reranked_candidates}
            rankKey="reranker_rank"
            scoreKeys={[
              "reranker_score",
              "combined_score",
              "reranker_normalized_score",
              "fusion_normalized_score",
            ]}
          />
        </>
      )}
    </main>
  );
}
