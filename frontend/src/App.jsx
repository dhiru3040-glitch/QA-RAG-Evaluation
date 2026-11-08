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
            <strong>{candidate.document_id}{rankKey && candidate[rankKey] ? ` · Rank ${candidate[rankKey]}` : ""}</strong>
            {scoreKeys?.map((key) =>
              candidate[key] !== undefined && candidate[key] !== null ? (
                <span key={key}>{key}: {Number(candidate[key]).toFixed(6)}</span>
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

const METRICS = [
  ["faithfulness", "FaithfulnessMetric", "Faithfulness", "Measures whether the answer is supported by retrieved context."],
  ["answer_relevancy", "AnswerRelevancyMetric", "Answer Relevancy", "Measures how relevant the answer is to the input question."],
  ["contextual_precision", "ContextualPrecisionMetric", "Contextual Precision", "Measures whether relevant context ranks ahead of irrelevant context."],
  ["contextual_recall", "ContextualRecallMetric", "Contextual Recall", "Measures coverage of the expected retrieval context."],
];

function percent(value) {
  return value === null || value === undefined ? "--" : `${Math.round(value * 100)}%`;
}

function EvaluationDashboard() {
  const [dataset, setDataset] = useState(null);
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState(null);
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      const [datasetData, resultsData, historyData, statusData] = await Promise.all([
        api("/evaluation/dataset"),
        api("/evaluation/results"),
        api("/evaluation/history"),
        api("/evaluation/status"),
      ]);
      setDataset(datasetData);
      setResults(resultsData);
      setHistory(historyData.history || []);
      setStatus(statusData);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (status?.status !== "Running") return undefined;
    const timer = setInterval(refresh, 2500);
    return () => clearInterval(timer);
  }, [status?.status]);

  async function runAll() {
    setError("");
    try {
      await api("/evaluation/run-all", { method: "POST" });
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  async function runCase(testCaseId) {
    setError("");
    try {
      await api(`/evaluation/run/${testCaseId}`, { method: "POST" });
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  const resultById = Object.fromEntries(
    (results?.results || []).filter((item) => item.id).map((item) => [item.id, item]),
  );
  const latestResult = results?.results?.find((item) => item.metrics) || null;
  const cardResult = selected || latestResult;
  const cases = (dataset?.test_cases || []).filter(
    (item) => category === "all" || item.category === category,
  );

  function caseStatus(item) {
    if (status?.status === "Running" && status.test_case_id === item.id) return "Running";
    const result = resultById[item.id];
    if (!result) return status?.status === "Running" && status.mode === "all" ? "Running" : "Pending";
    if (result.error) return "Failed";
    return result.overall_result === "passed" ? "Passed" : "Failed";
  }

  return (
    <div className="evaluation">
      <div className="evaluation-toolbar">
        <div>
          <p className="eyebrow">QUALITY ASSURANCE</p>
          <h2>Evaluation Dashboard</h2>
        </div>
        <button className="run-all" onClick={runAll} disabled={status?.status === "Running"}>
          {status?.status === "Running" ? "Evaluation running..." : "Run All Tests"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Loading evaluation data...</div>}

      <section className="stats metric-grid">
        {METRICS.map(([publicName, className, label, explanation]) => {
          const caseMetric = cardResult?.metrics?.[className];
          const metric = caseMetric || results?.average_metrics?.[publicName];
          return (
            <article
              key={publicName}
              className={`metric-card ${selectedMetric === publicName ? "selected" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => {
                if (!selected && latestResult) setSelected(latestResult);
                setSelectedMetric(publicName);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  if (!selected && latestResult) setSelected(latestResult);
                  setSelectedMetric(publicName);
                }
              }}
            >
              <span>{label}</span>
              <strong>{percent(metric?.score)}</strong>
              <small>{explanation}</small>
              <div className={`pass-indicator ${metric ? (metric.passed ? "pass" : "fail") : "neutral"}`}>
                {metric ? (metric.passed ? "Passed threshold" : "Below threshold") : "Not evaluated"}
              </div>
            </article>
          );
        })}
      </section>

      {selectedMetric && cardResult && (
        (() => {
          const metricEntry = METRICS.find(([publicName]) => publicName === selectedMetric);
          const className = metricEntry?.[1];
          const label = metricEntry?.[2] || selectedMetric;
          const metric = cardResult.metrics?.[className];
          const details = metric?.details || {};
          return (
            <section className="panel metric-detail-panel">
              <div className="detail-head">
                <div>
                  <p className="eyebrow">{cardResult.id} · {cardResult.category}</p>
                  <h2>{label} details</h2>
                </div>
                <button className="close-button" onClick={() => setSelectedMetric(null)}>Close metric</button>
              </div>
              <div className="metric-summary-row">
                <div><span>Score</span><strong>{metric?.score !== undefined ? percent(metric.score) : "--"}</strong></div>
                <div><span>Threshold</span><strong>{metric?.threshold !== undefined ? percent(metric.threshold) : "--"}</strong></div>
                <div><span>Status</span><strong>{metric ? (metric.passed ? "Passed" : "Failed") : "Not evaluated"}</strong></div>
              </div>
              <article className="deep-eval-reason">
                <h3>DeepEval evaluation reason</h3>
                <p>{metric?.reason || "Not provided by DeepEval"}</p>
              </article>
              <div className="metric-evidence-grid">
                <article>
                  <h3>Actual generated answer</h3>
                  <p>{cardResult.actual_output || "Not available"}</p>
                </article>
                <article>
                  <h3>Actual retrieved context</h3>
                  {cardResult.actual_retrieved_chunks?.length ? cardResult.actual_retrieved_chunks.map((chunk, index) => (
                    <div className="retrieved-chunk" key={index}>
                      <strong>Rank {chunk.reranker_rank ?? index + 1}</strong>
                      {chunk.source && <span>Source: {chunk.source}</span>}
                      <p>{chunk.content}</p>
                    </div>
                  )) : cardResult.actual_retrieval_context?.length ? cardResult.actual_retrieval_context.map((text, index) => (
                    <p key={index}>{text}</p>
                  )) : <p>Not available</p>}
                </article>
              </div>
              <div className="metric-evidence-grid">
                {selectedMetric === "faithfulness" && (
                  <>
                    <article><h3>Supported context truths</h3>{details.truths?.length ? <ul>{details.truths.map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul> : <p>Not provided by DeepEval</p>}</article>
                    <article><h3>Answer claims</h3>{details.claims?.length ? <ul>{details.claims.map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul> : <p>Not provided by DeepEval</p>}</article>
                    <article className="wide"><h3>Claim verdicts</h3>{details.verdicts?.length ? <ul>{details.verdicts.map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul> : <p>Not provided by DeepEval</p>}</article>
                  </>
                )}
                {selectedMetric === "answer_relevancy" && (
                  <>
                    <article><h3>Extracted answer statements</h3>{details.statements?.length ? <ul>{details.statements.map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul> : <p>Not available</p>}</article>
                    <article><h3>Relevancy verdicts</h3>{details.verdicts?.length ? <ul>{details.verdicts.map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul> : <p>Not available</p>}</article>
                  </>
                )}
                {(selectedMetric === "contextual_precision" || selectedMetric === "contextual_recall") && (
                  <article className="wide"><h3>DeepEval verdict details</h3>{details.verdicts?.length ? <ul>{details.verdicts.map((item, index) => <li key={index}>{typeof item === "string" ? item : JSON.stringify(item)}</li>)}</ul> : <p>Not available</p>}</article>
                )}
              </div>
            </section>
          );
        })()
      )}

      <section className="stats summary-grid">
        <article><span>Total test cases</span><strong>{results?.total_test_cases ?? dataset?.total ?? "-"}</strong></article>
        <article><span>Passed test cases</span><strong>{results?.passed_test_cases ?? "--"}</strong></article>
        <article><span>Failed test cases</span><strong>{results?.failed_test_cases ?? "--"}</strong></article>
        <article><span>Evaluation status</span><strong className="status-text">{status?.status || "Not evaluated"}</strong></article>
      </section>

      <section className="panel run-token-section">
        <h2>Token Usage</h2>
        <p className="muted">Actual Ollama usage only. Generation and judge calls are tracked separately.</p>
        <div className="token-usage-grid">
          <article>
            <strong>Total RAG generation tokens</strong>
            <div><span>Input / prompt</span><b>{results?.token_usage?.rag_generation?.input_tokens ?? "Not available"}</b></div>
            <div><span>Output / completion</span><b>{results?.token_usage?.rag_generation?.output_tokens ?? "Not available"}</b></div>
            <div><span>Total generation</span><b>{results?.token_usage?.rag_generation?.total_tokens ?? "Not available"}</b></div>
          </article>
          <article>
            <strong>Total DeepEval judge tokens</strong>
            <div><span>Input / prompt</span><b>{results?.token_usage?.deep_eval?.input_tokens ?? "Not available"}</b></div>
            <div><span>Output / completion</span><b>{results?.token_usage?.deep_eval?.output_tokens ?? "Not available"}</b></div>
            <div><span>Total judge</span><b>{results?.token_usage?.deep_eval?.total_tokens ?? "Not available"}</b></div>
          </article>
          <article>
            <strong>Complete run overall</strong>
            <div><span>Generation + evaluation</span><b>{results?.token_usage?.overall_run ?? "Not available"}</b></div>
            <small>{results?.token_usage?.scope || "Completed test cases only"}</small>
          </article>
        </div>
      </section>

      <section className="panel evaluation-controls">
        <div className="filter-row">
          <label>
            Test category
            <select value={category} onChange={(event) => setCategory(event.target.value)}>
              <option value="all">All categories</option>
              {(dataset?.categories || []).map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <span className="muted">{cases.length} test case(s)</span>
        </div>

        {!cases.length ? (
          <div className="empty-state">No test cases available.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th><th>Category</th><th>Input question</th><th>Status</th>
                  {METRICS.map(([, , label]) => <th key={label}>{label}</th>)}
                  <th>Overall</th><th></th>
                </tr>
              </thead>
              <tbody>
                {cases.map((item) => {
                  const result = resultById[item.id];
                  const rowStatus = caseStatus(item);
                  return (
                    <tr key={item.id} onClick={() => {
                      if (result) {
                        setSelected(result);
                        setSelectedMetric(null);
                      }
                    }}>
                      <td><strong>{item.id}</strong></td>
                      <td><span className="category">{item.category}</span></td>
                      <td className="question-cell">{item.input}</td>
                      <td><span className={`status-badge ${rowStatus.toLowerCase()}`}>{rowStatus}</span></td>
                      {METRICS.map(([, className]) => (
                        <td key={className}>
                          {result?.metrics?.[className]?.score !== undefined
                            ? percent(result.metrics[className].score)
                            : "--"}
                        </td>
                      ))}
                      <td>{result?.error ? "Failed" : result ? (result.overall_result === "passed" ? "Passed" : "Failed") : "--"}</td>
                      <td>
                        <button
                          className="run-case"
                          disabled={status?.status === "Running"}
                          onClick={(event) => { event.stopPropagation(); runCase(item.id); }}
                        >
                          Run
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selected && (
        <section className="panel detail-panel">
          <div className="detail-head">
            <div><p className="eyebrow">{selected.id} · {selected.category}</p><h2>Detailed evaluation result</h2></div>
            <button className="close-button" onClick={() => setSelected(null)}>Close</button>
          </div>
          <div className="detail-grid">
            <article><h3>Input question</h3><p>{selected.input}</p></article>
            <article><h3>Expected answer</h3><p>{selected.expected_output}</p></article>
            <article><h3>Actual generated answer</h3><p>{selected.actual_output || selected.error || "Unavailable"}</p></article>
            <article><h3>Expected retrieval context</h3>{selected.expected_retrieval_context?.map((text, index) => <p key={index}>{text}</p>)}</article>
            <article className="wide">
              <h3>Actual retrieved context</h3>
              {selected.actual_retrieved_chunks?.length ? selected.actual_retrieved_chunks.map((chunk, index) => (
                <div className="retrieved-chunk" key={index}>
                  <strong>Rank {chunk.reranker_rank ?? index + 1}</strong>
                  {chunk.source && <span>Source: {chunk.source}</span>}
                  {(chunk.vector_rank !== null && chunk.vector_rank !== undefined) && <span>Vector rank: {chunk.vector_rank}</span>}
                  {(chunk.bm25_rank !== null && chunk.bm25_rank !== undefined) && <span>BM25 rank: {chunk.bm25_rank}</span>}
                  <p>{chunk.content}</p>
                </div>
              )) : selected.actual_retrieval_context?.length ? selected.actual_retrieval_context.map((text, index) => (
                <p key={index}>{text}</p>
              )) : <p className="muted">Not available</p>}
            </article>
          </div>
          <section className="token-usage-section">
            <h3>Token Usage</h3>
            <div className="token-usage-grid">
              {[
                ["RAG generation", selected.token_usage?.rag_generation],
                ["DeepEval evaluation", selected.token_usage?.deep_eval],
              ].map(([label, usage]) => (
                <article key={label}>
                  <strong>{label}</strong>
                  <div><span>Input / prompt</span><b>{usage?.input_tokens ?? "Not available"}</b></div>
                  <div><span>Output / completion</span><b>{usage?.output_tokens ?? "Not available"}</b></div>
                  <div><span>Total</span><b>{usage?.total_tokens ?? "Not available"}</b></div>
                  <small>Source: {usage?.source || "Not available"}</small>
                </article>
              ))}
              <article>
                <strong>Overall test-case tokens</strong>
                <div><span>Combined generation + evaluation</span><b>{selected.token_usage?.overall_test_case ?? "Not available"}</b></div>
              </article>
            </div>
            <div className="metric-token-grid">
              {METRICS.map(([, className, label]) => {
                const usage = selected.token_usage?.per_metric?.[className];
                return (
                  <article key={className}>
                    <strong>{label}</strong>
                    <span>Judge input: {usage?.input_tokens ?? "Not available"}</span>
                    <span>Judge output: {usage?.output_tokens ?? "Not available"}</span>
                    <span>Judge total: {usage?.total_tokens ?? "Not available"}</span>
                  </article>
                );
              })}
            </div>
          </section>
          <div className="metric-detail-grid">
            {METRICS.map(([publicName, className, label]) => {
              const metric = selected.metrics?.[className];
              return (
                <article key={publicName}>
                  <div className="metric-detail-head"><strong>{label}</strong><span>{metric ? percent(metric.score) : "--"}</span></div>
                  <div className={`pass-indicator ${metric ? (metric.passed ? "pass" : "fail") : "neutral"}`}>{metric ? (metric.passed ? "Passed" : "Failed") : "Not evaluated"}</div>
                  <p>{metric?.reason || "No evaluation reason recorded."}</p>
                </article>
              );
            })}
          </div>
        </section>
      )}

      <section className="panel history-panel">
        <h2>Evaluation history</h2>
        {!history.length ? (
          <div className="empty-state">No previous evaluation runs.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Run ID</th><th>Date / time</th><th>Total</th><th>Passed</th><th>Failed</th>{METRICS.map(([, , label]) => <th key={label}>{label}</th>)}</tr></thead>
              <tbody>
                {history.map((run) => (
                  <tr key={run.run_id || run.evaluation_date}>
                    <td><strong>{run.run_id || "--"}</strong></td>
                    <td>{run.evaluation_date ? new Date(run.evaluation_date).toLocaleString() : "--"}</td>
                    <td>{run.total_test_cases ?? "--"}</td>
                    <td>{run.passed_test_cases ?? "--"}</td>
                    <td>{run.failed_test_cases ?? "--"}</td>
                    {METRICS.map(([publicName]) => <td key={publicName}>{percent(run.average_metrics?.[publicName]?.score)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("knowledge");
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

  useEffect(() => { refreshStatus(); }, []);

  async function ingest() {
    setIngesting(true);
    setError("");
    try {
      await api("/ingest", { method: "POST" });
      await refreshStatus();
    } catch (err) { setError(err.message); } finally { setIngesting(false); }
  }

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await api("/query", { method: "POST", body: JSON.stringify({ question: question.trim() }) }));
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  return (
    <main>
      <header>
        <div><p className="eyebrow">VWO QA KNOWLEDGE BASE</p><h1>RAG Application</h1></div>
        <nav className="app-tabs">
          <button className={tab === "knowledge" ? "active" : ""} onClick={() => setTab("knowledge")}>Knowledge Base</button>
          <button className={tab === "evaluation" ? "active" : ""} onClick={() => setTab("evaluation")}>Evaluation Dashboard</button>
        </nav>
      </header>

      {tab === "evaluation" ? <EvaluationDashboard /> : (
        <>
          <div className="knowledge-toolbar">
            <button className="ingest-button" onClick={ingest} disabled={ingesting}>{ingesting ? "Building index..." : "Rebuild knowledge base"}</button>
          </div>
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
            {status?.sources?.map((source) => <div className="source-row" key={source.name}><strong>{source.name}</strong><span>{source.source_type}</span><span>{source.document_count} document(s)</span></div>)}
          </section>
          <section className="panel query-panel">
            <h2>Ask the VWO knowledge base</h2>
            <form onSubmit={submit}>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about the PRD or login test cases..." rows={3} />
              <button type="submit" disabled={!question.trim() || busy}>{busy ? "Retrieving..." : "Query"}</button>
            </form>
          </section>
          {result && (
            <>
              <section className="answer panel"><p className="eyebrow">GENERATED ANSWER</p><h2>{result.question}</h2><p>{result.answer}</p></section>
              <Candidates title="Stage 1: Vector retrieval" candidates={result.vector_candidates} rankKey="vector_rank" scoreKeys={["cosine_similarity", "vector_distance"]} />
              <Candidates title="Stage 1: BM25 retrieval" candidates={result.bm25_candidates} rankKey="bm25_rank" scoreKeys={["bm25_score"]} />
              <Candidates title="Merged: Reciprocal-rank fusion" candidates={result.merged_candidates} rankKey={null} scoreKeys={["fusion_score"]} />
              <Candidates title="Final: Cross-encoder reranking" candidates={result.reranked_candidates} rankKey="reranker_rank" scoreKeys={["reranker_score", "combined_score", "reranker_normalized_score", "fusion_normalized_score"]} />
            </>
          )}
        </>
      )}
    </main>
  );
}
