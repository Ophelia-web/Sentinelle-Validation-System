import { useMemo, useState } from "react";

const API_BASE = "";

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

function riskClass(value) {
  const v = String(value || "").toLowerCase();
  if (v === "high") return "riskHigh";
  if (v === "medium") return "riskMedium";
  if (v === "low") return "riskLow";
  return "";
}

function MetricCard({ label, value, risk }) {
  return (
    <div className="metricCard">
      <div className="metricLabel">{label}</div>
      <div className={cx("metricValue", risk ? riskClass(value) : "")}>{value || "unknown"}</div>
      <div className="metricLine" />
    </div>
  );
}

function DataTable({ rows, emptyText = "No records detected.", maxRows = 60 }) {
  const safeRows = asArray(rows);
  const columns = useMemo(() => {
    const keys = new Set();
    safeRows.slice(0, maxRows).forEach((row) => Object.keys(row || {}).forEach((key) => keys.add(key)));
    return Array.from(keys);
  }, [safeRows, maxRows]);

  if (!safeRows.length || !columns.length) {
    return <div className="emptyState">{emptyText}</div>;
  }

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>{columns.map((col) => <th key={col}>{col}</th>)}</tr>
        </thead>
        <tbody>
          {safeRows.slice(0, maxRows).map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col}>{String(row?.[col] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BenchmarkChart({ rows }) {
  const safeRows = asArray(rows).filter((r) => typeof r.mean_score === "number");
  if (!safeRows.length) return <div className="emptyState">No comparison results available.</div>;

  const max = Math.max(...safeRows.map((r) => r.mean_score), 1e-9);

  return (
    <div className="chart">
      {safeRows.map((row) => {
        const width = Math.max(6, (row.mean_score / max) * 100);
        return (
          <div className="barRow" key={row.validation_design}>
            <div className="barLabel">{row.validation_design}</div>
            <div className="barTrack">
              <div className="barFill" style={{ width: `${width}%` }} />
            </div>
            <div className="barValue">{Number(row.mean_score).toFixed(4)}</div>
          </div>
        );
      })}
    </div>
  );
}

function DownloadLinks({ artifacts }) {
  if (!artifacts) return null;

  const entries = [
    ["PDF report", artifacts.report_pdf],
    ["JSON plan", artifacts.json_plan],
    ["Split code", artifacts.split_code],
    ["Benchmark CSV", artifacts.benchmark_csv],
  ].filter(([, url]) => Boolean(url));

  return (
    <div className="downloadGrid">
      {entries.map(([label, url]) => (
        <a className="downloadButton" href={url} key={label} download>
          {label}
        </a>
      ))}
    </div>
  );
}

function cleanInlineMarkdown(value) {
  return String(value || "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .trim();
}

function parseMarkdownTable(lines, startIndex) {
  const tableLines = [];
  let i = startIndex;

  while (i < lines.length && lines[i].trim().startsWith("|")) {
    tableLines.push(lines[i].trim());
    i += 1;
  }

  if (tableLines.length < 2) {
    return null;
  }

  const separator = tableLines[1];
  if (!/^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(separator)) {
    return null;
  }

  const splitRow = (line) =>
    line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cleanInlineMarkdown(cell));

  const headers = splitRow(tableLines[0]);
  const rows = tableLines.slice(2).map(splitRow);

  return {
    nextIndex: i,
    table: { headers, rows },
  };
}

function MarkdownReportPreview({ markdown }) {
  if (!markdown) return <div className="emptyState">Report preview is not available.</div>;

  const lines = markdown.split(/\r?\n/);
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
      continue;
    }

    if (line.trim().startsWith("|")) {
      const parsed = parseMarkdownTable(lines, i);
      if (parsed) {
        blocks.push({ type: "table", ...parsed.table });
        i = parsed.nextIndex;
        continue;
      }
    }

    if (line.startsWith("# ")) {
      blocks.push({ type: "h1", text: cleanInlineMarkdown(line.replace(/^#\s*/, "")) });
    } else if (line.startsWith("## ")) {
      blocks.push({ type: "h2", text: cleanInlineMarkdown(line.replace(/^##\s*/, "")) });
    } else if (line.startsWith("### ")) {
      blocks.push({ type: "h3", text: cleanInlineMarkdown(line.replace(/^###\s*/, "")) });
    } else if (line.startsWith("- ")) {
      blocks.push({ type: "bullet", text: cleanInlineMarkdown(line.slice(2)) });
    } else {
      blocks.push({ type: "p", text: cleanInlineMarkdown(line) });
    }

    i += 1;
  }

  return (
    <div className="reportPreview">
      {blocks.slice(0, 120).map((block, idx) => {
        if (block.type === "h1") return <h1 key={idx}>{block.text}</h1>;
        if (block.type === "h2") return <h2 key={idx}>{block.text}</h2>;
        if (block.type === "h3") return <h3 key={idx}>{block.text}</h3>;
        if (block.type === "bullet") return <p key={idx} className="reportBullet">• {block.text}</p>;
        if (block.type === "table") {
          return (
            <div className="reportTableWrap" key={idx}>
              <table className="reportTable">
                <thead>
                  <tr>{block.headers.map((h) => <th key={h}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {block.headers.map((_, colIdx) => (
                        <td key={colIdx}>{row[colIdx] || ""}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return <p key={idx}>{block.text}</p>;
      })}
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [useDemo, setUseDemo] = useState(false);
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [targetCol, setTargetCol] = useState("");
  const [runBenchmark, setRunBenchmark] = useState(true);
  const [modelName, setModelName] = useState("random_forest");
  const [nSplits, setNSplits] = useState(3);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function loadDemo() {
    setStatus("loading-demo");
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/demo`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setUseDemo(true);
      setFile(null);
      setColumns(data.columns || []);
      setPreview(data.preview || []);
      setTargetCol(data.default_target || "");
      setStatus("idle");
    } catch (err) {
      setError(String(err.message || err));
      setStatus("idle");
    }
  }

  async function handleFileChange(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setUseDemo(false);
    setResult(null);
    setError("");

    try {
      const text = await selected.text();
      const firstLine = text.split(/\r?\n/)[0] || "";
      const guessedColumns = firstLine.split(",").map((x) => x.trim()).filter(Boolean);
      setColumns(guessedColumns);
      setTargetCol("");
      setPreview([]);
    } catch {
      setColumns([]);
      setTargetCol("");
      setPreview([]);
    }
  }

  async function runAnalysis() {
    setStatus("running");
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      if (file && !useDemo) form.append("file", file);
      form.append("use_demo", useDemo ? "true" : "false");
      form.append("target_col", targetCol || "");
      form.append("run_cv_benchmark", runBenchmark ? "true" : "false");
      form.append("model_name", modelName);
      form.append("n_splits", String(nSplits));

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form
      });

      if (!res.ok) {
        let message = await res.text();
        try {
          message = JSON.parse(message).detail || message;
        } catch {}
        throw new Error(message);
      }

      const data = await res.json();
      setResult(data);
      setColumns(data.columns || columns);
      setPreview(data.preview || preview);
      setStatus("idle");
    } catch (err) {
      setError(String(err.message || err));
      setStatus("idle");
    }
  }

  const summary = result?.summary || {};
  const recommendation = result?.recommendation || {};
  const benchmarkRows = result?.benchmark?.rows || [];

  return (
    <main className="shell">
      <section className="hero">
        <div className="heroCopy">
          <div className="eyebrow">Sentinelle Validation System</div>
          <h1>Build validation splits you can trust.</h1>
          <p>
            Upload a training table, detect time, group, and panel structure,
            and generate safer cross-validation guidance.
          </p>
        </div>

        <div className="heroVisual" aria-hidden="true">
          <div className="logoHalo" />
          <img className="heroLogo" src="/C_logo.png" alt="" />
        </div>
      </section>

      <section className="workflowBar" aria-label="Workflow steps">
        <div className="workflowLabel">Workflow</div>

        <div className="workflowSteps">
          <div className="workflowStep">
            <span className="stepBadge">1</span>
            <span>Upload data</span>
          </div>

          <span className="stepArrow">→</span>

          <div className="workflowStep">
            <span className="stepBadge">2</span>
            <span>Detect structure</span>
          </div>

          <span className="stepArrow">→</span>

          <div className="workflowStep">
            <span className="stepBadge">3</span>
            <span>Compare CV strategies</span>
          </div>

          <span className="stepArrow">→</span>

          <div className="workflowStep">
            <span className="stepBadge">4</span>
            <span>Export artifacts</span>
          </div>
        </div>
      </section>

      <section className="layout">
        <aside className="controlPanel">
          <div className="panelHeader">
            <div className="panelTitle">Analyze your dataset</div>
            <div className="panelSub">
              Upload a CSV that contains your features and, when available, the target column.
            </div>
          </div>

          <label className="fieldLabel">Upload CSV</label>

          <label className="customFileInput">
            <input type="file" accept=".csv" onChange={handleFileChange} />
            <span className="fileButtonText">Choose file</span>
            <span className="fileName">{file ? file.name : "No file selected"}</span>
          </label>

          <div className="orDivider">
            <span></span>
            <em>or</em>
            <span></span>
          </div>

          <button className="secondaryButton" onClick={loadDemo} disabled={status !== "idle"}>
            {status === "loading-demo" ? "Generating..." : "Generate sample dataset"}
          </button>
          <div className="hintText sampleHint">
            No dataset yet? Use the button above to generate a temporary sample dataset.
          </div>

          <div className="divider" />

          <label className="fieldLabel">Target column</label>
          <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
            <option value="">No target selected</option>
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>

          <label className="checkRow">
            <input
              type="checkbox"
              checked={runBenchmark}
              onChange={(e) => setRunBenchmark(e.target.checked)}
            />
            <span>Compare CV strategies</span>
          </label>

          {runBenchmark && (
            <div className="nestedOptions">
              <label className="fieldLabel">Model for comparison</label>
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                <option value="hist_gradient_boosting">hist_gradient_boosting</option>
                <option value="random_forest">random_forest</option>
                <option value="ridge">ridge</option>
                <option value="dummy">dummy</option>
              </select>

              <label className="fieldLabel">CV folds (maximum)</label>
              <input
                type="range"
                min="2"
                max="8"
                value={nSplits}
                onChange={(e) => setNSplits(Number(e.target.value))}
              />
              <div className="hintText">
                {nSplits} folds · Some validation designs may use fewer folds when the data structure requires it.
              </div>
            </div>
          )}

          <button
            className={`primaryButton ${status === "running" ? "isLoading" : ""}`}
            onClick={runAnalysis}
            disabled={status !== "idle" || (!useDemo && !file)}
          >
            {status === "running" ? (
              <span className="buttonLoading">
                <span className="spinner" />
                Analyzing...
              </span>
            ) : (
              "Analyze dataset"
            )}
          </button>

          {error && <div className="errorBox">{error}</div>}
        </aside>

        <section className="workspace">
          <div className="metricGrid">
            <MetricCard label="Task type" value={summary.task_type} />
            <MetricCard label="Detected structure" value={summary.detected_structure} />
            <MetricCard label="Primary CV" value={summary.primary_cv} />
            <MetricCard label="Risk level" value={summary.risk_level} risk />
          </div>

          <div className="card">
            <div className="cardTop">
              <div>
                <h2>Dataset</h2>
                <p>{result ? `${result.source_label} · ${result.row_count} rows` : useDemo ? "Sample dataset loaded" : file ? file.name : "No dataset loaded"}</p>
              </div>
            </div>
            {Array.isArray(result?.input_warnings) && result.input_warnings.length > 0 && (
              <div className="warningBox">
                {result.input_warnings.map((warning, idx) => (
                  <div key={idx}>{warning}</div>
                ))}
              </div>
            )}
            <DataTable rows={preview} emptyText="Load a dataset to preview rows." maxRows={10} />
          </div>

          {result && (
            <>
              <div className="card">
                <h2>Recommended validation design</h2>
                <p className="recommendationText">{recommendation.explanation || "No explanation available."}</p>
                {Array.isArray(recommendation.avoid) && recommendation.avoid.length > 0 && (
                  <div className="avoidBox">
                    <strong>Avoid</strong>
                    <ul>
                      {recommendation.avoid.map((item, idx) => <li key={idx}>{item}</li>)}
                    </ul>
                  </div>
                )}
              </div>

              <div className="twoCol">
                <div className="card">
                  <h2>Candidate time columns</h2>
                  <DataTable rows={result.time_cols} />
                </div>
                <div className="card">
                  <h2>Candidate group/entity columns</h2>
                  <DataTable rows={result.group_cols} />
                </div>
              </div>

              <div className="card">
                <h2>Leakage risks</h2>
                <DataTable rows={result.risks} emptyText="No major leakage risks detected." />
              </div>

              {(benchmarkRows.length > 0 || result.benchmark?.warning || result.benchmark?.optimism_warning?.detected) && (
                <div className="card">
                  <h2>CV strategy comparison</h2>
                  {result.benchmark?.warning && <div className="warningBox">{result.benchmark.warning}</div>}
                  {result.benchmark?.optimism_warning?.detected && (
                    <div className="warningBox">{result.benchmark.optimism_warning.message}</div>
                  )}
                  <BenchmarkChart rows={benchmarkRows} />
                  <DataTable rows={benchmarkRows} emptyText="Comparison was not run or produced no results." />
                </div>
              )}

              <div className="card">
                <h2>Generated report</h2>
                <MarkdownReportPreview markdown={result.report_markdown} />
              </div>

              <div className="card">
                <h2>Artifacts</h2>
                <p>Download the generated report, machine-readable plan, split code, and optional comparison results.</p>
                <DownloadLinks artifacts={result.artifact_urls} />
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  );
}
