"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { TestRun, LogEvent } from "@/lib/db";
import type { LlmStatus, ModelSummary } from "@/lib/llm";

type Props = {
  initialRuns: TestRun[];
  initialLogs: LogEvent[];
  initialStatus: LlmStatus | null;
  initialModels: ModelSummary[];
  apiBase: string;
};

export default function Dashboard({
  initialRuns,
  initialLogs,
  initialStatus,
  initialModels,
  apiBase,
}: Props) {
  const [form, setForm] = useState({
    model_name: "",
    scenario: "",
    notes: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [runs, setRuns] = useState<TestRun[]>(initialRuns);
  const [logs, setLogs] = useState<LogEvent[]>(initialLogs);
  const [status, setStatus] = useState<LlmStatus | null>(initialStatus);
  const [models, setModels] = useState<ModelSummary[]>(initialModels);
  const [statusError, setStatusError] = useState<string | null>(null);

  const summarizeStatus = useMemo(() => {
    if (!status) {
      return {
        state: "API offline",
        detail: "Unable to reach llm-server",
        badge: "failed",
      };
    }
    if (!status.loaded_model) {
      return {
        state: "Idle",
        detail: "No model loaded",
        badge: "pending",
      };
    }
    return {
      state: `Loaded: ${status.loaded_model}`,
      detail: status.performance_logging ? "Logging enabled" : "Logging off",
      badge: "success",
    };
  }, [status]);

  const refreshTestRuns = useCallback(async () => {
    const res = await fetch("/api/test-runs");
    if (res.ok) {
      const data = await res.json();
      setRuns(data.runs);
    }
  }, []);

  const refreshLogs = useCallback(async () => {
    const res = await fetch("/api/log-events");
    if (res.ok) {
      const data = await res.json();
      setLogs(data.logs);
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/status`);
      if (!res.ok) {
        throw new Error(res.statusText);
      }
      const data = (await res.json()) as LlmStatus;
      setStatus(data);
      setStatusError(null);
    } catch (error) {
      setStatusError("Unable to reach llm-server");
      console.warn(error);
    }
  }, [apiBase]);

  const refreshModels = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/models`);
      if (!res.ok) return;
      const data = (await res.json()) as ModelSummary[];
      setModels(data);
    } catch (error) {
      console.warn(error);
    }
  }, [apiBase]);

  useEffect(() => {
    const statusInterval = setInterval(() => {
      refreshStatus();
    }, 8000);
    const logsInterval = setInterval(() => {
      refreshLogs();
    }, 10000);
    return () => {
      clearInterval(statusInterval);
      clearInterval(logsInterval);
    };
  }, [refreshStatus, refreshLogs]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      const res = await fetch("/api/test-runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const error = await res.json();
        alert(`Failed to create test run: ${error.error || res.statusText}`);
      } else {
        const data = await res.json();
        setRuns((prev) => [data.run, ...prev].slice(0, 15));
        setForm({ model_name: "", scenario: "", notes: "" });
      }
    } catch (error) {
      alert(`Network error: ${(error as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page-wrapper">
      <header>
        <p className="muted" style={{ textTransform: "uppercase" }}>
          Vector-Tester
        </p>
        <h1 className="page-title">LLM Model Testing</h1>
      </header>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h2>LLM API Status</h2>
            <span className={`status-pill ${summarizeStatus.badge}`}>
              {summarizeStatus.state}
            </span>
          </div>
          <p className="muted">{summarizeStatus.detail}</p>
          {status && (
            <dl className="grid" style={{ gridTemplateColumns: "repeat(2,1fr)" }}>
              <div>
                <dt className="muted">GPU Allocated</dt>
                <dd>{status.gpu_memory_allocated_mb?.toFixed(1) ?? "0"} MB</dd>
              </div>
              <div>
                <dt className="muted">GPU Reserved</dt>
                <dd>{status.gpu_memory_reserved_mb?.toFixed(1) ?? "0"} MB</dd>
              </div>
            </dl>
          )}
          {statusError && <p className="muted">{statusError}</p>}
          <button className="btn" style={{ marginTop: 12 }} onClick={refreshStatus}>
            Refresh Status
          </button>
        </section>

        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h2>Registered Models</h2>
            <button className="btn" onClick={refreshModels}>
              Refresh
            </button>
          </div>
          <div style={{ maxHeight: 220, overflowY: "auto" }}>
            {models.length === 0 && (
              <p className="muted">No models registered.</p>
            )}
            {models.map((model) => (
              <div
                key={model.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "6px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <div>
                  <strong>{model.model_name}</strong>
                  <p className="muted" style={{ margin: 0 }}>
                    {model.cache_location} · {model.compatibility_status ?? "unknown"}
                  </p>
                </div>
                <span className="muted">{model.total_inferences} runs</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <section className="card">
          <h2>Start Manual Test</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Model Name</label>
              <input
                value={form.model_name}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, model_name: e.target.value }))
                }
                required
                placeholder="e.g., Qwen2.5-3B-Instruct"
              />
            </div>
            <div className="form-group">
              <label>Scenario / Notes</label>
              <input
                value={form.scenario}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, scenario: e.target.value }))
                }
                placeholder="Latency test, GPU swap, etc."
              />
            </div>
            <div className="form-group">
              <label>Operator Notes</label>
              <textarea
                rows={3}
                value={form.notes}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, notes: e.target.value }))
                }
              />
            </div>
            <button className="btn" type="submit" disabled={submitting}>
              {submitting ? "Logging run..." : "Log Test Attempt"}
            </button>
          </form>
        </section>

        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h2>Recent Runs</h2>
            <button className="btn" onClick={refreshTestRuns}>
              Refresh
            </button>
          </div>
          <div style={{ maxHeight: 260, overflowY: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <strong>{run.model_name}</strong>
                      <p className="muted" style={{ margin: 0 }}>
                        {run.scenario || "—"}
                      </p>
                    </td>
                    <td>
                      <span className={`status-pill ${run.status}`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="muted">
                      {new Date(run.started_at).toLocaleTimeString()}
                    </td>
                    <td className="muted">
                      {run.load_duration_ms
                        ? `${run.load_duration_ms.toFixed(0)} ms`
                        : "—"}
                    </td>
                  </tr>
                ))}
                {runs.length === 0 && (
                  <tr>
                    <td colSpan={4} className="muted">
                      No test runs recorded yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <h2>Log Stream</h2>
          <button className="btn" onClick={refreshLogs}>
            Refresh Logs
          </button>
        </div>
        <div style={{ maxHeight: 320, overflowY: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Source</th>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.id}>
                  <td className="muted">
                    {new Date(log.created_at).toLocaleTimeString()}
                  </td>
                  <td className="muted">{log.source}</td>
                  <td>
                    <strong>[{log.level}]</strong> {log.message}
                  </td>
                </tr>
              ))}
              {logs.length === 0 && (
                <tr>
                  <td colSpan={3} className="muted">
                    No log events captured.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
