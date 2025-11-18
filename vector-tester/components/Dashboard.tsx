"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { TestRun, LogEvent, TestModelCopy } from "@/lib/db";
import type { LlmStatus, ModelSummary } from "@/lib/llm";

type SidebarItem = { label: string; anchor?: string };

type Props = {
  initialRuns: TestRun[];
  initialLogs: LogEvent[];
  initialStatus: LlmStatus | null;
  initialModels: ModelSummary[];
  initialLocalCopies: TestModelCopy[];
};

export default function Dashboard({
  initialRuns,
  initialLogs,
  initialStatus,
  initialModels,
  initialLocalCopies,
}: Props) {
  const [form, setForm] = useState({
    model_name: "",
    scenario: "",
    notes: "",
  });
  const [activeTab, setActiveTab] = useState("dashboard");
  const [submitting, setSubmitting] = useState(false);
  const [runs, setRuns] = useState<TestRun[]>(initialRuns);
  const [logs, setLogs] = useState<LogEvent[]>(initialLogs);
  const [status, setStatus] = useState<LlmStatus | null>(initialStatus);
  const [models, setModels] = useState<ModelSummary[]>(initialModels);
  const [localCopies, setLocalCopies] = useState<TestModelCopy[]>(
    initialLocalCopies
  );
  const [statusError, setStatusError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState(
    initialModels[0]?.model_name ?? ""
  );
  const [offlineSyncMessage, setOfflineSyncMessage] = useState<string | null>(
    null
  );
  const [runnerMessages, setRunnerMessages] = useState({
    health: "",
    logging: "",
    status: "",
    unload: "",
    load: "",
    validate: "",
    inference: "",
  });
  const [forceLoad, setForceLoad] = useState(false);
  const [samplePrompt, setSamplePrompt] = useState(
    "Hello model, please confirm you can respond to this diagnostic request."
  );
  const [altConfigNotes, setAltConfigNotes] = useState("");

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

  const logTesterEvent = useCallback(
    async (message: string, level: "debug" | "info" | "warn" | "error" = "info") => {
      try {
        await fetch("/api/log-events", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source: "vector-tester-runner",
            level,
            message,
          }),
        });
      } catch (error) {
        console.warn("Failed to log tester event", error);
      }
    },
    []
  );

  const responseMessage = (payload: unknown) => {
    if (typeof payload === "string") {
      return payload;
    }
    if (
      payload &&
      typeof payload === "object" &&
      "message" in payload &&
      typeof (payload as any).message === "string"
    ) {
      return (payload as any).message as string;
    }
    if (
      payload &&
      typeof payload === "object" &&
      "detail" in payload &&
      typeof (payload as any).detail === "string"
    ) {
      return (payload as any).detail as string;
    }
    return JSON.stringify(payload ?? {});
  };

  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch(`/api/llm/status`);
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
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const res = await fetch(`/api/llm/models`);
      if (!res.ok) return;
      const payload = await res.json();
      const data = (payload.models ?? []) as ModelSummary[];
      setModels(data);
    } catch (error) {
      console.warn(error);
    }
  }, []);

  useEffect(() => {
    if (models.length === 0) {
      setSelectedModel("");
      return;
    }
    if (!models.find((m) => m.model_name === selectedModel)) {
      setSelectedModel(models[0].model_name);
    }
  }, [models, selectedModel]);

  useEffect(() => {
    if (!form.model_name && localCopies.length > 0) {
      setForm((prev) => ({
        ...prev,
        model_name: localCopies[0].model_name,
      }));
    }
  }, [localCopies, form.model_name]);

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

  const handleOfflineSyncRequest = async () => {
    if (!selectedModel) return;
    try {
      const res = await fetch("/api/models/offline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: selectedModel,
        }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || "Failed to cache model");
      }
      const data = await res.json();
      setLocalCopies((prev) => {
        const filtered = prev.filter(
          (copy) => copy.model_name !== selectedModel
        );
        return [data.model, ...filtered];
      });
      setOfflineSyncMessage(`Offline copy captured for ${selectedModel}.`);
    } catch (error) {
      setOfflineSyncMessage(
        `Could not log request: ${(error as Error).message}`
      );
    }
  };

  const updateRunnerMessage = (key: keyof typeof runnerMessages, value: string) =>
    setRunnerMessages((prev) => ({ ...prev, [key]: value }));

  const handleHealthCheck = async () => {
    updateRunnerMessage("health", "Checking...");
    try {
      const res = await fetch("/api/llm/health");
      const data = await res.json();
      if (res.ok) {
        const message = responseMessage(data);
        updateRunnerMessage("health", message);
        await logTesterEvent(`Health check: ${message}`);
      } else {
        const message = responseMessage(data);
        updateRunnerMessage("health", `Failed: ${message}`);
        await logTesterEvent(`Health check failed: ${message}`, "error");
      }
    } catch (error) {
      const message = (error as Error).message;
      updateRunnerMessage("health", `Error: ${message}`);
      await logTesterEvent(`Health check error: ${message}`, "error");
    }
  };

  const handleStatusCheck = async () => {
    updateRunnerMessage("status", "Requesting status...");
    try {
      await refreshStatus();
      updateRunnerMessage("status", "Status refreshed from llm-server.");
      await logTesterEvent("Status refreshed via Test Runner.");
    } catch (error) {
      const message = (error as Error).message;
      updateRunnerMessage("status", `Failed: ${message}`);
      await logTesterEvent(`Status refresh failed: ${message}`, "error");
    }
  };

  const handleUnloadModel = async () => {
    updateRunnerMessage("unload", "Requesting unload...");
    try {
      const res = await fetch("/api/llm/orchestrate/unload", {
        method: "POST",
      });
      const data = await res.json();
      if (res.ok) {
        const msg = responseMessage(data);
        updateRunnerMessage("unload", msg || "Model unloaded.");
        await logTesterEvent(`Unload success: ${msg}`);
        await refreshStatus();
      } else {
        const msg = responseMessage(data);
        updateRunnerMessage("unload", `Failed: ${msg}`);
        await logTesterEvent(`Unload failed: ${msg}`, "error");
      }
    } catch (error) {
      const msg = (error as Error).message;
      updateRunnerMessage("unload", `Error: ${msg}`);
      await logTesterEvent(`Unload error: ${msg}`, "error");
    }
  };

  const handleLoadModel = async () => {
    if (!form.model_name) {
      updateRunnerMessage("load", "Select a staged model first.");
      return;
    }
    updateRunnerMessage("load", "Loading model...");
    try {
      const res = await fetch("/api/llm/orchestrate/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: form.model_name,
          force_load: forceLoad,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        const msg = responseMessage(data);
        updateRunnerMessage("load", msg || "Model loaded successfully.");
        await logTesterEvent(`Load success for ${form.model_name}: ${msg}`);
        await refreshStatus();
      } else {
        const msg = responseMessage(data);
        updateRunnerMessage("load", `Failed: ${msg}`);
        await logTesterEvent(
          `Load failed for ${form.model_name}: ${msg}`,
          "error"
        );
      }
    } catch (error) {
      const msg = (error as Error).message;
      updateRunnerMessage("load", `Error: ${msg}`);
      await logTesterEvent(`Load error for ${form.model_name}: ${msg}`, "error");
    }
  };

  const handleSaveAltNotes = async () => {
    if (!altConfigNotes.trim()) {
      return;
    }
    await logTesterEvent(
      `Alternate config note for ${form.model_name || "unknown"}: ${altConfigNotes}`,
      "debug"
    );
    setAltConfigNotes("");
  };

  const handleValidateModel = async () => {
    updateRunnerMessage("validate", "Validating model...");
    try {
      const res = await fetch("/api/llm/orchestrate/validate");
      const data = await res.json();
      if (res.ok) {
        const msg = responseMessage(data);
        updateRunnerMessage("validate", msg || "Validation successful.");
        await logTesterEvent(`Validation success: ${msg}`);
      } else {
        const msg = responseMessage(data);
        updateRunnerMessage("validate", `Failed: ${msg}`);
        await logTesterEvent(`Validation failed: ${msg}`, "error");
      }
    } catch (error) {
      const msg = (error as Error).message;
      updateRunnerMessage("validate", `Error: ${msg}`);
      await logTesterEvent(`Validation error: ${msg}`, "error");
    }
  };

  const handleInferenceTest = async () => {
    updateRunnerMessage("inference", "Running inference...");
    try {
      const res = await fetch("/api/llm/inference/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: samplePrompt,
          max_tokens: 128,
          temperature: 0.7,
          top_p: 0.9,
          do_sample: true,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        const msg =
          typeof data === "object" && data && "generated_text" in data
            ? (data as any).generated_text
            : responseMessage(data);
        updateRunnerMessage("inference", msg);
        await logTesterEvent("Inference test succeeded.");
      } else {
        const msg = responseMessage(data);
        updateRunnerMessage("inference", `Failed: ${msg}`);
        await logTesterEvent(`Inference test failed: ${msg}`, "error");
      }
    } catch (error) {
      const msg = (error as Error).message;
      updateRunnerMessage("inference", `Error: ${msg}`);
      await logTesterEvent(`Inference test error: ${msg}`, "error");
    }
  };

  const tabs = [
    { id: "dashboard", label: "Dashboard", summary: "High-level overview" },
    { id: "registry", label: "Model Registry", summary: "Manage models" },
    { id: "runner", label: "Test Runner", summary: "Execute new tests" },
    { id: "results", label: "Results & Analytics", summary: "Review history" },
    { id: "settings", label: "Settings", summary: "App configuration" },
  ];

  const sidebarSections: Record<string, SidebarItem[]> = {
    dashboard: [
      { label: "System Status" },
      { label: "Quick Actions" },
      { label: "Recent Activity" },
    ],
    registry: [
      { label: "Select model" },
      { label: "Offline copies" },
      { label: "Registry sync" },
    ],
    runner: [
      { label: "Health Check", anchor: "runner-health" },
      { label: "Logging & Status", anchor: "runner-logging" },
      { label: "Unload Model", anchor: "runner-unload" },
      { label: "Load Model", anchor: "runner-load" },
      { label: "Alternate Configs", anchor: "runner-alt" },
      { label: "Capture Logs", anchor: "runner-logs" },
      { label: "Validate & Inference", anchor: "runner-validate" },
      { label: "Record Run", anchor: "runner-record" },
    ],
    results: [
      { label: "Recent runs" },
      { label: "Log timeline" },
      { label: "Insights" },
    ],
    settings: [
      { label: "API connectivity" },
      { label: "Database" },
      { label: "Telemetry" },
    ],
  };

  const renderModelsList = () => (
    <div style={{ maxHeight: 260, overflowY: "auto" }}>
      {models.length === 0 && <p className="muted">No models registered.</p>}
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
  );

  const renderRunsTable = () => (
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
              <span className={`status-pill ${run.status}`}>{run.status}</span>
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
  );

  const renderLogsTable = () => (
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
  );

  const handleNavigateTo = (tabId: string) => setActiveTab(tabId);

  const renderDashboardTab = () => {
    const compatibleCount = models.filter(
      (m) => m.compatibility_status === "compatible"
    ).length;
    const recentRuns = runs.slice(0, 4);

    return (
      <>
        <div className="grid grid-2">
          <section className="card">
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <h2>System Status</h2>
              <span className={`status-pill ${summarizeStatus.badge}`}>
                {summarizeStatus.state}
              </span>
            </div>
            <p className="muted">{summarizeStatus.detail}</p>
            <dl className="grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
              <div>
                <dt className="muted">LLM API Status</dt>
                <dd>{status ? summarizeStatus.state : "Offline"}</dd>
              </div>
              <div>
                <dt className="muted">GPU Allocated</dt>
                <dd>{status?.gpu_memory_allocated_mb?.toFixed(1) ?? "0.0"} MB</dd>
              </div>
              <div>
                <dt className="muted">GPU Reserved</dt>
                <dd>{status?.gpu_memory_reserved_mb?.toFixed(1) ?? "0.0"} MB</dd>
              </div>
              <div>
                <dt className="muted">Logging</dt>
                <dd>{status?.performance_logging ? "Enabled" : "Disabled"}</dd>
              </div>
            </dl>
            {statusError && <p className="muted">{statusError}</p>}
            <button className="btn" style={{ marginTop: 12 }} onClick={refreshStatus}>
              Refresh Status
            </button>
          </section>
          <section className="card">
            <h2>Quick Actions</h2>
            <p className="muted">Jump straight into frequently used tools.</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <button className="btn" onClick={() => handleNavigateTo("runner")}>
                Start Manual Test
              </button>
              <button className="btn" onClick={() => handleNavigateTo("registry")}>
                Manage Models
              </button>
              <button className="btn" onClick={() => handleNavigateTo("results")}>
                Review Results
              </button>
            </div>
          </section>
        </div>

        <div className="grid grid-2">
          <section className="card">
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <h2>Recent Runs</h2>
              <button className="btn" onClick={() => handleNavigateTo("results")}>
                View All
              </button>
            </div>
            {recentRuns.length === 0 && (
              <p className="muted">No recent runs logged.</p>
            )}
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {recentRuns.map((run) => (
                <li
                  key={run.id}
                  style={{
                    borderBottom: "1px solid rgba(255,255,255,0.05)",
                    padding: "8px 0",
                  }}
                >
                  <strong>{run.model_name}</strong>
                  <p className="muted" style={{ margin: 0 }}>
                    {run.status} ·{" "}
                    {new Date(run.started_at).toLocaleTimeString()}
                  </p>
                </li>
              ))}
            </ul>
          </section>

          <section className="card">
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <h2>Model Status</h2>
              <button className="btn" onClick={() => handleNavigateTo("registry")}>
                Manage Models
              </button>
            </div>
            <p className="muted">
              {compatibleCount} of {models.length} models marked compatible.
            </p>
            {renderModelsList()}
          </section>
        </div>

        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h2>Log Stream</h2>
            <button className="btn" onClick={refreshLogs}>
              Refresh Logs
            </button>
          </div>
          <div style={{ maxHeight: 260, overflowY: "auto" }}>
            {renderLogsTable()}
          </div>
        </section>
      </>
    );
  };

  const renderLocalCopies = () => (
    <div style={{ maxHeight: 260, overflowY: "auto" }}>
      {localCopies.length === 0 && (
        <p className="muted">No offline copies captured yet.</p>
      )}
      {localCopies.map((copy) => (
        <div
          key={copy.id}
          style={{
            padding: "8px 0",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}
        >
          <strong>{copy.model_name}</strong>
          <p className="muted" style={{ margin: 0 }}>
            Cached {new Date(copy.cached_at).toLocaleString()} · Status:{" "}
            {copy.status}
          </p>
        </div>
      ))}
    </div>
  );

  const renderRegistryTab = () => (
    <>
      <section className="card">
        <h2>Model Offline Preparation</h2>
        <div className="form-group">
          <label>Model to stage</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.map((model) => (
              <option key={model.id} value={model.model_name}>
                {model.model_name}
              </option>
            ))}
          </select>
        </div>
        <button className="btn" onClick={handleOfflineSyncRequest} disabled={!selectedModel}>
          Request Offline Copy
        </button>
        {offlineSyncMessage && (
          <p className="muted" style={{ marginTop: 8 }}>
            {offlineSyncMessage}
          </p>
        )}
        <p className="muted" style={{ marginTop: 12 }}>
          Requesting an offline copy stores the current registry record inside Vector-Tester&apos;s
          database so testing can proceed even if the primary API goes offline.
        </p>
      </section>
      <section className="card">
        <h2>Registry Snapshot</h2>
        {renderModelsList()}
      </section>
      <section className="card">
        <h2>Models Staged for Testing</h2>
        {renderLocalCopies()}
      </section>
    </>
  );

  const renderRunnerTab = () => (
    <>
      <section className="card" id="runner-health">
        <h2>Health Check</h2>
        <p className="muted">
          Ping the LLM API health endpoint to confirm the orchestration backend is reachable before
          testing.
        </p>
        <button className="btn" onClick={handleHealthCheck}>
          Run Health Check
        </button>
        {runnerMessages.health && (
          <p className="muted" style={{ marginTop: 8 }}>
            {runnerMessages.health}
          </p>
        )}
      </section>

      <section className="card" id="runner-logging">
        <h2>Logging & Status Pre-check</h2>
        <p className="muted">
          Ensure performance logging is enabled before loading models so failures are captured in the
          main database.
        </p>
        <p className="muted">
          Current Logging:{" "}
          {status?.performance_logging ? "Enabled" : "Disabled (toggle in main UI)"}{" "}
        </p>
        <button className="btn" onClick={handleStatusCheck}>
          Refresh Status
        </button>
        {runnerMessages.status && (
          <p className="muted" style={{ marginTop: 8 }}>
            {runnerMessages.status}
          </p>
        )}
      </section>

      <section className="card" id="runner-unload">
        <h2>Unload Previous Model</h2>
        <p className="muted">
          Always free VRAM before attempting a fresh load to avoid conflicts with incompatible
          models.
        </p>
        <button className="btn" onClick={handleUnloadModel}>
          Unload Current Model
        </button>
        {runnerMessages.unload && (
          <p className="muted" style={{ marginTop: 8 }}>
            {runnerMessages.unload}
          </p>
        )}
      </section>

      <section className="card" id="runner-load">
        <h2>Load Model</h2>
        <div className="form-group">
          <label>Staged Model</label>
          <select
            value={form.model_name}
            onChange={(e) =>
              setForm((prev) => ({
                ...prev,
                model_name: e.target.value,
              }))
            }
            required
          >
            <option value="" disabled>
              Select a staged model
            </option>
            {localCopies.map((copy) => (
              <option key={copy.id} value={copy.model_name}>
                {copy.model_name}
              </option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>
            <input
              type="checkbox"
              checked={forceLoad}
              onChange={(e) => setForceLoad(e.target.checked)}
            />{" "}
            Force load even if marked incompatible
          </label>
        </div>
        <button className="btn" onClick={handleLoadModel} disabled={!form.model_name}>
          Load Selected Model
        </button>
        {runnerMessages.load && (
          <p className="muted" style={{ marginTop: 8 }}>
            {runnerMessages.load}
          </p>
        )}
      </section>

      <section className="card" id="runner-alt">
        <h2>Alternate Load Parameters</h2>
        <p className="muted">
          Track variations (attn implementation, dtype, KV cache changes, etc.). Notes are captured
          in the tester log for future reference.
        </p>
        <textarea
          rows={3}
          value={altConfigNotes}
          onChange={(e) => setAltConfigNotes(e.target.value)}
          placeholder='E.g., "Retry with attn_implementation=eager"'
          style={{ width: "100%", borderRadius: 8, padding: 10 }}
        />
        <button className="btn" style={{ marginTop: 8 }} onClick={handleSaveAltNotes}>
          Save Notes
        </button>
      </section>

      <section className="card" id="runner-logs">
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <h2>Capture Logs</h2>
          <button className="btn" onClick={refreshLogs}>
            Refresh Logs
          </button>
        </div>
        <div style={{ maxHeight: 260, overflowY: "auto" }}>{renderLogsTable()}</div>
      </section>

      <section className="card" id="runner-validate">
        <h2>Validate & Run Inference</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <button className="btn" onClick={handleValidateModel}>
              Validate Loaded Model
            </button>
            {runnerMessages.validate && (
              <p className="muted" style={{ marginTop: 6 }}>
                {runnerMessages.validate}
              </p>
            )}
          </div>
          <div>
            <label className="muted">Sample Prompt</label>
            <textarea
              rows={3}
              value={samplePrompt}
              onChange={(e) => setSamplePrompt(e.target.value)}
              style={{ width: "100%", borderRadius: 8, padding: 10 }}
            />
            <button className="btn" style={{ marginTop: 8 }} onClick={handleInferenceTest}>
              Run Inference Smoke Test
            </button>
            {runnerMessages.inference && (
              <p className="muted" style={{ marginTop: 6 }}>
                {runnerMessages.inference}
              </p>
            )}
          </div>
        </div>
      </section>

      <section className="card" id="runner-record">
        <h2>Log Manual Test Attempt</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Staged Model</label>
            <select
              value={form.model_name}
              onChange={(e) =>
                setForm((prev) => ({ ...prev, model_name: e.target.value }))
              }
              required
            >
              <option value="" disabled>
                Select a staged model
              </option>
              {localCopies.map((copy) => (
                <option key={copy.id} value={copy.model_name}>
                  {copy.model_name}
                </option>
              ))}
            </select>
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
    </>
  );

  const renderResultsTab = () => (
    <>
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <h2>Recent Runs</h2>
          <button className="btn" onClick={refreshTestRuns}>
            Refresh
          </button>
        </div>
        <div style={{ maxHeight: 320, overflowY: "auto" }}>{renderRunsTable()}</div>
      </section>
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <h2>Log Timeline</h2>
          <button className="btn" onClick={refreshLogs}>
            Refresh Logs
          </button>
        </div>
        <div style={{ maxHeight: 320, overflowY: "auto" }}>{renderLogsTable()}</div>
      </section>
    </>
  );

  const renderSettingsTab = () => (
    <section className="card">
      <h2>Tester Settings</h2>
      <p className="muted">
        Vector-Tester proxies all API traffic through `/api/llm/*`, ensuring that the UI works even
        when the FastAPI service is only reachable inside Docker. Review `.env` or docker-compose
        variables to adjust `LLM_API_BASE` and the local SQLite database path.
      </p>
      <ul className="muted">
        <li>LLM API Route: `/api/llm/status`, `/api/llm/models`</li>
        <li>Tester DB: `vector-tester/data/vector-tester.db` (mounted volume)</li>
        <li>Commit Template: `.gitmessage` (see `.codex` for context)</li>
      </ul>
    </section>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case "registry":
        return renderRegistryTab();
      case "runner":
        return renderRunnerTab();
      case "results":
        return renderResultsTab();
      case "settings":
        return renderSettingsTab();
      default:
        return renderDashboardTab();
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

      <div className="top-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="app-shell">
        <aside className="sidebar">
          <h3>{tabs.find((t) => t.id === activeTab)?.label}</h3>
          <ul>
            {sidebarSections[activeTab].map((item) => (
              <li key={item.label}>
                {item.anchor ? (
                  <a href={`#${item.anchor}`}>• {item.label}</a>
                ) : (
                  <>• {item.label}</>
                )}
              </li>
            ))}
          </ul>
        </aside>
        <div className="tab-content">{renderTabContent()}</div>
      </div>
    </div>
  );
}
