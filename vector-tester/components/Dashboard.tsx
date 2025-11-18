"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  TestRun,
  LogEvent,
  TestModelCopy,
  ServerEnvironment,
  TestProfile,
  SwaggerEndpoint,
  TestStep,
} from "@/lib/db";
import type { LlmStatus, ModelSummary } from "@/lib/llm";

type SidebarItem = { label: string; anchor?: string; section?: string };

type Props = {
  initialRuns: TestRun[];
  initialLogs: LogEvent[];
  initialStatus: LlmStatus | null;
  initialModels: ModelSummary[];
  initialLocalCopies: TestModelCopy[];
  initialServerEnvs: ServerEnvironment[];
  initialProfiles: TestProfile[];
  initialSwagger: SwaggerEndpoint[];
  initialStepCount: number;
};

export default function Dashboard({
  initialRuns,
  initialLogs,
  initialStatus,
  initialModels,
  initialLocalCopies,
  initialServerEnvs,
  initialProfiles,
  initialSwagger,
  initialStepCount,
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
  const [serverEnvs, setServerEnvs] =
    useState<ServerEnvironment[]>(initialServerEnvs);
  const [profiles, setProfiles] = useState<TestProfile[]>(initialProfiles);
  const [swagger, setSwagger] = useState<SwaggerEndpoint[]>(initialSwagger);
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
  const [envForm, setEnvForm] = useState({
    name: "",
    hostname: "",
    ip_address: "",
    gpu_model: "",
    gpu_vram_gb: "",
    cpu_model: "",
    os_version: "",
    wsl_version: "",
    rocm_version: "",
    notes: "",
  });
  const [profileForm, setProfileForm] = useState({
    name: "",
    description: "",
    model_test_id: "",
    server_environment_id: "",
    default_prompt: "",
    max_tokens: "",
    temperature: "",
    top_p: "",
  });
  const [stepForm, setStepForm] = useState({
    profile_id: "",
    step_order: "0",
    step_name: "",
    api_method: "POST",
    api_path: "",
    request_body: "",
    expected_status: "200",
    expected_contains: "",
    pass_rule: "",
    notes: "",
  });
  const [swaggerForm, setSwaggerForm] = useState({
    method: "POST",
    path: "",
    summary: "",
    description: "",
    request_schema: "",
    response_schema: "",
  });
  const [stepListProfile, setStepListProfile] = useState("");
  const [stepList, setStepList] = useState<TestStep[]>([]);
  const [modal, setModal] = useState<{
    type: "env" | "profile" | "step" | "swagger";
    data?: any;
  } | null>(null);
  const [editEnvForm, setEditEnvForm] = useState({
    id: 0,
    name: "",
    hostname: "",
    ip_address: "",
    gpu_model: "",
    gpu_vram_gb: "",
    cpu_model: "",
    os_version: "",
    wsl_version: "",
    rocm_version: "",
    notes: "",
  });
  const [editProfileForm, setEditProfileForm] = useState({
    id: 0,
    name: "",
    description: "",
    model_test_id: "",
    server_environment_id: "",
    default_prompt: "",
    max_tokens: "",
    temperature: "",
    top_p: "",
  });
  const [editStepForm, setEditStepForm] = useState({
    id: 0,
    profile_id: "",
    step_order: "",
    step_name: "",
    api_method: "",
    api_path: "",
    request_body: "",
    expected_status: "",
    expected_contains: "",
    pass_rule: "",
    notes: "",
  });
  const [editSwaggerForm, setEditSwaggerForm] = useState({
    id: 0,
    method: "",
    path: "",
    summary: "",
    description: "",
    request_schema: "",
    response_schema: "",
  });

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

  const fetchStepsForProfile = useCallback(async (profileId: string) => {
    if (!profileId) {
      setStepList([]);
      return;
    }
    try {
      const res = await fetch(
        `/api/test-config/test-steps?profile_id=${profileId}`
      );
      if (res.ok) {
        const data = await res.json();
        setStepList(data.steps || []);
      }
    } catch (error) {
      console.warn("Failed to load test steps", error);
    }
  }, []);

  useEffect(() => {
    fetchStepsForProfile(stepListProfile);
  }, [stepListProfile, fetchStepsForProfile]);

  useEffect(() => {
    if (
      stepListProfile &&
      !profiles.find((profile) => String(profile.id) === stepListProfile)
    ) {
      setStepListProfile("");
    }
  }, [profiles, stepListProfile]);

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

  const handleEnvSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const payload = {
        ...envForm,
        gpu_vram_gb: envForm.gpu_vram_gb
          ? Number(envForm.gpu_vram_gb)
          : undefined,
      };
      const res = await fetch("/api/test-config/server-environments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setServerEnvs((prev) => {
          const filtered = prev.filter((e) => e.name !== data.environment.name);
          return [data.environment, ...filtered];
        });
        setEnvForm({
          name: "",
          hostname: "",
          ip_address: "",
          gpu_model: "",
          gpu_vram_gb: "",
          cpu_model: "",
          os_version: "",
          wsl_version: "",
          rocm_version: "",
          notes: "",
        });
      } else {
        alert(`Failed to save environment: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Environment error: ${(error as Error).message}`);
    }
  };

  const handleProfileSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const payload = {
        name: profileForm.name,
        description: profileForm.description || undefined,
        model_test_id: profileForm.model_test_id
          ? Number(profileForm.model_test_id)
          : undefined,
        server_environment_id: profileForm.server_environment_id
          ? Number(profileForm.server_environment_id)
          : undefined,
        default_prompt: profileForm.default_prompt || undefined,
        max_tokens: profileForm.max_tokens
          ? Number(profileForm.max_tokens)
          : undefined,
        temperature: profileForm.temperature
          ? Number(profileForm.temperature)
          : undefined,
        top_p: profileForm.top_p ? Number(profileForm.top_p) : undefined,
      };
      const res = await fetch("/api/test-config/test-profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setProfiles((prev) => [data.profile, ...prev]);
        setProfileForm({
          name: "",
          description: "",
          model_test_id: "",
          server_environment_id: "",
          default_prompt: "",
          max_tokens: "",
          temperature: "",
          top_p: "",
        });
      } else {
        alert(`Failed to save profile: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Profile error: ${(error as Error).message}`);
    }
  };

  const handleStepSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!stepForm.profile_id) {
      alert("Select a profile first.");
      return;
    }
    try {
      const payload = {
        profile_id: Number(stepForm.profile_id),
        step_order: Number(stepForm.step_order),
        step_name: stepForm.step_name,
        api_method: stepForm.api_method,
        api_path: stepForm.api_path,
        request_body: stepForm.request_body || undefined,
        expected_status: stepForm.expected_status
          ? Number(stepForm.expected_status)
          : undefined,
        expected_contains: stepForm.expected_contains || undefined,
        pass_rule: stepForm.pass_rule || undefined,
        notes: stepForm.notes || undefined,
      };
      const res = await fetch("/api/test-config/test-steps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setStepForm((prev) => ({
          ...prev,
          step_name: "",
          request_body: "",
          expected_contains: "",
          pass_rule: "",
          notes: "",
          step_order: String(Number(prev.step_order) + 1),
        }));
        setStepCount((prev) => prev + 1);
        fetchStepsForProfile(String(payload.profile_id));
      } else {
        alert(`Failed to save step: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Step error: ${(error as Error).message}`);
    }
  };

  const handleSwaggerSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const res = await fetch("/api/test-config/swagger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(swaggerForm),
      });
      const data = await res.json();
      if (res.ok) {
        setSwagger((prev) => {
          const filtered = prev.filter(
            (ep) =>
              !(
                ep.method === data.endpoint.method &&
                ep.path === data.endpoint.path
              )
          );
          return [data.endpoint, ...filtered];
        });
        setSwaggerForm({
          method: "POST",
          path: "",
          summary: "",
          description: "",
          request_schema: "",
          response_schema: "",
        });
      } else {
        alert(`Swagger sync failed: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Swagger error: ${(error as Error).message}`);
    }
  };

  const closeModal = () => setModal(null);

  const openEnvModal = (env: ServerEnvironment) => {
    setEditEnvForm({
      id: env.id,
      name: env.name,
      hostname: env.hostname ?? "",
      ip_address: env.ip_address ?? "",
      gpu_model: env.gpu_model ?? "",
      gpu_vram_gb: env.gpu_vram_gb ? String(env.gpu_vram_gb) : "",
      cpu_model: env.cpu_model ?? "",
      os_version: env.os_version ?? "",
      wsl_version: env.wsl_version ?? "",
      rocm_version: env.rocm_version ?? "",
      notes: env.notes ?? "",
    });
    setModal({ type: "env", data: env });
  };

  const handleEnvUpdate = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const payload = {
        ...editEnvForm,
        gpu_vram_gb: editEnvForm.gpu_vram_gb
          ? Number(editEnvForm.gpu_vram_gb)
          : undefined,
      };
      const res = await fetch("/api/test-config/server-environments", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setServerEnvs((prev) => {
          const filtered = prev.filter((e) => e.id !== data.environment.id);
          return [data.environment, ...filtered];
        });
        closeModal();
      } else {
        alert(`Update failed: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Update error: ${(error as Error).message}`);
    }
  };

  const handleEnvDelete = async () => {
    if (!editEnvForm.id) return;
    if (!confirm("Delete this environment?")) return;
    await fetch("/api/test-config/server-environments", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: editEnvForm.id }),
    });
    setServerEnvs((prev) => prev.filter((env) => env.id !== editEnvForm.id));
    closeModal();
  };

  const openProfileModal = (profile: TestProfile) => {
    setEditProfileForm({
      id: profile.id,
      name: profile.name,
      description: profile.description ?? "",
      model_test_id: profile.model_test_id
        ? String(profile.model_test_id)
        : "",
      server_environment_id: profile.server_environment_id
        ? String(profile.server_environment_id)
        : "",
      default_prompt: profile.default_prompt ?? "",
      max_tokens: profile.max_tokens ? String(profile.max_tokens) : "",
      temperature: profile.temperature ? String(profile.temperature) : "",
      top_p: profile.top_p ? String(profile.top_p) : "",
    });
    setModal({ type: "profile", data: profile });
  };

  const handleProfileUpdate = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const payload = {
        id: editProfileForm.id,
        name: editProfileForm.name,
        description: editProfileForm.description || undefined,
        model_test_id: editProfileForm.model_test_id
          ? Number(editProfileForm.model_test_id)
          : undefined,
        server_environment_id: editProfileForm.server_environment_id
          ? Number(editProfileForm.server_environment_id)
          : undefined,
        default_prompt: editProfileForm.default_prompt || undefined,
        max_tokens: editProfileForm.max_tokens
          ? Number(editProfileForm.max_tokens)
          : undefined,
        temperature: editProfileForm.temperature
          ? Number(editProfileForm.temperature)
          : undefined,
        top_p: editProfileForm.top_p
          ? Number(editProfileForm.top_p)
          : undefined,
      };
      const res = await fetch("/api/test-config/test-profiles", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setProfiles((prev) => {
          const filtered = prev.filter((p) => p.id !== data.profile.id);
          return [data.profile, ...filtered];
        });
        closeModal();
      } else {
        alert(`Update failed: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Update error: ${(error as Error).message}`);
    }
  };

  const handleProfileDelete = async () => {
    if (!editProfileForm.id) return;
    if (!confirm("Delete this profile?")) return;
    await fetch("/api/test-config/test-profiles", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: editProfileForm.id }),
    });
    setProfiles((prev) => prev.filter((profile) => profile.id !== editProfileForm.id));
    closeModal();
  };

  const openStepModal = (step: TestStep) => {
    setEditStepForm({
      id: step.id,
      profile_id: String(step.profile_id),
      step_order: String(step.step_order),
      step_name: step.step_name,
      api_method: step.api_method,
      api_path: step.api_path,
      request_body: step.request_body ?? "",
      expected_status: step.expected_status
        ? String(step.expected_status)
        : "",
      expected_contains: step.expected_contains ?? "",
      pass_rule: step.pass_rule ?? "",
      notes: step.notes ?? "",
    });
    setModal({ type: "step", data: step });
  };

  const handleStepUpdate = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const payload = {
        id: editStepForm.id,
        step_order: Number(editStepForm.step_order),
        step_name: editStepForm.step_name,
        api_method: editStepForm.api_method,
        api_path: editStepForm.api_path,
        request_body: editStepForm.request_body || undefined,
        expected_status: editStepForm.expected_status
          ? Number(editStepForm.expected_status)
          : undefined,
        expected_contains: editStepForm.expected_contains || undefined,
        pass_rule: editStepForm.pass_rule || undefined,
        notes: editStepForm.notes || undefined,
      };
      const res = await fetch("/api/test-config/test-steps", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setStepList((prev) => {
          const filtered = prev.filter((s) => s.id !== data.step.id);
          return [...filtered, data.step].sort(
            (a, b) => a.step_order - b.step_order
          );
        });
        closeModal();
      } else {
        alert(`Update failed: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Update error: ${(error as Error).message}`);
    }
  };

  const handleStepDelete = async () => {
    if (!editStepForm.id) return;
    if (!confirm("Delete this step?")) return;
    await fetch("/api/test-config/test-steps", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: editStepForm.id }),
    });
    setStepList((prev) => prev.filter((step) => step.id !== editStepForm.id));
    setStepCount((prev) => Math.max(0, prev - 1));
    closeModal();
  };

  const openSwaggerModal = (entry: SwaggerEndpoint) => {
    setEditSwaggerForm({
      id: entry.id,
      method: entry.method,
      path: entry.path,
      summary: entry.summary ?? "",
      description: entry.description ?? "",
      request_schema: entry.request_schema ?? "",
      response_schema: entry.response_schema ?? "",
    });
    setModal({ type: "swagger", data: entry });
  };

  const handleSwaggerUpdate = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const res = await fetch("/api/test-config/swagger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editSwaggerForm),
      });
      const data = await res.json();
      if (res.ok) {
        setSwagger((prev) => {
          const filtered = prev.filter((e) => e.id !== editSwaggerForm.id);
          return [data.endpoint, ...filtered];
        });
        closeModal();
      } else {
        alert(`Update failed: ${responseMessage(data)}`);
      }
    } catch (error) {
      alert(`Update error: ${(error as Error).message}`);
    }
  };

  const handleSwaggerDelete = async () => {
    if (!editSwaggerForm.id) return;
    if (!confirm("Delete this endpoint?")) return;
    await fetch("/api/test-config/swagger", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: editSwaggerForm.id }),
    });
    setSwagger((prev) => prev.filter((ep) => ep.id !== editSwaggerForm.id));
    closeModal();
  };

  const tabs = [
    { id: "dashboard", label: "Dashboard", summary: "High-level overview" },
    { id: "registry", label: "Model Registry", summary: "Manage models" },
    { id: "runner", label: "Test Runner", summary: "Execute new tests" },
    { id: "config", label: "Test Config", summary: "Manage environments & profiles" },
    { id: "automation", label: "Test Automation", summary: "Configure workflows" },
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
    config: [
      { label: "Overview", section: "overview" },
      { label: "Server Environments", section: "envs" },
      { label: "Test Profiles", section: "profiles" },
      { label: "Test Steps", section: "steps" },
      { label: "Swagger Endpoints", section: "swagger" },
    ],
    automation: [
      { label: "Select Profile", anchor: "auto-select" },
      { label: "Step Overview", anchor: "auto-steps" },
      { label: "Run Sequence", anchor: "auto-run" },
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

  const renderConfigOverview = () => (
    <div className="grid grid-2">
      <section className="card">
        <h3>Server Environments</h3>
        <p className="muted">{serverEnvs.length} tracked</p>
        <button className="btn" onClick={() => setConfigSection("envs")}>
          Manage
        </button>
      </section>
      <section className="card">
        <h3>Test Profiles</h3>
        <p className="muted">{profiles.length} defined</p>
        <button className="btn" onClick={() => setConfigSection("profiles")}>
          Manage
        </button>
      </section>
      <section className="card">
        <h3>Test Steps</h3>
        <p className="muted">{stepCount} total steps</p>
        <button className="btn" onClick={() => setConfigSection("steps")}>
          Manage
        </button>
      </section>
      <section className="card">
        <h3>Swagger Endpoints</h3>
        <p className="muted">{swagger.length} cataloged</p>
        <button className="btn" onClick={() => setConfigSection("swagger")}>
          Manage
        </button>
      </section>
    </div>
  );

  const renderEnvSection = () => (
    <section className="card" id="config-envs">
        <h2>Server Environments</h2>
        <form onSubmit={handleEnvSubmit}>
          <div className="form-group">
            <label>Name</label>
            <input
              value={envForm.name}
              onChange={(e) =>
                setEnvForm((prev) => ({ ...prev, name: e.target.value }))
              }
              required
            />
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(2,1fr)" }}>
            <div className="form-group">
              <label>Hostname</label>
              <input
                value={envForm.hostname}
                onChange={(e) =>
                  setEnvForm((prev) => ({ ...prev, hostname: e.target.value }))
                }
              />
            </div>
            <div className="form-group">
              <label>IP Address</label>
              <input
                value={envForm.ip_address}
                onChange={(e) =>
                  setEnvForm((prev) => ({ ...prev, ip_address: e.target.value }))
                }
              />
            </div>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(2,1fr)" }}>
            <div className="form-group">
              <label>GPU Model</label>
              <input
                value={envForm.gpu_model}
                onChange={(e) =>
                  setEnvForm((prev) => ({ ...prev, gpu_model: e.target.value }))
                }
              />
            </div>
            <div className="form-group">
              <label>GPU VRAM (GB)</label>
              <input
                value={envForm.gpu_vram_gb}
                onChange={(e) =>
                  setEnvForm((prev) => ({
                    ...prev,
                    gpu_vram_gb: e.target.value,
                  }))
                }
              />
            </div>
          </div>
          <div className="form-group">
            <label>CPU Model</label>
            <input
              value={envForm.cpu_model}
              onChange={(e) =>
                setEnvForm((prev) => ({ ...prev, cpu_model: e.target.value }))
              }
            />
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
            <div className="form-group">
              <label>OS</label>
              <input
                value={envForm.os_version}
                onChange={(e) =>
                  setEnvForm((prev) => ({ ...prev, os_version: e.target.value }))
                }
              />
            </div>
            <div className="form-group">
              <label>WSL</label>
              <input
                value={envForm.wsl_version}
                onChange={(e) =>
                  setEnvForm((prev) => ({ ...prev, wsl_version: e.target.value }))
                }
              />
            </div>
            <div className="form-group">
              <label>ROCm</label>
              <input
                value={envForm.rocm_version}
                onChange={(e) =>
                  setEnvForm((prev) => ({
                    ...prev,
                    rocm_version: e.target.value,
                  }))
                }
              />
            </div>
          </div>
          <div className="form-group">
            <label>Notes</label>
            <textarea
              rows={2}
              value={envForm.notes}
              onChange={(e) =>
                setEnvForm((prev) => ({ ...prev, notes: e.target.value }))
              }
            />
          </div>
          <button className="btn" type="submit">
            Save Environment
          </button>
        </form>
        {serverEnvs.length > 0 && (
          <div style={{ marginTop: 12, overflowX: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>GPU</th>
                  <th>IP</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {serverEnvs.map((env) => (
                  <tr key={env.id}>
                    <td>{env.name}</td>
                    <td>{env.gpu_model || "Unknown"}</td>
                    <td>{env.ip_address || "—"}</td>
                    <td>
                      <button className="btn" onClick={() => openEnvModal(env)}>
                        Edit
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
  );

  const renderProfileSection = () => (
      <section className="card" id="config-profiles">
        <h2>Test Profiles</h2>
        <form onSubmit={handleProfileSubmit}>
          <div className="form-group">
            <label>Name</label>
            <input
              value={profileForm.name}
              onChange={(e) =>
                setProfileForm((prev) => ({ ...prev, name: e.target.value }))
              }
              required
            />
          </div>
          <div className="form-group">
            <label>Description</label>
            <input
              value={profileForm.description}
              onChange={(e) =>
                setProfileForm((prev) => ({
                  ...prev,
                  description: e.target.value,
                }))
              }
            />
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(2,1fr)" }}>
            <div className="form-group">
              <label>Staged Model</label>
              <select
                value={profileForm.model_test_id}
                onChange={(e) =>
                  setProfileForm((prev) => ({
                    ...prev,
                    model_test_id: e.target.value,
                  }))
                }
              >
                <option value="">Select model</option>
                {localCopies.map((copy) => (
                  <option key={copy.id} value={copy.id}>
                    {copy.model_name}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>Server Environment</label>
              <select
                value={profileForm.server_environment_id}
                onChange={(e) =>
                  setProfileForm((prev) => ({
                    ...prev,
                    server_environment_id: e.target.value,
                  }))
                }
              >
                <option value="">Select environment</option>
                {serverEnvs.map((env) => (
                  <option key={env.id} value={env.id}>
                    {env.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
            <div className="form-group">
              <label>Prompt</label>
              <input
                value={profileForm.default_prompt}
                onChange={(e) =>
                  setProfileForm((prev) => ({
                    ...prev,
                    default_prompt: e.target.value,
                  }))
                }
              />
            </div>
            <div className="form-group">
              <label>Max Tokens</label>
              <input
                value={profileForm.max_tokens}
                onChange={(e) =>
                  setProfileForm((prev) => ({
                    ...prev,
                    max_tokens: e.target.value,
                  }))
                }
              />
            </div>
            <div className="form-group">
              <label>Temperature</label>
              <input
                value={profileForm.temperature}
                onChange={(e) =>
                  setProfileForm((prev) => ({
                    ...prev,
                    temperature: e.target.value,
                  }))
                }
              />
            </div>
          </div>
          <div className="form-group">
            <label>Top-p</label>
            <input
              value={profileForm.top_p}
              onChange={(e) =>
                setProfileForm((prev) => ({ ...prev, top_p: e.target.value }))
              }
            />
          </div>
          <button className="btn" type="submit">
            Save Profile
          </button>
        </form>
        {profiles.length > 0 && (
          <div style={{ marginTop: 12, overflowX: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Model</th>
                  <th>Environment</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {profiles.map((profile) => (
                  <tr key={profile.id}>
                    <td>{profile.name}</td>
                    <td>{profile.model_test_id ?? "—"}</td>
                    <td>{profile.server_environment_id ?? "—"}</td>
                    <td>
                      <button className="btn" onClick={() => openProfileModal(profile)}>
                        Edit
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
  );

  const renderStepsSection = () => (
      <section className="card" id="config-steps">
        <h2>Test Steps</h2>
        <form onSubmit={handleStepSubmit}>
          <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
            <div className="form-group">
              <label>Profile</label>
              <select
                value={stepForm.profile_id}
                onChange={(e) =>
                  {
                    setStepForm((prev) => ({ ...prev, profile_id: e.target.value }));
                    setStepListProfile(e.target.value);
                  }
                }
                required
              >
                <option value="">Select profile</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>Step Order</label>
              <input
                value={stepForm.step_order}
                onChange={(e) =>
                  setStepForm((prev) => ({
                    ...prev,
                    step_order: e.target.value,
                  }))
                }
                required
              />
            </div>
            <div className="form-group">
              <label>HTTP Method</label>
              <input
                value={stepForm.api_method}
                onChange={(e) =>
                  setStepForm((prev) => ({
                    ...prev,
                    api_method: e.target.value,
                  }))
                }
                required
              />
            </div>
          </div>
          <div className="form-group">
            <label>API Path</label>
            <input
              value={stepForm.api_path}
              onChange={(e) =>
                setStepForm((prev) => ({ ...prev, api_path: e.target.value }))
              }
              required
            />
          </div>
          <div className="form-group">
            <label>Step Name</label>
            <input
              value={stepForm.step_name}
              onChange={(e) =>
                setStepForm((prev) => ({ ...prev, step_name: e.target.value }))
              }
              required
            />
          </div>
          <div className="form-group">
            <label>Request Body</label>
            <textarea
              rows={3}
              value={stepForm.request_body}
              onChange={(e) =>
                setStepForm((prev) => ({
                  ...prev,
                  request_body: e.target.value,
                }))
              }
            />
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
            <div className="form-group">
              <label>Expected Status</label>
              <input
                value={stepForm.expected_status}
                onChange={(e) =>
                  setStepForm((prev) => ({
                    ...prev,
                    expected_status: e.target.value,
                  }))
                }
              />
            </div>
            <div className="form-group">
              <label>Expected Contains</label>
              <input
                value={stepForm.expected_contains}
                onChange={(e) =>
                  setStepForm((prev) => ({
                    ...prev,
                    expected_contains: e.target.value,
                  }))
                }
              />
            </div>
            <div className="form-group">
              <label>Pass Rule</label>
              <input
                value={stepForm.pass_rule}
                onChange={(e) =>
                  setStepForm((prev) => ({
                    ...prev,
                    pass_rule: e.target.value,
                  }))
                }
              />
            </div>
          </div>
          <div className="form-group">
            <label>Notes</label>
            <textarea
              rows={2}
              value={stepForm.notes}
              onChange={(e) =>
                setStepForm((prev) => ({ ...prev, notes: e.target.value }))
              }
            />
          </div>
          <button className="btn" type="submit">
            Add Step
          </button>
        </form>
        <div style={{ marginTop: 12 }}>
          <label className="muted">View steps for profile</label>
          <select
            value={stepListProfile}
            onChange={(e) => setStepListProfile(e.target.value)}
            style={{ marginLeft: 8 }}
          >
            <option value="">Select profile</option>
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
              </option>
            ))}
          </select>
        </div>
        {stepListProfile && (
          <div style={{ marginTop: 12, overflowX: "auto" }}>
            {stepList.length === 0 ? (
              <p className="muted">No steps defined for this profile.</p>
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Name</th>
                    <th>API</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {stepList.map((step) => (
                    <tr key={step.id}>
                      <td>{step.step_order}</td>
                      <td>{step.step_name}</td>
                      <td>
                        {step.api_method} {step.api_path}
                      </td>
                      <td>
                        <button className="btn" onClick={() => openStepModal(step)}>
                          Edit
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </section>
  );

  const renderSwaggerSection = () => (
      <section className="card" id="config-swagger">
        <h2>Swagger Endpoint Catalog</h2>
        <form onSubmit={handleSwaggerSubmit}>
          <div className="grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
            <div className="form-group">
              <label>Method</label>
              <input
                value={swaggerForm.method}
                onChange={(e) =>
                  setSwaggerForm((prev) => ({ ...prev, method: e.target.value }))
                }
                required
              />
            </div>
            <div className="form-group">
              <label>Path</label>
              <input
                value={swaggerForm.path}
                onChange={(e) =>
                  setSwaggerForm((prev) => ({ ...prev, path: e.target.value }))
                }
                required
              />
            </div>
            <div className="form-group">
              <label>Summary</label>
              <input
                value={swaggerForm.summary}
                onChange={(e) =>
                  setSwaggerForm((prev) => ({
                    ...prev,
                    summary: e.target.value,
                  }))
                }
              />
            </div>
          </div>
          <div className="form-group">
            <label>Description</label>
            <textarea
              rows={2}
              value={swaggerForm.description}
              onChange={(e) =>
                setSwaggerForm((prev) => ({
                  ...prev,
                  description: e.target.value,
                }))
              }
            />
          </div>
          <div className="form-group">
            <label>Request Schema</label>
            <textarea
              rows={2}
              value={swaggerForm.request_schema}
              onChange={(e) =>
                setSwaggerForm((prev) => ({
                  ...prev,
                  request_schema: e.target.value,
                }))
              }
            />
          </div>
          <div className="form-group">
            <label>Response Schema</label>
            <textarea
              rows={2}
              value={swaggerForm.response_schema}
              onChange={(e) =>
                setSwaggerForm((prev) => ({
                  ...prev,
                  response_schema: e.target.value,
                }))
              }
            />
        </div>
        <button className="btn" type="submit">
          Save Endpoint
        </button>
        </form>
        {swagger.length > 0 && (
          <div style={{ marginTop: 12, maxHeight: 200, overflowY: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Path</th>
                  <th>Summary</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {swagger.map((ep) => (
                  <tr key={ep.id}>
                    <td>{ep.method}</td>
                    <td>{ep.path}</td>
                    <td>{ep.summary || "—"}</td>
                    <td>
                      <button className="btn" onClick={() => openSwaggerModal(ep)}>
                        Edit
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
  );

  const renderConfigTab = () => (
    <>
      {configSection === "overview" && renderConfigOverview()}
      {configSection === "envs" && renderEnvSection()}
      {configSection === "profiles" && renderProfileSection()}
      {configSection === "steps" && renderStepsSection()}
      {configSection === "swagger" && renderSwaggerSection()}
    </>
  );

  const renderAutomationTab = () => (
    <section className="card" id="auto-select">
      <h2>Test Automation (Preview)</h2>
      <p className="muted">
        Select a profile to view its configured steps. Future iterations will allow running the full
        sequence automatically.
      </p>
      <div className="form-group">
        <label>Profile</label>
        <select disabled>
          <option>Select profile</option>
        </select>
      </div>
      <div className="muted">
        Connect this tab to a scheduler or runbook to execute the API steps defined in Test Config.
      </div>
    </section>
  );

  const renderModal = () => {
    if (!modal) return null;
    if (modal.type === "env") {
      return (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3>Edit Server Environment</h3>
            <form onSubmit={handleEnvUpdate}>
              <div className="form-group">
                <label>Name</label>
                <input
                  value={editEnvForm.name}
                  onChange={(e) =>
                    setEditEnvForm((prev) => ({ ...prev, name: e.target.value }))
                  }
                  required
                />
              </div>
              <div className="form-group">
                <label>Hostname</label>
                <input
                  value={editEnvForm.hostname}
                  onChange={(e) =>
                    setEditEnvForm((prev) => ({ ...prev, hostname: e.target.value }))
                  }
                />
              </div>
              <div className="form-group">
                <label>IP Address</label>
                <input
                  value={editEnvForm.ip_address}
                  onChange={(e) =>
                    setEditEnvForm((prev) => ({ ...prev, ip_address: e.target.value }))
                  }
                />
              </div>
              <div className="form-group">
                <label>GPU Model</label>
                <input
                  value={editEnvForm.gpu_model}
                  onChange={(e) =>
                    setEditEnvForm((prev) => ({ ...prev, gpu_model: e.target.value }))
                  }
                />
              </div>
              <div className="form-group">
                <label>GPU VRAM (GB)</label>
                <input
                  value={editEnvForm.gpu_vram_gb}
                  onChange={(e) =>
                    setEditEnvForm((prev) => ({
                      ...prev,
                      gpu_vram_gb: e.target.value,
                    }))
                  }
                />
              </div>
              <div className="form-group">
                <label>Notes</label>
                <textarea
                  rows={2}
                  value={editEnvForm.notes}
                  onChange={(e) =>
                    setEditEnvForm((prev) => ({ ...prev, notes: e.target.value }))
                  }
                />
              </div>
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={handleEnvDelete}
                >
                  Delete
                </button>
                <button className="btn" type="submit">
                  Save
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={closeModal}
                >
                  Close
                </button>
              </div>
            </form>
          </div>
        </div>
      );
    }
    if (modal.type === "profile") {
      return (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3>Edit Test Profile</h3>
            <form onSubmit={handleProfileUpdate}>
              <div className="form-group">
                <label>Name</label>
                <input
                  value={editProfileForm.name}
                  onChange={(e) =>
                    setEditProfileForm((prev) => ({ ...prev, name: e.target.value }))
                  }
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <input
                  value={editProfileForm.description}
                  onChange={(e) =>
                    setEditProfileForm((prev) => ({
                      ...prev,
                      description: e.target.value,
                    }))
                  }
                />
              </div>
              <div className="form-group">
                <label>Default Prompt</label>
                <input
                  value={editProfileForm.default_prompt}
                  onChange={(e) =>
                    setEditProfileForm((prev) => ({
                      ...prev,
                      default_prompt: e.target.value,
                    }))
                  }
                />
              </div>
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={handleProfileDelete}
                >
                  Delete
                </button>
                <button className="btn" type="submit">
                  Save
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={closeModal}
                >
                  Close
                </button>
              </div>
            </form>
          </div>
        </div>
      );
    }
    if (modal.type === "step") {
      return (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3>Edit Test Step</h3>
            <form onSubmit={handleStepUpdate}>
              <div className="form-group">
                <label>Step Name</label>
                <input
                  value={editStepForm.step_name}
                  onChange={(e) =>
                    setEditStepForm((prev) => ({ ...prev, step_name: e.target.value }))
                  }
                  required
                />
              </div>
              <div className="form-group">
                <label>API Path</label>
                <input
                  value={editStepForm.api_path}
                  onChange={(e) =>
                    setEditStepForm((prev) => ({ ...prev, api_path: e.target.value }))
                  }
                  required
                />
              </div>
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={handleStepDelete}
                >
                  Delete
                </button>
                <button className="btn" type="submit">
                  Save
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={closeModal}
                >
                  Close
                </button>
              </div>
            </form>
          </div>
        </div>
      );
    }
    if (modal.type === "swagger") {
      return (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3>Edit Swagger Endpoint</h3>
            <form onSubmit={handleSwaggerUpdate}>
              <div className="form-group">
                <label>Method</label>
                <input
                  value={editSwaggerForm.method}
                  onChange={(e) =>
                    setEditSwaggerForm((prev) => ({ ...prev, method: e.target.value }))
                  }
                />
              </div>
              <div className="form-group">
                <label>Path</label>
                <input
                  value={editSwaggerForm.path}
                  onChange={(e) =>
                    setEditSwaggerForm((prev) => ({ ...prev, path: e.target.value }))
                  }
                />
              </div>
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={handleSwaggerDelete}
                >
                  Delete
                </button>
                <button className="btn" type="submit">
                  Save
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={closeModal}
                >
                  Close
                </button>
              </div>
            </form>
          </div>
        </div>
      );
    }
    return null;
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case "registry":
        return renderRegistryTab();
      case "runner":
        return renderRunnerTab();
      case "config":
        return renderConfigTab();
      case "automation":
        return renderAutomationTab();
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
                {activeTab === "config" && item.section ? (
                  <button
                    className="btn"
                    style={{
                      background: configSection === item.section ? "#2563eb" : "transparent",
                      color: configSection === item.section ? "#fff" : "var(--text-muted)",
                      border:
                        configSection === item.section
                          ? "1px solid rgba(255,255,255,0.2)"
                          : "1px solid transparent",
                      width: "100%",
                      justifyContent: "flex-start",
                    }}
                    onClick={() => setConfigSection(item.section!)}
                  >
                    {item.label}
                  </button>
                ) : item.anchor ? (
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
      {renderModal()}
    </div>
  );
}
  const [configSection, setConfigSection] = useState("overview");
  const [stepCount, setStepCount] = useState(initialStepCount);
