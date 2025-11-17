export type LlmStatus = {
  loaded_model: string | null;
  loaded_model_id: number | null;
  gpu_available: boolean;
  gpu_memory_allocated_mb: number | null;
  gpu_memory_reserved_mb: number | null;
  performance_logging: boolean;
};

export type ModelSummary = {
  id: number;
  model_name: string;
  cache_location: string;
  compatibility_status: string | null;
  total_inferences: number;
};

const DEFAULT_BASE = "http://llm-server:8000";

export async function fetchStatus(base?: string): Promise<LlmStatus | null> {
  try {
    const res = await fetch(`${base ?? getApiBase()}/api/status`, {
      cache: "no-store",
    });
    if (!res.ok) {
      return null;
    }
    return (await res.json()) as LlmStatus;
  } catch (error) {
    console.warn("Failed to fetch LLM status", error);
    return null;
  }
}

export async function fetchModels(base?: string): Promise<ModelSummary[]> {
  try {
    const res = await fetch(`${base ?? getApiBase()}/api/models`, {
      cache: "no-store",
    });
    if (!res.ok) {
      return [];
    }
    const data = (await res.json()) as ModelSummary[];
    return data;
  } catch (error) {
    console.warn("Failed to fetch models", error);
    return [];
  }
}

export function getApiBase(): string {
  return process.env.LLM_API_BASE ?? DEFAULT_BASE;
}
