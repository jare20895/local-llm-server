import { getRecentLogs, getRecentRuns, getTestModels } from "@/lib/db";
import { fetchModels, fetchStatus } from "@/lib/llm";

export async function getDashboardData() {
  const [runs, logs, status, models, localCopies] = await Promise.all([
    getRecentRuns(),
    getRecentLogs(),
    fetchStatus(),
    fetchModels(),
    getTestModels(),
  ]);

  return {
    runs,
    logs,
    status,
    models,
    localCopies,
  };
}
