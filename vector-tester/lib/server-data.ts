import { getRecentLogs, getRecentRuns } from "@/lib/db";
import { fetchModels, fetchStatus } from "@/lib/llm";

export async function getDashboardData() {
  const [runs, logs, status, models] = await Promise.all([
    getRecentRuns(),
    getRecentLogs(),
    fetchStatus(),
    fetchModels(),
  ]);

  return {
    runs,
    logs,
    status,
    models,
  };
}
