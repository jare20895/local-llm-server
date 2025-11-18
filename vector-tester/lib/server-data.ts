import {
  getRecentLogs,
  getRecentRuns,
  getTestModels,
  getServerEnvironments,
  getTestProfiles,
  getSwaggerEndpoints,
  countTestSteps,
} from "@/lib/db";
import { fetchModels, fetchStatus } from "@/lib/llm";

export async function getDashboardData() {
  const [
    runs,
    logs,
    status,
    models,
    localCopies,
    serverEnvs,
    profiles,
    swagger,
  ] = await Promise.all([
    getRecentRuns(),
    getRecentLogs(),
    fetchStatus(),
    fetchModels(),
    getTestModels(),
    getServerEnvironments(),
    getTestProfiles(),
    getSwaggerEndpoints(),
  ]);
  const stepCount = countTestSteps();

  return {
    runs,
    logs,
    status,
    models,
    localCopies,
    serverEnvs,
    profiles,
    swagger,
    stepCount,
  };
}
