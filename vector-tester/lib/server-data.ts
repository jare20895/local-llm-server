import {
  getRecentLogs,
  getRecentRuns,
  getTestModels,
  getServerEnvironments,
  getTestProfiles,
  getSwaggerEndpoints,
  countTestSteps,
  getHuggingfaceMetaTags,
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
    metadata,
  ] = await Promise.all([
    getRecentRuns(),
    getRecentLogs(),
    fetchStatus(),
    fetchModels(),
    getTestModels(),
    getServerEnvironments(),
    getTestProfiles(),
    getSwaggerEndpoints(),
    getHuggingfaceMetaTags(),
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
    metadata,
  };
}
