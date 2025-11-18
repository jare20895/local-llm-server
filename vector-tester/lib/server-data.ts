import {
  getRecentLogs,
  getRecentRuns,
  getTestModels,
  getServerEnvironments,
  getTestProfiles,
  getStepsForProfile,
  getSwaggerEndpoints,
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

  return {
    runs,
    logs,
    status,
    models,
    localCopies,
    serverEnvs,
    profiles,
    swagger,
  };
}
