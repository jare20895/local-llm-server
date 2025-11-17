import Dashboard from "@/components/Dashboard";
import { getDashboardData } from "@/lib/server-data";
import { getApiBase } from "@/lib/llm";

export default async function Page() {
  const data = await getDashboardData();
  const apiBase = getApiBase();

  return (
    <Dashboard
      initialRuns={data.runs}
      initialLogs={data.logs}
      initialStatus={data.status}
      initialModels={data.models}
      apiBase={apiBase}
    />
  );
}
