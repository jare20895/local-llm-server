import Dashboard from "@/components/Dashboard";
import { getDashboardData } from "@/lib/server-data";

export default async function Page() {
  const data = await getDashboardData();

  return (
    <Dashboard
      initialRuns={data.runs}
      initialLogs={data.logs}
      initialStatus={data.status}
      initialModels={data.models}
      initialLocalCopies={data.localCopies}
      initialServerEnvs={data.serverEnvs}
      initialProfiles={data.profiles}
      initialSwagger={data.swagger}
    />
  );
}
