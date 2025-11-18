import { NextResponse } from "next/server";
import { getApiBase, fetchStatus } from "@/lib/llm";

export async function GET() {
  try {
    const res = await fetch(`${getApiBase()}/health`, {
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.text();
      return NextResponse.json({
        status: "healthy",
        detail: data || "llm-server responded to /health",
      });
    }
  } catch (error) {
    console.warn("Health check /health failed, falling back to status", error);
  }

  const status = await fetchStatus();
  if (!status) {
    return NextResponse.json(
      { status: "unreachable", detail: "Unable to reach llm-server" },
      { status: 503 }
    );
  }

  return NextResponse.json({
    status: status.loaded_model ? "ready" : "idle",
    detail: `Loaded model: ${status.loaded_model ?? "none"}`,
    payload: status,
  });
}
