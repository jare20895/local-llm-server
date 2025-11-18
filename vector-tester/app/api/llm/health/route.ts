import { NextResponse } from "next/server";
import { forwardLlmRequest } from "@/lib/llmProxy";
import { fetchStatus } from "@/lib/llm";

export async function GET() {
  const result = await forwardLlmRequest("/health");
  if (result.ok) {
    return NextResponse.json({
      status: "healthy",
      detail: result.data || "llm-server responded to /health",
    });
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
