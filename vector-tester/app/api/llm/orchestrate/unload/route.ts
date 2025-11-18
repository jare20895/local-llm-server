import "server-only";
import { NextResponse } from "next/server";
import { forwardLlmRequest } from "@/lib/llmProxy";

export async function POST() {
  const result = await forwardLlmRequest("/api/orchestrate/unload", {
    method: "POST",
    body: JSON.stringify({}),
  });

  return NextResponse.json(result.data, { status: result.status });
}
