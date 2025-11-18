import { NextResponse } from "next/server";
import { forwardLlmRequest } from "@/lib/llmProxy";

export async function GET() {
  const result = await forwardLlmRequest("/api/orchestrate/validate");
  return NextResponse.json(result.data, { status: result.status });
}
