import { NextResponse } from "next/server";
import { forwardLlmRequest } from "@/lib/llmProxy";

export async function POST(request: Request) {
  const body = await request.json();
  const result = await forwardLlmRequest("/api/orchestrate/load", {
    method: "POST",
    body: JSON.stringify(body),
  });

  return NextResponse.json(result.data, { status: result.status });
}
