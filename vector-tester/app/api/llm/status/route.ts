import "server-only";
import { NextResponse } from "next/server";
import { fetchStatus } from "@/lib/llm";

export async function GET() {
  const status = await fetchStatus();
  if (!status) {
    return NextResponse.json(
      { error: "Unable to reach llm-server" },
      { status: 503 }
    );
  }
  return NextResponse.json(status);
}
