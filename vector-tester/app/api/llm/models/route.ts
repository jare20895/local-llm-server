import { NextResponse } from "next/server";
import { fetchModels } from "@/lib/llm";

export async function GET() {
  const models = await fetchModels();
  return NextResponse.json({ models });
}
