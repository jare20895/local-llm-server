import { NextResponse } from "next/server";
import { z } from "zod";
import { getRecentLogs, insertLogEvent } from "@/lib/db";

const schema = z.object({
  run_id: z.number().int().positive().optional(),
  source: z.string().min(2),
  level: z.enum(["debug", "info", "warn", "error"]).optional(),
  message: z.string().min(1),
});

export async function GET() {
  const logs = getRecentLogs();
  return NextResponse.json({ logs });
}

export async function POST(request: Request) {
  const body = await request.json();
  const result = schema.safeParse(body);
  if (!result.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: result.error.flatten() },
      { status: 400 }
    );
  }

  const log = insertLogEvent(result.data);
  return NextResponse.json({ log }, { status: 201 });
}
