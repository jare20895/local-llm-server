import { NextResponse } from "next/server";
import { z } from "zod";
import {
  getRecentRuns,
  insertTestRun,
  updateRunStatus,
  getRunById,
} from "@/lib/db";

const createSchema = z.object({
  model_name: z.string().min(2),
  scenario: z.string().optional(),
  notes: z.string().optional(),
});

const updateSchema = z.object({
  id: z.number().int().positive(),
  status: z.string().optional(),
  load_duration_ms: z.number().nonnegative().optional(),
  notes: z.string().optional(),
});

export async function GET() {
  const runs = getRecentRuns();
  return NextResponse.json({ runs });
}

export async function POST(request: Request) {
  const body = await request.json();
  const result = createSchema.safeParse(body);
  if (!result.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: result.error.flatten() },
      { status: 400 }
    );
  }

  const run = insertTestRun(result.data);
  return NextResponse.json({ run }, { status: 201 });
}

export async function PATCH(request: Request) {
  const body = await request.json();
  const result = updateSchema.safeParse(body);
  if (!result.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: result.error.flatten() },
      { status: 400 }
    );
  }

  const existing = getRunById(result.data.id);
  if (!existing) {
    return NextResponse.json(
      { error: `Test run ${result.data.id} not found` },
      { status: 404 }
    );
  }

  const updated = updateRunStatus(result.data);
  return NextResponse.json({ run: updated });
}
