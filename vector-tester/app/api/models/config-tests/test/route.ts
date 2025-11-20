import { NextResponse } from "next/server";
import { z } from "zod";
import { updateConfigTestMeta } from "@/lib/db";

const schema = z.object({
  id: z.number().int().positive(),
  name: z.string().optional(),
  description: z.string().nullable().optional(),
  load_status: z.string().optional(),
  load_notes: z.string().nullable().optional(),
  last_tested_at: z.string().nullable().optional(),
});

export async function PATCH(request: Request) {
  const payload = await request.json();
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }
  try {
    const updated = updateConfigTestMeta(parsed.data);
    return NextResponse.json({ test: updated });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to update config test",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
