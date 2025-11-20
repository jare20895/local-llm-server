import { NextResponse } from "next/server";
import { z } from "zod";
import {
  addConfigTestEntry,
  deleteConfigTestEntry,
  updateConfigTestEntry,
} from "@/lib/db";

const updateSchema = z.object({
  id: z.number().int().positive(),
  active: z.boolean().optional(),
  inherit_default: z.boolean().optional(),
  value_text: z.string().nullable().optional(),
  value_json: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
});

const createSchema = z.object({
  config_test_id: z.number().int().positive(),
  json_path: z.string().min(1),
  value_text: z.string().nullable().optional(),
  value_json: z.string().nullable().optional(),
  data_type: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
});

const deleteSchema = z.object({
  id: z.number().int().positive(),
});

export async function PATCH(request: Request) {
  const payload = await request.json();
  const parsed = updateSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }
  try {
    const updated = updateConfigTestEntry(parsed.data);
    return NextResponse.json({ entry: updated });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to update entry",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const payload = await request.json();
  const parsed = createSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }
  try {
    const created = addConfigTestEntry(parsed.data);
    return NextResponse.json({ entry: created }, { status: 201 });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to add entry",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  const payload = await request.json();
  const parsed = deleteSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }
  try {
    deleteConfigTestEntry(parsed.data.id);
    return NextResponse.json({ success: true });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to delete entry",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
