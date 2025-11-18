import "server-only";
import { NextResponse } from "next/server";
import { z } from "zod";
import {
  getSwaggerEndpoints,
  upsertSwaggerEndpoint,
  deleteSwaggerEndpoint,
} from "@/lib/db";

const schema = z.object({
  method: z.string().min(3),
  path: z.string().min(1),
  summary: z.string().optional(),
  description: z.string().optional(),
  request_schema: z.string().optional(),
  response_schema: z.string().optional(),
});

export async function GET() {
  const endpoints = getSwaggerEndpoints();
  return NextResponse.json({ endpoints });
}

export async function POST(request: Request) {
  const payload = await request.json();
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }

  const endpoint = upsertSwaggerEndpoint(parsed.data);
  return NextResponse.json({ endpoint }, { status: 201 });
}

const deleteSchema = z.object({
  id: z.number().int().positive(),
});

export async function DELETE(request: Request) {
  const payload = await request.json();
  const parsed = deleteSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }

  deleteSwaggerEndpoint(parsed.data.id);
  return NextResponse.json({ success: true });
}
