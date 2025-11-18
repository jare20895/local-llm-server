import "server-only";
import { NextResponse } from "next/server";
import { z } from "zod";
import {
  getTestProfiles,
  insertTestProfile,
  getTestModels,
  getServerEnvironments,
  updateTestProfile,
  deleteTestProfile,
} from "@/lib/db";

const schema = z.object({
  name: z.string().min(2),
  description: z.string().optional(),
  model_test_id: z.number().optional(),
  server_environment_id: z.number().optional(),
  default_prompt: z.string().optional(),
  max_tokens: z.number().optional(),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
});

export async function GET() {
  const profiles = getTestProfiles();
  const models = getTestModels();
  const envs = getServerEnvironments();
  return NextResponse.json({
    profiles,
    staged_models: models,
    server_environments: envs,
  });
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

  const profile = insertTestProfile(parsed.data);
  return NextResponse.json({ profile }, { status: 201 });
}

const updateSchema = z.object({
  id: z.number().int().positive(),
  name: z.string().min(2).optional(),
  description: z.string().optional(),
  model_test_id: z.number().optional(),
  server_environment_id: z.number().optional(),
  default_prompt: z.string().optional(),
  max_tokens: z.number().optional(),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  active: z.number().optional(),
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

  const profile = updateTestProfile(parsed.data);
  return NextResponse.json({ profile });
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

  deleteTestProfile(parsed.data.id);
  return NextResponse.json({ success: true });
}
