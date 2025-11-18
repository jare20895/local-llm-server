import "server-only";
import { NextResponse } from "next/server";
import { z } from "zod";
import {
  getTestProfiles,
  getStepsForProfile,
  insertTestStep,
  getSwaggerEndpoints,
  updateTestStep,
  deleteTestStep,
} from "@/lib/db";

const schema = z.object({
  profile_id: z.number().int().positive(),
  step_order: z.number().int().nonnegative(),
  step_name: z.string().min(2),
  api_method: z.string().min(3),
  api_path: z.string().min(2),
  request_body: z.string().optional(),
  expected_status: z.number().int().optional(),
  expected_contains: z.string().optional(),
  pass_rule: z.string().optional(),
  notes: z.string().optional(),
});

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const profileId = searchParams.get("profile_id");
  const profiles = getTestProfiles();
  const swagger = getSwaggerEndpoints();
  const steps =
    profileId && Number(profileId)
      ? getStepsForProfile(Number(profileId))
      : [];

  return NextResponse.json({
    profiles,
    steps,
    swagger,
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

  const step = insertTestStep(parsed.data);
  return NextResponse.json({ step }, { status: 201 });
}

const updateSchema = z.object({
  id: z.number().int().positive(),
  step_order: z.number().optional(),
  step_name: z.string().optional(),
  api_method: z.string().optional(),
  api_path: z.string().optional(),
  request_body: z.string().optional(),
  expected_status: z.number().optional(),
  expected_contains: z.string().optional(),
  pass_rule: z.string().optional(),
  notes: z.string().optional(),
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

  const step = updateTestStep(parsed.data);
  return NextResponse.json({ step });
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

  deleteTestStep(parsed.data.id);
  return NextResponse.json({ success: true });
}
