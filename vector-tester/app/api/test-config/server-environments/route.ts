import "server-only";
import { NextResponse } from "next/server";
import { z } from "zod";
import {
  getServerEnvironments,
  upsertServerEnvironment,
  updateServerEnvironment,
  deleteServerEnvironment,
} from "@/lib/db";

const schema = z.object({
  name: z.string().min(2),
  hostname: z.string().optional(),
  ip_address: z.string().optional(),
  gpu_model: z.string().optional(),
  gpu_vram_gb: z.number().optional(),
  cpu_model: z.string().optional(),
  os_version: z.string().optional(),
  wsl_version: z.string().optional(),
  rocm_version: z.string().optional(),
  notes: z.string().optional(),
});

export async function GET() {
  const envs = getServerEnvironments();
  return NextResponse.json({ environments: envs });
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

  const env = upsertServerEnvironment(parsed.data);
  return NextResponse.json({ environment: env }, { status: 201 });
}

const updateSchema = z.object({
  id: z.number().int().positive(),
  name: z.string().min(2).optional(),
  hostname: z.string().optional(),
  ip_address: z.string().optional(),
  gpu_model: z.string().optional(),
  gpu_vram_gb: z.number().optional(),
  cpu_model: z.string().optional(),
  os_version: z.string().optional(),
  wsl_version: z.string().optional(),
  rocm_version: z.string().optional(),
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

  const env = updateServerEnvironment(parsed.data);
  return NextResponse.json({ environment: env });
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

  deleteServerEnvironment(parsed.data.id);
  return NextResponse.json({ success: true });
}
