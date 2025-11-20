import { NextResponse } from "next/server";
import { z } from "zod";
import {
  createModelConfigTest,
  getTestModelById,
} from "@/lib/db";

const schema = z.object({
  model_test_id: z.number().int().positive(),
  config_type: z.enum(["config", "generation"]),
  name: z.string().min(1),
  description: z.string().optional(),
});

export async function POST(request: Request) {
  const body = await request.json();
  const result = schema.safeParse(body);
  if (!result.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: result.error.flatten() },
      { status: 400 }
    );
  }

  try {
    const model = getTestModelById(result.data.model_test_id);
    if (!model) {
      return NextResponse.json(
        { error: "Model test record not found" },
        { status: 404 }
      );
    }
    const test = createModelConfigTest({
      model_test_id: result.data.model_test_id,
      config_type: result.data.config_type,
      name: result.data.name,
      description: result.data.description ?? null,
    });
    return NextResponse.json({ test }, { status: 201 });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to create config test",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
