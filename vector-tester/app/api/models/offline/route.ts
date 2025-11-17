import { NextResponse } from "next/server";
import { z } from "zod";
import { fetchModels } from "@/lib/llm";
import { insertOrUpdateTestModel } from "@/lib/db";

const schema = z.object({
  model_name: z.string().min(1),
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

  const models = await fetchModels();
  const found = models.find(
    (model) => model.model_name === result.data.model_name
  );

  if (!found) {
    return NextResponse.json(
      { error: `Model '${result.data.model_name}' not found in registry` },
      { status: 404 }
    );
  }

  const copy = insertOrUpdateTestModel({
    source_model_id: found.id,
    model_name: found.model_name,
    hf_path: (found as any).hf_path ?? null,
    cache_location: found.cache_location ?? null,
    compatibility_status: found.compatibility_status ?? null,
    metadata: JSON.stringify(found),
    notes: "Offline copy requested",
  });

  return NextResponse.json({ model: copy }, { status: 201 });
}
