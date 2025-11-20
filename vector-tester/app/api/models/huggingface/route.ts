import { NextResponse } from "next/server";
import { z } from "zod";
import { spawn } from "node:child_process";
import path from "node:path";
import {
  getTestModelByName,
  insertOrUpdateTestModel,
  type TestModelCopy,
} from "@/lib/db";
import { fetchModels } from "@/lib/llm";

const detailLevelEnum = z.enum(["basic", "detailed", "extensive"]);
const schema = z.object({
  model_name: z.string().min(1),
  detail_level: detailLevelEnum.optional(),
});

async function ensureLocalModel(modelName: string): Promise<TestModelCopy> {
  const existing = getTestModelByName(modelName);
  if (existing && existing.hf_path) {
    return existing;
  }
  const models = await fetchModels();
  const source = models.find((model) => model.model_name === modelName);
  if (!source) {
    if (existing) {
      return existing;
    }
    throw new Error(`Model '${modelName}' was not found in registry`);
  }
  const updated = insertOrUpdateTestModel({
    source_model_id: source.id,
    model_name: source.model_name,
    hf_path: (source as any).hf_path ?? null,
    cache_location: source.cache_location ?? null,
    compatibility_status: source.compatibility_status ?? null,
    metadata: JSON.stringify(source),
    notes: "HF metadata sync staged this model",
  });
  return updated;
}

function runPythonScript(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", args, {
      cwd: process.cwd(),
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(stderr.trim() || `Python exited with code ${code}`));
      }
    });
    child.on("error", reject);
  });
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

  try {
    const detailLevel = result.data.detail_level ?? "basic";
    const localModel = await ensureLocalModel(result.data.model_name);
    const dbPath =
      process.env.LOG_DB_PATH ||
      path.join(process.cwd(), "data", "vector-tester.db");
    const scriptPath = path.join(
      process.cwd(),
      "scripts",
      "sync_hf_metadata.py"
    );
    const args = [
      scriptPath,
      "--db",
      dbPath,
      "--model-test-id",
      localModel.id.toString(),
      "--model-name",
      localModel.model_name,
    ];
    args.push("--detail-level", detailLevel);
    if (localModel.hf_path) {
      args.push("--hf-path", localModel.hf_path);
    }

    const raw = await runPythonScript(args);
    let payload: unknown;
    try {
      payload = JSON.parse(raw);
    } catch (error) {
      return NextResponse.json(
        {
          error: "Metadata sync failed: invalid JSON response",
          details: raw,
        },
        { status: 502 }
      );
    }
    const refreshed = getTestModelByName(localModel.model_name) ?? localModel;
    return NextResponse.json(
      { result: payload, model: refreshed },
      { status: 200 }
    );
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to sync HuggingFace metadata",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
