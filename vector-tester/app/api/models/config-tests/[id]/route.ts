import { NextResponse } from "next/server";
import {
  getConfigTestsForModel,
  getConfigTestEntries,
} from "@/lib/db";

type Params = {
  params: { id: string };
};

export function GET(_: Request, { params }: Params) {
  const modelId = Number(params.id);
  if (Number.isNaN(modelId)) {
    return NextResponse.json(
      { error: "Invalid model id" },
      { status: 400 }
    );
  }
  try {
    const tests = getConfigTestsForModel(modelId).map((test) => ({
      ...test,
      entries: getConfigTestEntries(test.id),
    }));
    return NextResponse.json({ tests });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to load config tests",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
