import { NextResponse } from "next/server";
import {
  getConfigEntriesForFile,
  getConfigFilesForModel,
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
    const files = getConfigFilesForModel(modelId).map((file) => ({
      ...file,
      entries: getConfigEntriesForFile(file.id),
    }));
    return NextResponse.json({ files });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to load configs",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
