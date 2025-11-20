import { NextResponse } from "next/server";
import { getHuggingfaceMetadataForModel } from "@/lib/db";

type Params = {
  params: {
    id: string;
  };
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
    const metadata = getHuggingfaceMetadataForModel(modelId);
    return NextResponse.json({ metadata });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to load metadata",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
