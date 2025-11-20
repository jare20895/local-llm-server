import { NextResponse } from "next/server";
import { z } from "zod";
import {
  getHuggingfaceMetaTags,
  updateHuggingfaceMetaTag,
} from "@/lib/db";

const updateSchema = z.object({
  id: z.number(),
  active: z.boolean().optional(),
  detailed: z.boolean().optional(),
  extensive: z.boolean().optional(),
});

export function GET() {
  const metadata = getHuggingfaceMetaTags();
  return NextResponse.json({ metadata });
}

export async function PATCH(request: Request) {
  const payload = await request.json();
  const parsed = updateSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.flatten() },
      { status: 400 }
    );
  }
  try {
    const updated = updateHuggingfaceMetaTag(parsed.data);
    return NextResponse.json({ meta: updated });
  } catch (error: unknown) {
    return NextResponse.json(
      {
        error: "Failed to update metadata entry",
        details:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 }
    );
  }
}
