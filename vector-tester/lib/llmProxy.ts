import { getApiBase } from "./llm";

export async function forwardLlmRequest(
  path: string,
  init?: RequestInit & { expectJson?: boolean }
) {
  const url = `${getApiBase()}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  let payload: unknown;
  const text = await res.text();
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text;
  }

  return {
    ok: res.ok,
    status: res.status,
    data: payload,
  };
}
