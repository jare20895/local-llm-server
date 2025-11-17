import { NextResponse } from "next/server";

const spec = {
  openapi: "3.0.3",
  info: {
    title: "Vector-Tester API",
    version: "1.0.0",
    description:
      "API for logging LLM model test runs and ingestion of orchestrator/docker logs.",
  },
  servers: [
    {
      url: "{scheme}://{host}",
      variables: {
        scheme: {
          enum: ["http", "https"],
          default: "http",
        },
        host: {
          default: "localhost:4173",
        },
      },
    },
  ],
  paths: {
    "/api/test-runs": {
      get: {
        summary: "List recent test runs",
        tags: ["Test Runs"],
        responses: {
          200: {
            description: "OK",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    runs: {
                      type: "array",
                      items: { $ref: "#/components/schemas/TestRun" },
                    },
                  },
                },
              },
            },
          },
        },
      },
      post: {
        summary: "Create a test run entry",
        tags: ["Test Runs"],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["model_name"],
                properties: {
                  model_name: { type: "string" },
                  scenario: { type: "string" },
                  notes: { type: "string" },
                },
              },
            },
          },
        },
        responses: {
          201: {
            description: "Created",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    run: { $ref: "#/components/schemas/TestRun" },
                  },
                },
              },
            },
          },
          400: { description: "Invalid payload" },
        },
      },
      patch: {
        summary: "Update a test run status",
        tags: ["Test Runs"],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["id"],
                properties: {
                  id: { type: "integer" },
                  status: { type: "string" },
                  load_duration_ms: { type: "number" },
                  notes: { type: "string" },
                },
              },
            },
          },
        },
        responses: {
          200: {
            description: "OK",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: { run: { $ref: "#/components/schemas/TestRun" } },
                },
              },
            },
          },
          400: { description: "Invalid payload" },
          404: { description: "Not found" },
        },
      },
    },
    "/api/log-events": {
      get: {
        summary: "List recent log events",
        tags: ["Log Events"],
        responses: {
          200: {
            description: "OK",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    logs: {
                      type: "array",
                      items: { $ref: "#/components/schemas/LogEvent" },
                    },
                  },
                },
              },
            },
          },
        },
      },
      post: {
        summary: "Create a log event",
        tags: ["Log Events"],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["source", "message"],
                properties: {
                  run_id: { type: "integer" },
                  source: { type: "string" },
                  level: {
                    type: "string",
                    enum: ["debug", "info", "warn", "error"],
                  },
                  message: { type: "string" },
                },
              },
            },
          },
        },
        responses: {
          201: {
            description: "Created",
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: { log: { $ref: "#/components/schemas/LogEvent" } },
                },
              },
            },
          },
          400: { description: "Invalid payload" },
        },
      },
    },
  },
  components: {
    schemas: {
      TestRun: {
        type: "object",
        properties: {
          id: { type: "integer" },
          model_name: { type: "string" },
          scenario: { type: "string", nullable: true },
          status: { type: "string" },
          started_at: { type: "string", format: "date-time" },
          completed_at: { type: "string", format: "date-time", nullable: true },
          load_duration_ms: { type: "number", nullable: true },
          notes: { type: "string", nullable: true },
        },
      },
      LogEvent: {
        type: "object",
        properties: {
          id: { type: "integer" },
          run_id: { type: "integer", nullable: true },
          source: { type: "string" },
          level: { type: "string" },
          message: { type: "string" },
          created_at: { type: "string", format: "date-time" },
        },
      },
    },
  },
};

export async function GET() {
  return NextResponse.json(spec);
}
