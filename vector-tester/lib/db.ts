import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";

const dbPath =
  process.env.LOG_DB_PATH ||
  path.join(process.cwd(), "data", "vector-tester.db");

fs.mkdirSync(path.dirname(dbPath), { recursive: true });

const db = new Database(dbPath);

db.pragma("journal_mode = WAL");

db.exec(`
CREATE TABLE IF NOT EXISTS test_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT NOT NULL,
  scenario TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  completed_at TEXT,
  load_duration_ms REAL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS models_test (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_model_id INTEGER,
  model_name TEXT NOT NULL UNIQUE,
  hf_path TEXT,
  cache_location TEXT,
  compatibility_status TEXT,
  metadata TEXT,
  status TEXT NOT NULL DEFAULT 'staged',
  notes TEXT,
  cached_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_parameters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  data_type TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT,
  unit TEXT,
  default_value TEXT
);

CREATE TABLE IF NOT EXISTS model_parameter_values (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_test_id INTEGER NOT NULL,
  parameter_id INTEGER NOT NULL,
  value TEXT,
  json_value TEXT,
  notes TEXT,
  recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (model_test_id) REFERENCES models_test(id) ON DELETE CASCADE,
  FOREIGN KEY (parameter_id) REFERENCES model_parameters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS server_environments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  hostname TEXT,
  ip_address TEXT,
  gpu_model TEXT,
  gpu_vram_gb REAL,
  cpu_model TEXT,
  os_version TEXT,
  wsl_version TEXT,
  rocm_version TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS test_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  model_test_id INTEGER,
  server_environment_id INTEGER,
  default_prompt TEXT,
  max_tokens INTEGER,
  temperature REAL,
  top_p REAL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT,
  FOREIGN KEY (model_test_id) REFERENCES models_test(id) ON DELETE SET NULL,
  FOREIGN KEY (server_environment_id) REFERENCES server_environments(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS test_steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER NOT NULL,
  step_order INTEGER NOT NULL,
  step_name TEXT NOT NULL,
  api_method TEXT NOT NULL,
  api_path TEXT NOT NULL,
  request_body TEXT,
  expected_status INTEGER,
  expected_contains TEXT,
  pass_rule TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (profile_id) REFERENCES test_profiles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS log_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  model_id INTEGER,
  test_profile_id INTEGER,
  source TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE CASCADE,
  FOREIGN KEY (model_id) REFERENCES models_test(id) ON DELETE SET NULL,
  FOREIGN KEY (test_profile_id) REFERENCES test_profiles(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS huggingface_meta (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_path TEXT NOT NULL UNIQUE,
  display_path TEXT,
  element_type TEXT NOT NULL,
  description TEXT,
  category TEXT,
  semantic_role TEXT,
  example_value TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  detailed INTEGER NOT NULL DEFAULT 0,
  extensive INTEGER NOT NULL DEFAULT 0,
  parent_path TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS models_test_huggingface (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_test_id INTEGER NOT NULL,
  meta_id INTEGER NOT NULL,
  canonical_path TEXT NOT NULL,
  display_path TEXT,
  element_type TEXT NOT NULL,
  value_text TEXT,
  value_json TEXT,
  model_card_path TEXT,
  model_card_section TEXT,
  source_line INTEGER,
  detected_at TEXT NOT NULL DEFAULT (datetime('now')),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (model_test_id) REFERENCES models_test(id) ON DELETE CASCADE,
  FOREIGN KEY (meta_id) REFERENCES huggingface_meta(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_models_test_hf_model
  ON models_test_huggingface(model_test_id);
CREATE INDEX IF NOT EXISTS idx_models_test_hf_meta
  ON models_test_huggingface(meta_id);

CREATE TABLE IF NOT EXISTS swagger_endpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  method TEXT NOT NULL,
  path TEXT NOT NULL,
  summary TEXT,
  description TEXT,
  request_schema TEXT,
  response_schema TEXT,
  last_synced TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(method, path)
);
`);

type TableColumnInfo = { name: string };
function ensureColumn(table: string, column: string, definition: string) {
  const columns = db
    .prepare(`PRAGMA table_info(${table})`)
    .all() as TableColumnInfo[];
  if (!columns.some((col) => col.name === column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
  }
}

ensureColumn(
  "log_events",
  "model_id",
  "INTEGER REFERENCES models_test(id)"
);
ensureColumn(
  "log_events",
  "test_profile_id",
  "INTEGER REFERENCES test_profiles(id)"
);
ensureColumn(
  "models_test_huggingface",
  "created_at",
  "TEXT NOT NULL DEFAULT (datetime('now'))"
);
ensureColumn(
  "models_test_huggingface",
  "updated_at",
  "TEXT NOT NULL DEFAULT (datetime('now'))"
);

export type TestRun = {
  id: number;
  model_name: string;
  scenario: string | null;
  status: string;
  started_at: string;
  completed_at: string | null;
  load_duration_ms: number | null;
  notes: string | null;
};

export type LogEvent = {
  id: number;
  run_id: number | null;
  model_id: number | null;
  test_profile_id: number | null;
  source: string;
  level: string;
  message: string;
  created_at: string;
};

export type HuggingfaceMeta = {
  id: number;
  canonical_path: string;
  display_path: string | null;
  element_type: string;
  description: string | null;
  category: string | null;
  semantic_role: string | null;
  example_value: string | null;
  active: number;
  detailed: number;
  extensive: number;
  parent_path: string | null;
  updated_at: string;
};

export type ModelHuggingfaceElement = {
  id: number;
  model_test_id: number;
  meta_id: number;
  canonical_path: string;
  display_path: string | null;
  element_type: string;
  value_text: string | null;
  value_json: string | null;
  model_card_path: string | null;
  model_card_section: string | null;
  source_line: number | null;
  detected_at: string;
  created_at: string;
  updated_at: string;
};

export type ModelHuggingfaceRecord = ModelHuggingfaceElement & {
  category: string | null;
  semantic_role: string | null;
  active: number;
  detailed: number;
  extensive: number;
};

export type TestModelCopy = {
  id: number;
  source_model_id: number | null;
  model_name: string;
  hf_path: string | null;
  cache_location: string | null;
  compatibility_status: string | null;
  metadata: string | null;
  status: string;
  notes: string | null;
  cached_at: string;
};

export type ModelParameter = {
  id: number;
  name: string;
  description: string | null;
  data_type: string;
  unit: string | null;
  default_value: string | null;
  created_at: string;
  updated_at: string | null;
};

export type ModelParameterValue = {
  id: number;
  model_test_id: number;
  parameter_id: number;
  value: string | null;
  json_value: string | null;
  notes: string | null;
  recorded_at: string;
};

export type ServerEnvironment = {
  id: number;
  name: string;
  hostname: string | null;
  ip_address: string | null;
  gpu_model: string | null;
  gpu_vram_gb: number | null;
  cpu_model: string | null;
  os_version: string | null;
  wsl_version: string | null;
  rocm_version: string | null;
  notes: string | null;
  created_at: string;
};

export type TestProfile = {
  id: number;
  name: string;
  description: string | null;
  model_test_id: number | null;
  server_environment_id: number | null;
  default_prompt: string | null;
  max_tokens: number | null;
  temperature: number | null;
  top_p: number | null;
  active: number;
  created_at: string;
  updated_at: string | null;
};

export type TestStep = {
  id: number;
  profile_id: number;
  step_order: number;
  step_name: string;
  api_method: string;
  api_path: string;
  request_body: string | null;
  expected_status: number | null;
  expected_contains: string | null;
  pass_rule: string | null;
  notes: string | null;
  created_at: string;
};

export type SwaggerEndpoint = {
  id: number;
  method: string;
  path: string;
  summary: string | null;
  description: string | null;
  request_schema: string | null;
  response_schema: string | null;
  last_synced: string;
};

export function getRecentRuns(limit = 15): TestRun[] {
  const stmt = db.prepare(
    `SELECT * FROM test_runs ORDER BY datetime(started_at) DESC LIMIT ?`
  );
  return stmt.all(limit) as TestRun[];
}

export function insertTestRun(data: {
  model_name: string;
  scenario?: string;
  notes?: string;
}): TestRun {
  const stmt = db.prepare(
    `INSERT INTO test_runs (model_name, scenario, notes)
     VALUES (@model_name, @scenario, @notes)`
  );
  const info = stmt.run({
    model_name: data.model_name,
    scenario: data.scenario ?? null,
    notes: data.notes ?? null,
  });
  return getRunById(Number(info.lastInsertRowid));
}

export function updateRunStatus(data: {
  id: number;
  status?: string;
  load_duration_ms?: number | null;
  notes?: string | null;
}): TestRun {
  const stmt = db.prepare(
    `UPDATE test_runs
     SET status = COALESCE(@status, status),
         load_duration_ms = COALESCE(@load_duration_ms, load_duration_ms),
         completed_at = CASE
            WHEN @status IN ('failed','success') THEN datetime('now')
            ELSE completed_at
         END,
         notes = COALESCE(@notes, notes)
     WHERE id = @id`
  );
  stmt.run({
    id: data.id,
    status: data.status,
    load_duration_ms: data.load_duration_ms ?? null,
    notes: data.notes,
  });
  return getRunById(data.id);
}

export function getRunById(id: number): TestRun {
  const stmt = db.prepare(`SELECT * FROM test_runs WHERE id = ?`);
  return stmt.get(id) as TestRun;
}

export function insertLogEvent(data: {
  run_id?: number;
  model_id: number;
  test_profile_id?: number;
  source: string;
  level?: string;
  message: string;
}): LogEvent {
  const stmt = db.prepare(
    `INSERT INTO log_events (run_id, model_id, test_profile_id, source, level, message)
     VALUES (@run_id, @model_id, @test_profile_id, @source, COALESCE(@level, 'info'), @message)`
  );
  const info = stmt.run({
    run_id: data.run_id ?? null,
    model_id: data.model_id,
    test_profile_id: data.test_profile_id ?? null,
    source: data.source,
    level: data.level ?? "info",
    message: data.message,
  });
  return getLogById(Number(info.lastInsertRowid));
}

export function getLogById(id: number): LogEvent {
  const stmt = db.prepare(`SELECT * FROM log_events WHERE id = ?`);
  return stmt.get(id) as LogEvent;
}

export function getRecentLogs(limit = 50): LogEvent[] {
  const stmt = db.prepare(
    `SELECT * FROM log_events ORDER BY datetime(created_at) DESC LIMIT ?`
  );
  return stmt.all(limit) as LogEvent[];
}

export function insertOrUpdateTestModel(data: {
  source_model_id?: number | null;
  model_name: string;
  hf_path?: string | null;
  cache_location?: string | null;
  compatibility_status?: string | null;
  metadata?: string | null;
  status?: string;
  notes?: string | null;
}): TestModelCopy {
  const existing = getTestModelByName(data.model_name);
  if (existing) {
    const stmt = db.prepare(
      `UPDATE models_test
       SET source_model_id = COALESCE(@source_model_id, source_model_id),
           hf_path = COALESCE(@hf_path, hf_path),
           cache_location = COALESCE(@cache_location, cache_location),
           compatibility_status = COALESCE(@compatibility_status, compatibility_status),
           metadata = COALESCE(@metadata, metadata),
           status = COALESCE(@status, status),
           notes = COALESCE(@notes, notes),
           cached_at = datetime('now')
       WHERE model_name = @model_name`
    );
    stmt.run({
      ...data,
      source_model_id: data.source_model_id ?? null,
      hf_path: data.hf_path ?? null,
      cache_location: data.cache_location ?? null,
      compatibility_status: data.compatibility_status ?? null,
      metadata: data.metadata ?? null,
      status: data.status ?? null,
      notes: data.notes ?? null,
    });
    return getTestModelByName(data.model_name)!;
  }

  const stmt = db.prepare(
    `INSERT INTO models_test
      (source_model_id, model_name, hf_path, cache_location, compatibility_status, metadata, status, notes)
     VALUES
      (@source_model_id, @model_name, @hf_path, @cache_location, @compatibility_status, @metadata, COALESCE(@status, 'staged'), @notes)`
  );
  const info = stmt.run({
    source_model_id: data.source_model_id ?? null,
    model_name: data.model_name,
    hf_path: data.hf_path ?? null,
    cache_location: data.cache_location ?? null,
    compatibility_status: data.compatibility_status ?? null,
    metadata: data.metadata ?? null,
    status: data.status ?? "staged",
    notes: data.notes ?? null,
  });
  return getTestModelById(Number(info.lastInsertRowid));
}

export function getTestModels(): TestModelCopy[] {
  const stmt = db.prepare(
    `SELECT * FROM models_test ORDER BY datetime(cached_at) DESC`
  );
  return stmt.all() as TestModelCopy[];
}

export function getTestModelById(id: number): TestModelCopy {
  const stmt = db.prepare(`SELECT * FROM models_test WHERE id = ?`);
  return stmt.get(id) as TestModelCopy;
}

export function getTestModelByName(model_name: string): TestModelCopy | null {
  const stmt = db.prepare(`SELECT * FROM models_test WHERE model_name = ?`);
  const result = stmt.get(model_name);
  return (result as TestModelCopy) || null;
}

export function getHuggingfaceMetaTags(): HuggingfaceMeta[] {
  const stmt = db.prepare(
    `SELECT * FROM huggingface_meta ORDER BY canonical_path`
  );
  return stmt.all() as HuggingfaceMeta[];
}

export function getHuggingfaceMetaById(id: number): HuggingfaceMeta | null {
  const stmt = db.prepare(`SELECT * FROM huggingface_meta WHERE id = ?`);
  return (stmt.get(id) as HuggingfaceMeta) ?? null;
}

export function updateHuggingfaceMetaTag(data: {
  id: number;
  active?: boolean;
  detailed?: boolean;
  extensive?: boolean;
}): HuggingfaceMeta {
  const updates: string[] = [];
  const params: Record<string, unknown> = { id: data.id };
  if (data.active !== undefined) {
    updates.push("active = @active");
    params.active = data.active ? 1 : 0;
  }
  if (data.detailed !== undefined) {
    updates.push("detailed = @detailed");
    params.detailed = data.detailed ? 1 : 0;
  }
  if (data.extensive !== undefined) {
    updates.push("extensive = @extensive");
    params.extensive = data.extensive ? 1 : 0;
  }
  if (updates.length > 0) {
    const stmt = db.prepare(
      `UPDATE huggingface_meta
       SET ${updates.join(", ")},
           updated_at = datetime('now')
       WHERE id = @id`
    );
    stmt.run(params);
  }
  return getHuggingfaceMetaById(data.id)!;
}

export function getHuggingfaceMetadataForModel(
  model_test_id: number
): ModelHuggingfaceRecord[] {
  const stmt = db.prepare(
    `SELECT
       mth.*,
       hm.category,
       hm.semantic_role,
       hm.active,
       hm.detailed,
       hm.extensive
     FROM models_test_huggingface AS mth
     JOIN huggingface_meta AS hm ON hm.id = mth.meta_id
     WHERE mth.model_test_id = ?
     ORDER BY hm.canonical_path, mth.source_line`
  );
  return stmt.all(model_test_id) as ModelHuggingfaceRecord[];
}

export function deleteHuggingfaceMetadataForModel(model_test_id: number) {
  const stmt = db.prepare(
    `DELETE FROM models_test_huggingface WHERE model_test_id = ?`
  );
  stmt.run(model_test_id);
}

export function upsertModelParameter(param: {
  name: string;
  description?: string | null;
  data_type: string;
  unit?: string | null;
  default_value?: string | null;
}): ModelParameter {
  const stmt = db.prepare(
    `INSERT INTO model_parameters (name, description, data_type, unit, default_value)
     VALUES (@name, @description, @data_type, @unit, @default_value)
     ON CONFLICT(name) DO UPDATE SET
       description = excluded.description,
       data_type = excluded.data_type,
       unit = excluded.unit,
       default_value = excluded.default_value,
       updated_at = datetime('now')`
  );
  stmt.run({
    name: param.name,
    description: param.description ?? null,
    data_type: param.data_type,
    unit: param.unit ?? null,
    default_value: param.default_value ?? null,
  });
  const row = db
    .prepare(`SELECT * FROM model_parameters WHERE name = ?`)
    .get(param.name);
  return row as ModelParameter;
}

export function getModelParameters(): ModelParameter[] {
  const stmt = db.prepare(
    `SELECT * FROM model_parameters ORDER BY datetime(created_at) DESC`
  );
  return stmt.all() as ModelParameter[];
}

export function insertModelParameterValue(data: {
  model_test_id: number;
  parameter_id: number;
  value?: string | null;
  json_value?: unknown;
  notes?: string | null;
}): ModelParameterValue {
  const stmt = db.prepare(
    `INSERT INTO model_parameter_values
      (model_test_id, parameter_id, value, json_value, notes)
     VALUES
      (@model_test_id, @parameter_id, @value, @json_value, @notes)`
  );
  const info = stmt.run({
    model_test_id: data.model_test_id,
    parameter_id: data.parameter_id,
    value: data.value ?? null,
    json_value:
      data.json_value === undefined || data.json_value === null
        ? null
        : typeof data.json_value === "string"
        ? data.json_value
        : JSON.stringify(data.json_value),
    notes: data.notes ?? null,
  });
  return getModelParameterValueById(Number(info.lastInsertRowid));
}

export function getModelParameterValueById(
  id: number
): ModelParameterValue {
  const stmt = db.prepare(
    `SELECT * FROM model_parameter_values WHERE id = ?`
  );
  return stmt.get(id) as ModelParameterValue;
}

export function getParameterValuesForModel(
  model_test_id: number
): ModelParameterValue[] {
  const stmt = db.prepare(
    `SELECT * FROM model_parameter_values
     WHERE model_test_id = ?
     ORDER BY datetime(recorded_at) DESC`
  );
  return stmt.all(model_test_id) as ModelParameterValue[];
}

export function deleteModelParameterValue(id: number) {
  db.prepare(`DELETE FROM model_parameter_values WHERE id = ?`).run(id);
}

export function upsertServerEnvironment(env: {
  name: string;
  hostname?: string | null;
  ip_address?: string | null;
  gpu_model?: string | null;
  gpu_vram_gb?: number | null;
  cpu_model?: string | null;
  os_version?: string | null;
  wsl_version?: string | null;
  rocm_version?: string | null;
  notes?: string | null;
}): ServerEnvironment {
  const stmt = db.prepare(
    `INSERT INTO server_environments
      (name, hostname, ip_address, gpu_model, gpu_vram_gb, cpu_model, os_version, wsl_version, rocm_version, notes)
     VALUES
      (@name, @hostname, @ip_address, @gpu_model, @gpu_vram_gb, @cpu_model, @os_version, @wsl_version, @rocm_version, @notes)
     ON CONFLICT(name) DO UPDATE SET
      hostname = excluded.hostname,
      ip_address = excluded.ip_address,
      gpu_model = excluded.gpu_model,
      gpu_vram_gb = excluded.gpu_vram_gb,
      cpu_model = excluded.cpu_model,
      os_version = excluded.os_version,
      wsl_version = excluded.wsl_version,
      rocm_version = excluded.rocm_version,
      notes = excluded.notes`
  );
  stmt.run({
    name: env.name,
    hostname: env.hostname ?? null,
    ip_address: env.ip_address ?? null,
    gpu_model: env.gpu_model ?? null,
    gpu_vram_gb: env.gpu_vram_gb ?? null,
    cpu_model: env.cpu_model ?? null,
    os_version: env.os_version ?? null,
    wsl_version: env.wsl_version ?? null,
    rocm_version: env.rocm_version ?? null,
    notes: env.notes ?? null,
  });
  const row = db
    .prepare(`SELECT * FROM server_environments WHERE name = ?`)
    .get(env.name);
  return row as ServerEnvironment;
}

export function getServerEnvironments(): ServerEnvironment[] {
  const stmt = db.prepare(
    `SELECT * FROM server_environments ORDER BY datetime(created_at) DESC`
  );
  return stmt.all() as ServerEnvironment[];
}

export function insertTestProfile(profile: {
  name: string;
  description?: string;
  model_test_id?: number | null;
  server_environment_id?: number | null;
  default_prompt?: string | null;
  max_tokens?: number | null;
  temperature?: number | null;
  top_p?: number | null;
}): TestProfile {
  const stmt = db.prepare(
    `INSERT INTO test_profiles
      (name, description, model_test_id, server_environment_id, default_prompt, max_tokens, temperature, top_p)
     VALUES
      (@name, @description, @model_test_id, @server_environment_id, @default_prompt, @max_tokens, @temperature, @top_p)`
  );
  const info = stmt.run({
    name: profile.name,
    description: profile.description ?? null,
    model_test_id: profile.model_test_id ?? null,
    server_environment_id: profile.server_environment_id ?? null,
    default_prompt: profile.default_prompt ?? null,
    max_tokens: profile.max_tokens ?? null,
    temperature: profile.temperature ?? null,
    top_p: profile.top_p ?? null,
  });
  return getTestProfileById(Number(info.lastInsertRowid));
}

export function updateTestProfile(data: {
  id: number;
  name?: string;
  description?: string | null;
  model_test_id?: number | null;
  server_environment_id?: number | null;
  default_prompt?: string | null;
  max_tokens?: number | null;
  temperature?: number | null;
  top_p?: number | null;
  active?: number;
}): TestProfile {
  const stmt = db.prepare(
    `UPDATE test_profiles
     SET name = COALESCE(@name, name),
         description = COALESCE(@description, description),
         model_test_id = COALESCE(@model_test_id, model_test_id),
         server_environment_id = COALESCE(@server_environment_id, server_environment_id),
         default_prompt = COALESCE(@default_prompt, default_prompt),
         max_tokens = COALESCE(@max_tokens, max_tokens),
         temperature = COALESCE(@temperature, temperature),
         top_p = COALESCE(@top_p, top_p),
         active = COALESCE(@active, active),
         updated_at = datetime('now')
     WHERE id = @id`
  );
  stmt.run({
    id: data.id,
    name: data.name,
    description: data.description ?? null,
    model_test_id: data.model_test_id ?? null,
    server_environment_id: data.server_environment_id ?? null,
    default_prompt: data.default_prompt ?? null,
    max_tokens: data.max_tokens ?? null,
    temperature: data.temperature ?? null,
    top_p: data.top_p ?? null,
    active: data.active ?? null,
  });
  return getTestProfileById(data.id);
}

export function deleteTestProfile(id: number) {
  db.prepare(`DELETE FROM test_profiles WHERE id = ?`).run(id);
}

export function getTestProfiles(limit = 25): TestProfile[] {
  const stmt = db.prepare(
    `SELECT * FROM test_profiles ORDER BY datetime(created_at) DESC LIMIT ?`
  );
  return stmt.all(limit) as TestProfile[];
}

export function getTestProfileById(id: number): TestProfile {
  const stmt = db.prepare(`SELECT * FROM test_profiles WHERE id = ?`);
  return stmt.get(id) as TestProfile;
}

export function insertTestStep(step: {
  profile_id: number;
  step_order: number;
  step_name: string;
  api_method: string;
  api_path: string;
  request_body?: string | null;
  expected_status?: number | null;
  expected_contains?: string | null;
  pass_rule?: string | null;
  notes?: string | null;
}): TestStep {
  const stmt = db.prepare(
    `INSERT INTO test_steps
      (profile_id, step_order, step_name, api_method, api_path, request_body, expected_status, expected_contains, pass_rule, notes)
     VALUES
      (@profile_id, @step_order, @step_name, @api_method, @api_path, @request_body, @expected_status, @expected_contains, @pass_rule, @notes)`
  );
  const info = stmt.run({
    profile_id: step.profile_id,
    step_order: step.step_order,
    step_name: step.step_name,
    api_method: step.api_method,
    api_path: step.api_path,
    request_body: step.request_body ?? null,
    expected_status: step.expected_status ?? null,
    expected_contains: step.expected_contains ?? null,
    pass_rule: step.pass_rule ?? null,
    notes: step.notes ?? null,
  });
  return getTestStepById(Number(info.lastInsertRowid));
}

export function getStepsForProfile(profile_id: number): TestStep[] {
  const stmt = db.prepare(
    `SELECT * FROM test_steps WHERE profile_id = ? ORDER BY step_order ASC`
  );
  return stmt.all(profile_id) as TestStep[];
}

export function getTestStepById(id: number): TestStep {
  const stmt = db.prepare(`SELECT * FROM test_steps WHERE id = ?`);
  return stmt.get(id) as TestStep;
}

export function updateTestStep(data: {
  id: number;
  step_order?: number;
  step_name?: string;
  api_method?: string;
  api_path?: string;
  request_body?: string | null;
  expected_status?: number | null;
  expected_contains?: string | null;
  pass_rule?: string | null;
  notes?: string | null;
}): TestStep {
  const stmt = db.prepare(
    `UPDATE test_steps
     SET step_order = COALESCE(@step_order, step_order),
         step_name = COALESCE(@step_name, step_name),
         api_method = COALESCE(@api_method, api_method),
         api_path = COALESCE(@api_path, api_path),
         request_body = COALESCE(@request_body, request_body),
         expected_status = COALESCE(@expected_status, expected_status),
         expected_contains = COALESCE(@expected_contains, expected_contains),
         pass_rule = COALESCE(@pass_rule, pass_rule),
         notes = COALESCE(@notes, notes)
     WHERE id = @id`
  );
  stmt.run({
    id: data.id,
    step_order: data.step_order,
    step_name: data.step_name,
    api_method: data.api_method,
    api_path: data.api_path,
    request_body: data.request_body ?? null,
    expected_status: data.expected_status ?? null,
    expected_contains: data.expected_contains ?? null,
    pass_rule: data.pass_rule ?? null,
    notes: data.notes ?? null,
  });
  return getTestStepById(data.id);
}

export function deleteTestStep(id: number) {
  db.prepare(`DELETE FROM test_steps WHERE id = ?`).run(id);
}

export function upsertSwaggerEndpoint(data: {
  method: string;
  path: string;
  summary?: string | null;
  description?: string | null;
  request_schema?: string | null;
  response_schema?: string | null;
}): SwaggerEndpoint {
  const stmt = db.prepare(
    `INSERT INTO swagger_endpoints
      (method, path, summary, description, request_schema, response_schema)
     VALUES
      (@method, @path, @summary, @description, @request_schema, @response_schema)
     ON CONFLICT(method, path) DO UPDATE SET
      summary = excluded.summary,
      description = excluded.description,
      request_schema = excluded.request_schema,
      response_schema = excluded.response_schema,
      last_synced = datetime('now')`
  );
  stmt.run({
    method: data.method.toUpperCase(),
    path: data.path,
    summary: data.summary ?? null,
    description: data.description ?? null,
    request_schema: data.request_schema ?? null,
    response_schema: data.response_schema ?? null,
  });
  const row = db
    .prepare(
      `SELECT * FROM swagger_endpoints WHERE method = ? AND path = ?`
    )
    .get(data.method.toUpperCase(), data.path);
  return row as SwaggerEndpoint;
}

export function getSwaggerEndpoints(): SwaggerEndpoint[] {
  const stmt = db.prepare(
    `SELECT * FROM swagger_endpoints ORDER BY path, method`
  );
  return stmt.all() as SwaggerEndpoint[];
}

export function deleteSwaggerEndpoint(id: number) {
  db.prepare(`DELETE FROM swagger_endpoints WHERE id = ?`).run(id);
}

export function countTestSteps(): number {
  const row = db
    .prepare(`SELECT COUNT(*) as count FROM test_steps`)
    .get() as { count: number } | undefined;
  return row?.count ?? 0;
}

export { db };
export function updateServerEnvironment(data: {
  id: number;
  name?: string;
  hostname?: string | null;
  ip_address?: string | null;
  gpu_model?: string | null;
  gpu_vram_gb?: number | null;
  cpu_model?: string | null;
  os_version?: string | null;
  wsl_version?: string | null;
  rocm_version?: string | null;
  notes?: string | null;
}): ServerEnvironment {
  const stmt = db.prepare(
    `UPDATE server_environments
     SET name = COALESCE(@name, name),
         hostname = COALESCE(@hostname, hostname),
         ip_address = COALESCE(@ip_address, ip_address),
         gpu_model = COALESCE(@gpu_model, gpu_model),
         gpu_vram_gb = COALESCE(@gpu_vram_gb, gpu_vram_gb),
         cpu_model = COALESCE(@cpu_model, cpu_model),
         os_version = COALESCE(@os_version, os_version),
         wsl_version = COALESCE(@wsl_version, wsl_version),
         rocm_version = COALESCE(@rocm_version, rocm_version),
         notes = COALESCE(@notes, notes)
     WHERE id = @id`
  );
  stmt.run({
    id: data.id,
    name: data.name,
    hostname: data.hostname ?? null,
    ip_address: data.ip_address ?? null,
    gpu_model: data.gpu_model ?? null,
    gpu_vram_gb: data.gpu_vram_gb ?? null,
    cpu_model: data.cpu_model ?? null,
    os_version: data.os_version ?? null,
    wsl_version: data.wsl_version ?? null,
    rocm_version: data.rocm_version ?? null,
    notes: data.notes ?? null,
  });
  const row = db
    .prepare(`SELECT * FROM server_environments WHERE id = ?`)
    .get(data.id);
  return row as ServerEnvironment;
}

export function deleteServerEnvironment(id: number) {
  db.prepare(`DELETE FROM server_environments WHERE id = ?`).run(id);
}
