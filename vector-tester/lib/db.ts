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

CREATE TABLE IF NOT EXISTS log_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  source TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE CASCADE
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
`);

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
  source: string;
  level: string;
  message: string;
  created_at: string;
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
  source: string;
  level?: string;
  message: string;
}): LogEvent {
  const stmt = db.prepare(
    `INSERT INTO log_events (run_id, source, level, message)
     VALUES (@run_id, @source, COALESCE(@level, 'info'), @message)`
  );
  const info = stmt.run({
    run_id: data.run_id ?? null,
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

export { db };
