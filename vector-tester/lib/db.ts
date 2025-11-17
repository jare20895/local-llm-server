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

export { db };
