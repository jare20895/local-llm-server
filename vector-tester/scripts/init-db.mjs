import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import path from "node:path";

const dbPath =
  process.env.LOG_DB_PATH ||
  path.join(process.cwd(), "data", "vector-tester.db");

mkdirSync(path.dirname(dbPath), { recursive: true });

const db = new Database(dbPath);

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

CREATE TABLE IF NOT EXISTS model_config_files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_test_id INTEGER NOT NULL,
  config_type TEXT NOT NULL,
  file_name TEXT NOT NULL,
  source_url TEXT,
  sha256 TEXT,
  content TEXT NOT NULL,
  parsed_at TEXT NOT NULL DEFAULT (datetime('now')),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (model_test_id) REFERENCES models_test(id) ON DELETE CASCADE,
  UNIQUE(model_test_id, config_type)
);

CREATE TABLE IF NOT EXISTS model_config_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  config_file_id INTEGER NOT NULL,
  parameter_id INTEGER,
  json_path TEXT NOT NULL,
  value_text TEXT,
  value_json TEXT,
  data_type TEXT,
  detected_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  notes TEXT,
  FOREIGN KEY (config_file_id) REFERENCES model_config_files(id) ON DELETE CASCADE,
  FOREIGN KEY (parameter_id) REFERENCES model_parameters(id) ON DELETE SET NULL,
  UNIQUE(config_file_id, json_path)
);

CREATE INDEX IF NOT EXISTS idx_model_config_entries_file
  ON model_config_entries(config_file_id);

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

console.log(`Initialized tester database at ${dbPath}`);
db.close();
