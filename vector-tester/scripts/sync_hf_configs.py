#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download, HfApi


try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
except Exception:
    LOCAL_TZ = timezone(timedelta(hours=-5))


def current_timestamp() -> str:
    return datetime.now(LOCAL_TZ).replace(microsecond=0).isoformat()


CONFIG_TARGETS = {
    "config": "config.json",
    "generation": "generation_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync HuggingFace config files into Vector Tester DB"
    )
    parser.add_argument("--db", type=str, default=None, help="Path to tester DB")
    parser.add_argument("--model-test-id", type=int, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--hf-path", type=str, default=None)
    parser.add_argument(
        "--config-types",
        type=str,
        nargs="+",
        default=["config", "generation"],
        help="Which config files to sync (config, generation)",
    )
    return parser.parse_args()


def resolve_db_path(user_path: Optional[str]) -> str:
    if user_path:
        return user_path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "vector-tester.db")


def determine_repo_id(row: sqlite3.Row, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if row["hf_path"]:
        return row["hf_path"]
    try:
        metadata = json.loads(row["metadata"] or "{}")
        if isinstance(metadata, dict) and metadata.get("hf_path"):
            return metadata["hf_path"]
    except json.JSONDecodeError:
        pass
    return row["model_name"]


def safe_json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def flatten_json(data) -> List[Tuple[str, object]]:
    results: List[Tuple[str, object]] = []

    def walk(value, prefix: str):
        if isinstance(value, dict):
            for key, child in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                walk(child, new_prefix)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                walk(child, new_prefix)
        else:
            results.append((prefix or "root", value))

    walk(data, "")
    return results


def detect_data_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def load_parameter_lookup(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute("SELECT id, name FROM model_parameters").fetchall()
    mapping = {}
    for row in rows:
        mapping[row["name"].lower()] = row["id"]
    return mapping


def infer_parameter_id(path: str, lookup: Dict[str, int]) -> Optional[int]:
    key = path.split(".")[-1]
    key = key.split("[")[0]
    return lookup.get(key.lower())


def upsert_config_file(
    conn: sqlite3.Connection,
    model_test_id: int,
    config_type: str,
    file_name: str,
    source_url: Optional[str],
    content: str,
) -> Tuple[int, bool]:
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    row = conn.execute(
        """
        SELECT id, sha256 FROM model_config_files
        WHERE model_test_id = ? AND config_type = ?
        """,
        (model_test_id, config_type),
    ).fetchone()
    now = current_timestamp()
    if row:
        changed = row["sha256"] != sha or row["sha256"] is None
        conn.execute(
            """
            UPDATE model_config_files
            SET file_name = ?,
                source_url = ?,
                sha256 = ?,
                content = ?,
                parsed_at = ?,
                updated_at = CASE WHEN ? THEN ? ELSE updated_at END
            WHERE id = ?
            """,
            (
                file_name,
                source_url,
                sha,
                content,
                now,
                1 if changed else 0,
                now,
                row["id"],
            ),
        )
        return row["id"], changed
    cursor = conn.execute(
        """
        INSERT INTO model_config_files
          (model_test_id, config_type, file_name, source_url, sha256, content, parsed_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_test_id,
            config_type,
            file_name,
            source_url,
            sha,
            content,
            now,
            now,
            now,
        ),
    )
    return cursor.lastrowid, True


def sync_config_entries(
    conn: sqlite3.Connection,
    config_file_id: int,
    entries: List[Tuple[str, object]],
    parameter_lookup: Dict[str, int],
) -> Tuple[int, int, int]:
    existing_rows = conn.execute(
        "SELECT * FROM model_config_entries WHERE config_file_id = ?",
        (config_file_id,),
    ).fetchall()
    existing = {row["json_path"]: row for row in existing_rows}
    seen_paths = set()
    now = current_timestamp()
    inserted = 0
    updated = 0

    for path, value in entries:
        seen_paths.add(path)
        value_text: Optional[str] = None
        value_json: Optional[str] = None
        if isinstance(value, (dict, list)):
            value_json = safe_json_dumps(value)
        elif value is None:
            value_text = "null"
        else:
            value_text = str(value)

        payload = {
            "json_path": path,
            "value_text": value_text,
            "value_json": value_json,
            "data_type": detect_data_type(value),
            "parameter_id": infer_parameter_id(path, parameter_lookup),
        }

        if path in existing:
            row = existing[path]
            changed = any(row[key] != payload[key] for key in ["value_text", "value_json", "data_type", "parameter_id"])
            if changed:
                conn.execute(
                    """
                    UPDATE model_config_entries
                    SET value_text = ?,
                        value_json = ?,
                        data_type = ?,
                        parameter_id = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        payload["value_text"],
                        payload["value_json"],
                        payload["data_type"],
                        payload["parameter_id"],
                        now,
                        row["id"],
                    ),
                )
                updated += 1
        else:
            conn.execute(
                """
                INSERT INTO model_config_entries
                  (config_file_id, parameter_id, json_path, value_text, value_json, data_type, detected_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config_file_id,
                    payload["parameter_id"],
                    payload["json_path"],
                    payload["value_text"],
                    payload["value_json"],
                    payload["data_type"],
                    now,
                    now,
                ),
            )
            inserted += 1

    stale_paths = [path for path in existing.keys() if path not in seen_paths]
    pruned = 0
    if stale_paths:
        placeholders = ",".join("?" for _ in stale_paths)
        pruned = conn.execute(
            f"""
            DELETE FROM model_config_entries
            WHERE config_file_id = ? AND json_path IN ({placeholders})
            """,
            (config_file_id, *stale_paths),
        ).rowcount
    return inserted, updated, pruned


def load_json_content(repo_id: str, filename: str, token: Optional[str]) -> Optional[str]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token, repo_type="model")
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    db_path = resolve_db_path(args.db)
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM models_test WHERE id = ?", (args.model_test_id,)
    ).fetchone()
    if not row:
        print(f"models_test row {args.model_test_id} not found", file=sys.stderr)
        return 2
    repo_id = determine_repo_id(row, args.hf_path)
    if not repo_id:
        print("Unable to determine HuggingFace repo id", file=sys.stderr)
        return 3

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
    except Exception as exc:
        print(f"Failed to load model info for {repo_id}: {exc}", file=sys.stderr)
        return 4

    parameter_lookup = load_parameter_lookup(conn)
    results = []
    for cfg in args.config_types:
        cfg_key = cfg.lower()
        if cfg_key not in CONFIG_TARGETS:
            continue
        filename = CONFIG_TARGETS[cfg_key]
        raw = load_json_content(repo_id, filename, token)
        if raw is None:
            results.append(
                {
                    "config_type": cfg_key,
                    "status": "missing",
                }
            )
            continue
        try:
            content = json.loads(raw)
        except json.JSONDecodeError as exc:
            results.append(
                {
                    "config_type": cfg_key,
                    "status": "invalid",
                    "error": str(exc),
                }
            )
            continue
        file_id, changed = upsert_config_file(
            conn,
            args.model_test_id,
            cfg_key,
            filename,
            f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            safe_json_dumps(content),
        )
        flattened = flatten_json(content)
        inserted, updated, pruned = sync_config_entries(
            conn, file_id, flattened, parameter_lookup
        )
        conn.commit()
        results.append(
            {
                "config_type": cfg_key,
                "file_changed": changed,
                "entries_created": inserted,
                "entries_updated": updated,
                "entries_removed": pruned,
            }
        )

    summary = {
        "model_test_id": args.model_test_id,
        "model_name": args.model_name or row["model_name"],
        "repo_id": repo_id,
        "results": results,
        "model_sha": info.sha,
        "timestamp": current_timestamp(),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
