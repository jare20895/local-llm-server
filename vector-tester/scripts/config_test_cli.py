#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from datetime import datetime


def resolve_db_path(path: str | None) -> str:
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "vector-tester.db")


def list_tests(conn: sqlite3.Connection, model_test_id: int | None):
    if model_test_id:
        rows = conn.execute(
            """
            SELECT * FROM model_config_tests
            WHERE model_test_id = ?
            ORDER BY datetime(created_at)
            """,
            (model_test_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM model_config_tests
            ORDER BY datetime(created_at)
            """
        ).fetchall()
    payload = []
    for row in rows:
        entries = conn.execute(
            """
            SELECT json_path, inherit_default, value_text, value_json
            FROM model_config_test_entries
            WHERE config_test_id = ?
            ORDER BY json_path
            """,
            (row["id"],),
        ).fetchall()
        payload.append(
            {
                "id": row["id"],
                "model_test_id": row["model_test_id"],
                "config_type": row["config_type"],
                "name": row["name"],
                "status": row["load_status"],
                "updated_at": row["updated_at"],
                "entries": [
                    {
                        "path": entry["json_path"],
                        "inherit_default": bool(entry["inherit_default"]),
                        "value": entry["value_json"]
                        if entry["value_json"] is not None
                        else entry["value_text"],
                    }
                    for entry in entries
                ],
            }
        )
    print(json.dumps(payload, indent=2))


def update_entry(
    conn: sqlite3.Connection,
    test_id: int,
    json_path: str,
    inherit_default: bool | None,
    value: str | None,
):
    row = conn.execute(
        """
        SELECT * FROM model_config_test_entries
        WHERE config_test_id = ? AND json_path = ?
        """,
        (test_id, json_path),
    ).fetchone()
    if not row:
        raise SystemExit(f"Entry '{json_path}' not found on config test {test_id}")
    updates = []
    params = {}
    if inherit_default is not None:
        updates.append("inherit_default = :inherit_default")
        params["inherit_default"] = 1 if inherit_default else 0
    if value is not None:
        updates.append("value_text = :value_text")
        updates.append("value_json = NULL")
        params["value_text"] = value
    if not updates:
        print("No changes provided.")
        return
    params["id"] = row["id"]
    conn.execute(
        f"""
        UPDATE model_config_test_entries
        SET {", ".join(updates)},
            updated_at = :updated_at
        WHERE id = :id
        """,
        {
            **params,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    conn.commit()
    print(
        f"Updated {json_path} on config test {test_id} "
        f"(inherit_default={inherit_default}, value={value})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CLI helper for model config tests"
    )
    parser.add_argument("--db", dest="db_path", help="Path to tester SQLite db")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List config tests")
    list_cmd.add_argument(
        "--model-test-id", type=int, help="Filter by models_test id"
    )

    set_cmd = sub.add_parser("set", help="Update a config test entry")
    set_cmd.add_argument("--test-id", type=int, required=True)
    set_cmd.add_argument("--path", required=True, help="JSON path")
    set_cmd.add_argument(
        "--inherit",
        choices=["true", "false"],
        help="Toggle inherit default for this entry",
    )
    set_cmd.add_argument(
        "--value",
        help="Override string value (omit to keep current value)",
    )

    args = parser.parse_args()
    db_path = resolve_db_path(args.db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.command == "list":
        list_tests(conn, args.model_test_id)
    elif args.command == "set":
        inherit = None
        if args.inherit is not None:
          inherit = args.inherit == "true"
        update_entry(conn, args.test_id, args.path, inherit, args.value)


if __name__ == "__main__":
    main()
