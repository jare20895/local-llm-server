#!/usr/bin/env python3
import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from typing import Any, Dict, List, Optional, Tuple

import yaml
from huggingface_hub import HfApi, hf_hub_download


try:
    LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
except Exception:
    LOCAL_TZ = timezone(timedelta(hours=-5))


def current_timestamp() -> str:
    return datetime.now(LOCAL_TZ).replace(microsecond=0).isoformat()


DEFAULT_FLAG_MAP = {
    "YAML Key": (1, 0, 0),
    "YAML List": (1, 1, 0),
    "YAML NestedMapping": (0, 1, 1),
    "Markdown H1": (1, 0, 0),
    "Markdown H2": (1, 0, 0),
    "Markdown Paragraph": (0, 1, 1),
    "Markdown Link": (1, 0, 0),
    "Model Card Section": (0, 1, 1),
}

SECTION_RULES = [
    ("training data", "Training", "Training_Data"),
    ("datasets", "Training", "Training_Data"),
    ("intended use", "Usage", "Intended_Use"),
    ("usage", "Usage", "Usage"),
    ("how to use", "Usage", "Usage"),
    ("limitations", "Ethics & Safety", "Limitations"),
    ("risks", "Ethics & Safety", "Safety_Ethics"),
    ("safety", "Ethics & Safety", "Safety_Ethics"),
    ("responsible", "Ethics & Safety", "Safety_Ethics"),
    ("evaluation", "Performance", "Evaluation_Results"),
    ("benchmarks", "Performance", "Performance"),
    ("results", "Performance", "Performance"),
    ("performance", "Performance", "Performance"),
    ("license", "Metadata", "License"),
    ("summary", "Model Overview", "Model_Summary"),
    ("overview", "Model Overview", "Model_Summary"),
    ("model card", "Model Overview", "Model_Summary"),
    ("deployment", "Deployment", "Deployment_Notes"),
    ("environment", "Deployment", "Deployment_Notes"),
]

YAML_RULES = [
    ("license", "Metadata", "License"),
    ("tags", "Metadata", "Tags"),
    ("pipeline_tag", "Metadata", "Tags"),
    ("language", "Localization", "Localization"),
    ("languages", "Localization", "Localization"),
    ("inference.parameters", "Usage", "Parameters"),
    ("inference", "Usage", "Usage"),
    ("datasets", "Training", "Training_Data"),
    ("dataset", "Training", "Training_Data"),
    ("safety", "Ethics & Safety", "Safety_Ethics"),
    ("evaluation", "Performance", "Evaluation_Results"),
    ("author", "Metadata", "Author_Attribution"),
    ("organization", "Metadata", "Author_Attribution"),
    ("hardware", "Deployment", "Deployment_Notes"),
    ("gpu", "Deployment", "Deployment_Notes"),
]

LINK_RULES = [
    ("arxiv", "Model Overview", "Technical_Report"),
    ("paper", "Model Overview", "Technical_Report"),
    ("huggingface", "Model Overview", "Model_Summary"),
]

CRITICAL_SEMANTIC_ROLES = {
    "License",
    "Tags",
    "Parameters",
    "Usage",
    "Intended_Use",
    "Deployment_Notes",
    "Performance",
    "Evaluation_Results",
    "Localization",
    "Safety_Ethics",
    "Model_Summary",
}

CRITICAL_CATEGORIES = {
    "Metadata",
    "Usage",
    "Deployment",
    "Performance",
    "Localization",
}

CRITICAL_PATH_KEYWORDS = [
    "inference",
    "parameter",
    "hardware",
    "gpu",
    "system",
    "environment",
    "cache",
]


@dataclass
class Entry:
    canonical_path: str
    display_path: Optional[str]
    element_type: str
    value_text: Optional[str]
    value_json: Optional[str]
    parent_path: Optional[str]
    model_card_path: str
    model_card_section: Optional[str]
    source_line: Optional[int]
    category: Optional[str]
    semantic_role: Optional[str]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "root"


def truncate_value(value: Optional[str], length: int = 800) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if len(value) <= length:
        return value
    return value[: length - 3] + "..."


def infer_semantics_from_yaml(canonical_path: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = canonical_path.lower()
    for key, category, semantic in YAML_RULES:
        if key in lowered:
            return category, semantic
    return None, None


def infer_semantics_from_section(title: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = title.lower()
    for token, category, semantic in SECTION_RULES:
        if token in lowered:
            return category, semantic
    return None, None


def infer_semantics_for_link(text: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    haystack = f"{text} {url}".lower()
    for token, category, semantic in LINK_RULES:
        if token in haystack:
            return category, semantic
    return None, None


def extract_front_matter(readme_text: str) -> Tuple[Optional[str], Optional[int], str, int]:
    lines = readme_text.splitlines()
    if not lines:
        return None, None, readme_text, 1
    if lines[0].strip() != "---":
        return None, None, readme_text, 1
    closing_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_index = idx
            break
    if closing_index is None:
        return None, None, readme_text, 1
    front_lines = lines[1:closing_index]
    rest_lines = lines[closing_index + 1 :]
    front_text = "\n".join(front_lines)
    front_start_line = 2  # first line after initial ---
    rest_start_line = closing_index + 2
    return front_text, front_start_line, "\n".join(rest_lines), rest_start_line


def format_display_path(tokens: List[str]) -> str:
    lines = []
    for idx, token in enumerate(tokens):
        indent = "  " * idx
        suffix = ":" if idx < len(tokens) - 1 else ""
        lines.append(f"{indent}{token}{suffix}")
    return "\n".join(lines)


def build_yaml_entries(
    node: yaml.Node,
    py_value: Any,
    base_canonical: List[str],
    display_tokens: List[str],
    start_line: int,
    entries: List[Entry],
    section_label: str,
):
    if not isinstance(node, yaml.MappingNode):
        return
    for key_node, value_node in node.value:
        key_name = key_node.value
        canonical_tokens = base_canonical + [key_name]
        display_tokens_key = display_tokens + [key_name]
        line_number = start_line + key_node.start_mark.line
        parent_path = ".".join(base_canonical) if base_canonical else None
        if isinstance(value_node, yaml.MappingNode):
            category, semantic = infer_semantics_from_yaml(".".join(canonical_tokens))
            entries.append(
                Entry(
                    canonical_path=".".join(canonical_tokens),
                    display_path=format_display_path(display_tokens_key),
                    element_type="YAML NestedMapping",
                    value_text=truncate_value(json.dumps(py_value.get(key_name, {}), ensure_ascii=False)),
                    value_json=json.dumps(py_value.get(key_name, {}), ensure_ascii=False)
                    if isinstance(py_value.get(key_name, {}), (dict, list))
                    else None,
                    parent_path=parent_path,
                    model_card_path="README.md",
                    model_card_section=section_label,
                    source_line=line_number,
                    category=category,
                    semantic_role=semantic,
                )
            )
            child_value = py_value.get(key_name, {})
            build_yaml_entries(
                value_node,
                child_value if isinstance(child_value, dict) else {},
                canonical_tokens,
                display_tokens_key,
                start_line,
                entries,
                section_label,
            )
        elif isinstance(value_node, yaml.SequenceNode):
            seq_canonical = canonical_tokens + ["[]"]
            seq_display = display_tokens_key + ["-"]
            category, semantic = infer_semantics_from_yaml(".".join(canonical_tokens))
            iterable = py_value.get(key_name, [])
            if not isinstance(iterable, list):
                iterable = []
            for idx, item_node in enumerate(value_node.value):
                item_value = iterable[idx] if idx < len(iterable) else None
                entry_value = (
                    json.dumps(item_value, ensure_ascii=False)
                    if isinstance(item_value, (dict, list))
                    else (str(item_value) if item_value is not None else None)
                )
                entries.append(
                    Entry(
                        canonical_path=".".join(seq_canonical),
                        display_path=format_display_path(seq_display),
                        element_type="YAML List",
                        value_text=truncate_value(entry_value),
                        value_json=json.dumps(item_value, ensure_ascii=False)
                        if isinstance(item_value, (dict, list))
                        else None,
                        parent_path=".".join(canonical_tokens),
                        model_card_path="README.md",
                        model_card_section=section_label,
                        source_line=start_line + item_node.start_mark.line,
                        category=category,
                        semantic_role=semantic,
                    )
                )
                if isinstance(item_node, yaml.MappingNode):
                    child_value = item_value if isinstance(item_value, dict) else {}
                    build_yaml_entries(
                        item_node,
                        child_value,
                        canonical_tokens + ["[]"],
                        display_tokens_key + [f"- item {idx + 1}"],
                        start_line,
                        entries,
                        section_label,
                    )
        else:
            node_value = py_value.get(key_name)
            category, semantic = infer_semantics_from_yaml(".".join(canonical_tokens))
            entries.append(
                Entry(
                    canonical_path=".".join(canonical_tokens),
                    display_path=format_display_path(display_tokens_key),
                    element_type="YAML Key",
                    value_text=truncate_value(str(node_value)) if node_value is not None else None,
                    value_json=None,
                    parent_path=parent_path,
                    model_card_path="README.md",
                    model_card_section=section_label,
                    source_line=line_number,
                    category=category,
                    semantic_role=semantic,
                )
            )


def parse_yaml_front_matter(front_text: str, start_line: int) -> List[Entry]:
    entries: List[Entry] = []
    if not front_text.strip():
        return entries
    try:
        node = yaml.compose(front_text)
        py_value = yaml.safe_load(front_text) or {}
    except Exception:
        return entries
    if isinstance(node, yaml.MappingNode):
        build_yaml_entries(
            node,
            py_value if isinstance(py_value, dict) else {},
            [],
            [],
            start_line,
            entries,
            "front_matter",
        )
    return entries


def parse_markdown(markdown_text: str, start_line: int) -> List[Entry]:
    entries: List[Entry] = []
    lines = markdown_text.splitlines()
    current_section: Optional[Dict[str, Any]] = None
    paragraph_lines: List[str] = []
    paragraph_start: Optional[int] = None

    def flush_paragraph():
        nonlocal paragraph_lines, paragraph_start
        if paragraph_lines:
            content = " ".join(paragraph_lines).strip()
            if content:
                section_title = current_section["title"] if current_section else "root"
                category, semantic = (
                    infer_semantics_from_section(section_title)
                    if section_title != "root"
                    else (None, None)
                )
                slug = slugify(section_title)
                entries.append(
                    Entry(
                        canonical_path=f"markdown.paragraph.{slug}",
                        display_path=section_title,
                        element_type="Markdown Paragraph",
                        value_text=truncate_value(content),
                        value_json=None,
                        parent_path=None,
                        model_card_path="README.md",
                        model_card_section=section_title,
                        source_line=paragraph_start,
                        category=category,
                        semantic_role=semantic,
                    )
                )
            paragraph_lines = []
            paragraph_start = None

    def flush_section_summary():
        nonlocal current_section
        if current_section:
            content = "\n".join(current_section["lines"]).strip()
            section_title = current_section["title"]
            if content:
                slug = slugify(section_title)
                category, semantic = infer_semantics_from_section(section_title)
                entries.append(
                    Entry(
                        canonical_path=f"markdown.section.{slug}",
                        display_path=section_title,
                        element_type="Model Card Section",
                        value_text=truncate_value(content),
                        value_json=None,
                        parent_path=None,
                        model_card_path="README.md",
                        model_card_section=section_title,
                        source_line=current_section["start_line"],
                        category=category,
                        semantic_role=semantic,
                    )
                )
            current_section = None

    for idx, line in enumerate(lines, start=start_line):
        stripped = line.strip()
        link_matches = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", line)
        if link_matches:
            section_title = current_section["title"] if current_section else "root"
            slug = slugify(section_title)
            for text, url in link_matches:
                category, semantic = infer_semantics_for_link(text, url)
                entries.append(
                    Entry(
                        canonical_path=f"markdown.link.{slug}",
                        display_path=text,
                        element_type="Markdown Link",
                        value_text=f"{text} -> {url}",
                        value_json=json.dumps({"text": text, "url": url}),
                        parent_path=None,
                        model_card_path="README.md",
                        model_card_section=section_title,
                        source_line=idx,
                        category=category,
                        semantic_role=semantic,
                    )
                )
        if stripped.startswith("# "):
            flush_paragraph()
            flush_section_summary()
            heading = stripped[2:].strip()
            entries.append(
                Entry(
                    canonical_path=f"markdown.h1.{slugify(heading)}",
                    display_path=f"# {heading}",
                    element_type="Markdown H1",
                    value_text=truncate_value(heading),
                    value_json=None,
                    parent_path=None,
                    model_card_path="README.md",
                    model_card_section=heading,
                    source_line=idx,
                    category="Model Overview",
                    semantic_role="Model_Name",
                )
            )
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_section_summary()
            section_title = stripped[3:].strip()
            category, semantic = infer_semantics_from_section(section_title)
            entries.append(
                Entry(
                    canonical_path=f"markdown.h2.{slugify(section_title)}",
                    display_path=f"## {section_title}",
                    element_type="Markdown H2",
                    value_text=truncate_value(section_title),
                    value_json=None,
                    parent_path=None,
                    model_card_path="README.md",
                    model_card_section=section_title,
                    source_line=idx,
                    category=category,
                    semantic_role=semantic,
                )
            )
            current_section = {"title": section_title, "start_line": idx, "lines": []}
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            if current_section:
                current_section["lines"].append(line)
            continue
        if not stripped:
            flush_paragraph()
            if current_section:
                current_section["lines"].append("")
            continue
        if paragraph_start is None:
            paragraph_start = idx
        paragraph_lines.append(stripped)
        if current_section:
            current_section["lines"].append(line)

    flush_paragraph()
    flush_section_summary()
    return entries


def ensure_meta(conn: sqlite3.Connection, entry: Entry) -> Tuple[sqlite3.Row, bool]:
    row = conn.execute(
        "SELECT * FROM huggingface_meta WHERE canonical_path = ?",
        (entry.canonical_path,),
    ).fetchone()
    now = current_timestamp()
    example_value = entry.value_text
    parent_path = entry.parent_path
    description = row["description"] if row else None
    category = entry.category
    semantic = entry.semantic_role
    critical = is_critical_entry(entry)
    if row:
        conn.execute(
            """
            UPDATE huggingface_meta
            SET display_path = COALESCE(?, display_path),
                element_type = ?,
                category = COALESCE(?, category),
                semantic_role = COALESCE(?, semantic_role),
                example_value = COALESCE(?, example_value),
                parent_path = COALESCE(?, parent_path),
                updated_at = ?
            WHERE id = ?
            """,
            (
                entry.display_path,
                entry.element_type,
                category,
                semantic,
                row["example_value"] or example_value,
                parent_path,
                now,
                row["id"],
            ),
        )
        fresh = conn.execute(
            "SELECT * FROM huggingface_meta WHERE id = ?", (row["id"],)
        ).fetchone()
        return fresh, False
    active, detailed, extensive = DEFAULT_FLAG_MAP.get(
        entry.element_type, (1, 0, 0)
    )
    if critical:
        active = 1
    cursor = conn.execute(
        """
        INSERT INTO huggingface_meta
          (canonical_path, display_path, element_type, description, category,
           semantic_role, example_value, active, detailed, extensive, parent_path, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.canonical_path,
            entry.display_path,
            entry.element_type,
            description,
            category,
            semantic,
            example_value,
            active,
            detailed,
            extensive,
            parent_path,
            now,
        ),
    )
    fresh = conn.execute(
        "SELECT * FROM huggingface_meta WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    return fresh, True


def is_critical_entry(entry: Entry) -> bool:
    canonical = entry.canonical_path.lower()
    if entry.semantic_role and entry.semantic_role in CRITICAL_SEMANTIC_ROLES:
        return True
    if entry.category and entry.category in CRITICAL_CATEGORIES:
        return True
    return any(keyword in canonical for keyword in CRITICAL_PATH_KEYWORDS)


def should_capture(meta_row: sqlite3.Row, detail_level: str) -> bool:
    if meta_row["active"] != 1:
        return False
    if meta_row["extensive"] == 1 and detail_level != "extensive":
        return False
    if meta_row["detailed"] == 1 and detail_level == "basic":
        return False
    return True


def persist_entries(
    conn: sqlite3.Connection,
    model_test_id: int,
    entries: List[Entry],
    detail_level: str,
) -> Tuple[int, int, int, int]:
    existing_rows = conn.execute(
        "SELECT * FROM models_test_huggingface WHERE model_test_id = ?",
        (model_test_id,),
    ).fetchall()
    existing_by_meta = {row["meta_id"]: row for row in existing_rows}
    seen_meta_ids = set()
    inserted = 0
    updated = 0
    meta_updates = 0

    now_iso = current_timestamp()

    for entry in entries:
        meta_row, created_meta = ensure_meta(conn, entry)
        if created_meta:
            meta_updates += 1
        if not should_capture(meta_row, detail_level):
            continue

        seen_meta_ids.add(meta_row["id"])
        existing = existing_by_meta.get(meta_row["id"])

        payload = {
            "canonical_path": entry.canonical_path,
            "display_path": entry.display_path,
            "element_type": entry.element_type,
            "value_text": entry.value_text,
            "value_json": entry.value_json,
            "model_card_path": entry.model_card_path,
            "model_card_section": entry.model_card_section,
            "source_line": entry.source_line,
        }

        if existing:
            changed = any(
                existing[key] != payload[key]
                for key in payload
            )
            if changed:
                conn.execute(
                    """
                    UPDATE models_test_huggingface
                    SET canonical_path = ?,
                        display_path = ?,
                        element_type = ?,
                        value_text = ?,
                        value_json = ?,
                        model_card_path = ?,
                        model_card_section = ?,
                        source_line = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        payload["canonical_path"],
                        payload["display_path"],
                        payload["element_type"],
                        payload["value_text"],
                        payload["value_json"],
                        payload["model_card_path"],
                        payload["model_card_section"],
                        payload["source_line"],
                        now_iso,
                        existing["id"],
                    ),
                )
                updated += 1
        else:
            conn.execute(
                """
                INSERT INTO models_test_huggingface
                  (model_test_id, meta_id, canonical_path, display_path, element_type,
                   value_text, value_json, model_card_path, model_card_section, source_line,
                   detected_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_test_id,
                    meta_row["id"],
                    payload["canonical_path"],
                    payload["display_path"],
                    payload["element_type"],
                    payload["value_text"],
                    payload["value_json"],
                    payload["model_card_path"],
                    payload["model_card_section"],
                    payload["source_line"],
                    now_iso,
                    now_iso,
                    now_iso,
                ),
            )
            inserted += 1

    stale_meta_ids = [
        meta_id
        for meta_id in existing_by_meta.keys()
        if meta_id not in seen_meta_ids
    ]
    if stale_meta_ids:
        placeholders = ",".join("?" for _ in stale_meta_ids)
        conn.execute(
            f"""
            DELETE FROM models_test_huggingface
            WHERE model_test_id = ? AND meta_id IN ({placeholders})
            """,
            (model_test_id, *stale_meta_ids),
        )

    pruned_cursor = conn.execute(
        """
        DELETE FROM models_test_huggingface
        WHERE meta_id IN (
          SELECT id FROM huggingface_meta WHERE active = 0
        )
        """
    )
    pruned = pruned_cursor.rowcount
    conn.commit()
    return inserted, updated, pruned, meta_updates


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


def load_model_card(repo_id: str, token: Optional[str]) -> str:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            token=token,
            repo_type="model",
        )
        with open(local_path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to download README.md for {repo_id}: {exc}") from exc


def collect_entries(readme_text: str) -> List[Entry]:
    front_matter, front_start, markdown_text, markdown_start = extract_front_matter(
        readme_text
    )
    entries: List[Entry] = []
    if front_matter:
        entries.extend(parse_yaml_front_matter(front_matter, front_start or 1))
    entries.extend(parse_markdown(markdown_text, markdown_start))
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync HuggingFace metadata into Vector Tester database"
    )
    parser.add_argument("--db", type=str, required=False, default=None, help="Path to vector tester SQLite DB")
    parser.add_argument("--model-test-id", type=int, required=True, help="models_test row id to sync")
    parser.add_argument("--model-name", type=str, required=False, help="Model name override")
    parser.add_argument("--hf-path", type=str, required=False, help="Explicit HuggingFace repo id")
    parser.add_argument(
        "--detail-level",
        type=str,
        choices=["basic", "detailed", "extensive"],
        default="basic",
        help="Control which tiers of metadata are captured",
    )
    return parser.parse_args()


def resolve_db_path(user_path: Optional[str]) -> str:
    if user_path:
        return user_path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "vector-tester.db")


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
    try:
        readme_text = load_model_card(repo_id, token)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    entries = collect_entries(readme_text)
    inserted, updated_entries, pruned, meta_updates = persist_entries(
        conn, args.model_test_id, entries, args.detail_level
    )
    summary = {
        "model_test_id": args.model_test_id,
        "model_name": args.model_name or row["model_name"],
        "repo_id": repo_id,
        "total_entries": inserted + updated_entries,
        "new_entries": inserted,
        "updated_entries": updated_entries,
        "new_meta": meta_updates,
        "pruned_inactive": pruned,
        "hf_card_sha": info.sha,
        "detail_level": args.detail_level,
        "timestamp": current_timestamp(),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
