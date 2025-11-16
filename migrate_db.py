#!/usr/bin/env python3
"""
Database migration script to add new benchmark and operational metric columns.
Run this once to upgrade your existing database schema.
"""

import sqlite3
import os

# Get database path from environment or use default
db_path = os.getenv("DATABASE_PATH", "data/models.db")

print(f"Migrating database: {db_path}")

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List of new columns to add
new_columns = [
    # Benchmark Metrics
    ("mmlu_score", "REAL"),
    ("gpqa_score", "REAL"),
    ("hellaswag_score", "REAL"),
    ("humaneval_score", "REAL"),
    ("mbpp_score", "REAL"),
    ("math_score", "REAL"),
    ("truthfulqa_score", "REAL"),
    ("perplexity", "REAL"),
    # Operational Metrics
    ("max_throughput_tokens_sec", "REAL"),
    ("avg_latency_ms", "REAL"),
    ("quantization", "TEXT"),
]

# Add each column if it doesn't exist
for column_name, column_type in new_columns:
    try:
        cursor.execute(f"ALTER TABLE modelregistry ADD COLUMN {column_name} {column_type}")
        print(f"✓ Added column: {column_name}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print(f"- Column already exists: {column_name}")
        else:
            print(f"✗ Error adding column {column_name}: {e}")

# Commit changes
conn.commit()
conn.close()

print("\n✅ Database migration completed!")
print("You can now restart your container with: docker-compose restart")
