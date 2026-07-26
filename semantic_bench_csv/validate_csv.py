#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import tempfile
from pathlib import Path

from benchprep.schema import CSV_COLUMNS
from benchprep.utils import set_max_csv_field_size

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "tasks.csv"
ALLOWED_SPLITS = {"train", "validation", "test", "generalization"}
ALLOWED_VARIANTS = {"full", "oracle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a generated task CSV.")
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_CSV)
    return parser.parse_args()


def validate(path: Path) -> tuple[int, list[str]]:
    set_max_csv_field_size()
    errors: list[str] = []
    count = 0
    handle = tempfile.NamedTemporaryFile(prefix="semantic-bench-ids-", suffix=".sqlite", delete=False)
    database_path = Path(handle.name)
    handle.close()
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE task_ids (id TEXT PRIMARY KEY) WITHOUT ROWID;
            """
        )
        cursor = connection.cursor()
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != CSV_COLUMNS:
                return 0, [f"columns differ: {reader.fieldnames!r}"]
            for line_no, row in enumerate(reader, start=2):
                count += 1
                row_id = row["id"]
                if not row_id:
                    errors.append(f"line {line_no}: empty id")
                else:
                    try:
                        cursor.execute("INSERT INTO task_ids(id) VALUES (?)", (row_id,))
                    except sqlite3.IntegrityError:
                        errors.append(f"line {line_no}: duplicate id {row_id}")
                for field in ("dataset", "config", "split", "task", "variant", "input", "expected_output"):
                    if not row[field]:
                        errors.append(f"line {line_no}: empty {field}")
                if row["split"] not in ALLOWED_SPLITS:
                    errors.append(f"line {line_no}: invalid split {row['split']!r}")
                if row["variant"] not in ALLOWED_VARIANTS:
                    errors.append(f"line {line_no}: invalid variant {row['variant']!r}")
                try:
                    probability = float(row["expected_probability"])
                    if not math.isfinite(probability) or not 0 < probability <= 1:
                        raise ValueError
                except ValueError:
                    errors.append(f"line {line_no}: invalid expected_probability")
                for field in ("answer_candidates_json", "metadata_json"):
                    if row[field]:
                        try:
                            json.loads(row[field])
                        except json.JSONDecodeError as exc:
                            errors.append(f"line {line_no}: invalid {field}: {exc}")
        if count == 0:
            errors.append("CSV contains no task rows")
        return count, errors
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not args.path.is_file():
        raise SystemExit(f"CSV file does not exist: {args.path}")
    count, errors = validate(args.path)
    if errors:
        print(f"FAIL {args.path} ({count:,} rows)")
        for error in errors[:20]:
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more")
        return 1
    print(f"OK   {args.path} ({count:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
