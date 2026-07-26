#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sqlite3
import sys
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from benchprep.adapters import ADAPTERS, semantic_parsing_row
from benchprep.schema import CSV_COLUMNS, TaskRow
from benchprep.sources import (
    SourceSpec,
    discover_sources,
    iter_semantic_fields,
    iter_structured_records,
)
from download_datasets import (
    DEFAULT_MANIFEST,
    DEFAULT_RAW_DIR,
    comma_values,
    ensure_datasets,
    load_manifest,
    selected_names,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "tasks.csv"
ALLOWED_VARIANTS = {"full", "oracle"}
ALLOWED_SPLITS = {"train", "validation", "test", "generalization"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and atomically build one normalized benchmark CSV."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default="all", help="Comma-separated names or 'all'.")
    parser.add_argument("--variants", default="full,oracle")
    parser.add_argument("--mrcr-needles", default="2needle,4needle,8needle")
    parser.add_argument("--babilong-lengths", default="0k,1k,2k,4k,8k,16k")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def validate_variants(value: str) -> list[str]:
    variants = comma_values(value)
    unknown = sorted(set(variants) - ALLOWED_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return [variant for variant in ("full", "oracle") if variant in variants]


def expected_task_count(
    dataset: str,
    sources: list[SourceSpec],
    variants: set[str],
    *,
    mrcr_needles: list[str],
) -> int:
    if dataset == "babi":
        return 1_040_000 * len(variants & {"full", "oracle"})
    if dataset == "clutrr":
        return 70_631 * len(variants & {"full", "oracle"})
    if "full" not in variants:
        return 0
    if dataset == "mrcr":
        return 800 * len(mrcr_needles)
    if dataset == "babilong":
        return 100 * len(sources)
    return {
        "proofwriter": 845_496,
        "recogs": 1_102_402,
        "slog": 115_694,
    }[dataset]


def converted_rows(source: SourceSpec, variants: set[str]) -> Iterator[TaskRow]:
    if source.dataset in {"recogs", "slog"}:
        if "full" not in variants:
            return
        for index, fields in enumerate(iter_semantic_fields(source)):
            row = semantic_parsing_row(
                dataset=source.dataset,
                config=source.config,
                split=source.split,
                source_index=index,
                fields=fields,
                source_path=source.source_key,
            )
            if row is not None:
                yield row
        return

    adapter = ADAPTERS[source.dataset]
    for index, record in enumerate(iter_structured_records(source)):
        yield from adapter.convert(
            record,
            config=source.config,
            split=source.split,
            source_key=source.source_key,
            source_index=index,
            variants=variants,
        )


def validate_task(row: TaskRow, requested_variants: set[str]) -> None:
    required = {
        "id": row.id,
        "dataset": row.dataset,
        "config": row.config,
        "split": row.split,
        "task": row.task,
        "variant": row.variant,
        "input": row.input,
        "expected_output": row.expected_output,
    }
    empty = [name for name, value in required.items() if not str(value).strip()]
    if empty:
        raise ValueError(f"Task {row.id or '<no-id>'} has empty fields: {', '.join(empty)}")
    if row.split not in ALLOWED_SPLITS:
        raise ValueError(f"Task {row.id} has unsupported split {row.split!r}")
    if row.variant not in requested_variants:
        raise ValueError(f"Task {row.id} emitted unrequested variant {row.variant!r}")
    probability = float(row.expected_probability)
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError(f"Task {row.id} has invalid expected probability {probability!r}")


def _temporary_paths(output: Path) -> tuple[Path, Path]:
    token = uuid.uuid4().hex[:10]
    return (
        output.with_name(f".{output.name}.partial-{token}"),
        output.with_name(f".{output.name}.ids-{token}.sqlite"),
    )


def build_tasks(
    *,
    output: Path,
    names: list[str],
    sources_by_dataset: dict[str, list[SourceSpec]],
    variants: list[str],
    mrcr_needles: list[str],
    enforce_expected_counts: bool = True,
) -> tuple[int, Counter[str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial, id_database = _temporary_paths(output)
    requested_variants = set(variants)
    counts: Counter[str] = Counter()
    connection: sqlite3.Connection | None = None
    completed = False
    try:
        connection = sqlite3.connect(id_database)
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA locking_mode=EXCLUSIVE;
            CREATE TABLE task_ids (
                id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        cursor = connection.cursor()
        with partial.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=CSV_COLUMNS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for dataset in names:
                sources = sources_by_dataset[dataset]
                dataset_count = 0
                for source in sources:
                    source_count = 0
                    for row in converted_rows(source, requested_variants):
                        validate_task(row, requested_variants)
                        try:
                            cursor.execute(
                                "INSERT INTO task_ids(id, source_key) VALUES (?, ?)",
                                (row.id, source.source_key),
                            )
                        except sqlite3.IntegrityError as exc:
                            previous = cursor.execute(
                                "SELECT source_key FROM task_ids WHERE id = ?", (row.id,)
                            ).fetchone()
                            raise ValueError(
                                f"Duplicate task id {row.id}: "
                                f"{previous[0] if previous else '<unknown>'} and {source.source_key}"
                            ) from exc
                        writer.writerow(row.to_csv_dict())
                        source_count += 1
                    if source_count == 0:
                        raise ValueError(f"Source emitted zero tasks: {source.source_key}")
                    dataset_count += source_count
                if enforce_expected_counts:
                    expected = expected_task_count(
                        dataset,
                        sources,
                        requested_variants,
                        mrcr_needles=mrcr_needles,
                    )
                    if dataset_count != expected:
                        raise ValueError(
                            f"{dataset} emitted {dataset_count:,} tasks; expected {expected:,}"
                        )
                counts[dataset] = dataset_count
                print(f"[{dataset}] {len(sources)} sources -> {dataset_count:,} tasks")
            connection.commit()
            fh.flush()
            os.fsync(fh.fileno())
        connection.close()
        connection = None
        os.replace(partial, output)
        completed = True
        return sum(counts.values()), counts
    finally:
        if connection is not None:
            connection.close()
        id_database.unlink(missing_ok=True)
        if not completed:
            partial.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    names = selected_names(args.datasets, manifest)
    variants = validate_variants(args.variants)
    mrcr_needles = comma_values(args.mrcr_needles)
    babilong_lengths = comma_values(args.babilong_lengths)

    ensure_datasets(
        manifest=manifest,
        names=names,
        raw_dir=args.raw_dir,
        mrcr_needles=mrcr_needles,
        babilong_lengths=babilong_lengths,
        force_download=args.force_download,
    )

    sources_by_dataset: dict[str, list[SourceSpec]] = {}
    for name in names:
        sources_by_dataset[name] = discover_sources(
            name,
            args.raw_dir / name,
            mrcr_needles=mrcr_needles,
            babilong_lengths=babilong_lengths,
        )

    total, _ = build_tasks(
        output=args.output,
        names=names,
        sources_by_dataset=sources_by_dataset,
        variants=variants,
        mrcr_needles=mrcr_needles,
    )
    print(f"wrote {total:,} tasks atomically to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
