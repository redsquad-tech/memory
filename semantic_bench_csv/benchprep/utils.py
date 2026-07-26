from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


from .schema import CSV_COLUMNS, TaskRow


def set_max_csv_field_size() -> int:
    """Raise csv's parser limit to the largest value accepted by this platform."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return value
    text = clean_text(value)
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default if default is not None else value


def first_present(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_sequence_of_records(value: Any) -> list[dict[str, Any]]:
    """Normalize Arrow/Pandas representations of Sequence[struct]."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        keys = list(value.keys())
        columns: dict[str, list[Any]] = {}
        size = 0
        for key in keys:
            item = value[key]
            if hasattr(item, "tolist"):
                item = item.tolist()
            if not isinstance(item, list):
                item = [item]
            columns[key] = item
            size = max(size, len(item))
        return [
            {key: columns[key][i] if i < len(columns[key]) else None for key in keys}
            for i in range(size)
        ]
    return []


def iter_parquet_rows(path: Path, batch_size: int = 256) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSON array or JSONL file. Uses ijson when available."""
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None

    with path.open("rb") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == b"[":
            if ijson is not None:
                for item in ijson.items(fh, "item"):
                    if isinstance(item, Mapping):
                        yield dict(item)
                return
            payload = json.load(fh)
            for item in payload:
                if isinstance(item, Mapping):
                    yield dict(item)
            return

        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if isinstance(item, Mapping):
                yield dict(item)


def iter_tsv_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if row and any(cell.strip() for cell in row):
                yield [cell.strip() for cell in row]


def infer_split(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    name = path.stem.lower()
    candidates = [
        ("validation", ("validation", "valid", "dev")),
        ("generalization", ("generalization", "gen")),
        ("test", ("test",)),
        ("train", ("train",)),
    ]
    for normalized, aliases in candidates:
        if any(re.search(rf"(^|[_/.-]){re.escape(alias)}($|[_/.-])", text) for alias in aliases):
            return normalized
    if name.startswith("qa"):
        return "test"
    return "unknown"


def write_rows(path: Path, rows: Iterable[TaskRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())
            count += 1
    return count


def safe_extract_zip(zip_path: Path, output_dir: Path, password: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    pwd = password.encode("utf-8") if password else None
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (output_dir / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe path in zip: {info.filename}")
            archive.extract(info, output_dir, pwd=pwd)
    return output_dir
