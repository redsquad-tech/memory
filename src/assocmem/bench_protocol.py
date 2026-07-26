from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .bench_checkpoint import (
    BenchCheckpoint,
    commit_checkpoint,
    copy_catalog,
    new_checkpoint_directory,
    write_model_state,
)
from .categorical import CategoricalAssociativeMemory


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            yield line_number, value


def _validate_exact_keys(
    row: dict[str, Any], expected: set[str], *, path: Path, line_number: int
) -> None:
    if set(row) != expected:
        raise ValueError(
            f"{path}:{line_number}: expected fields {sorted(expected)}, got {sorted(row)}"
        )


def _validate_identifier(value: Any, *, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
    return value


def _birth_ids(path: Path, capacity: int, seed: int) -> set[str]:
    heap: list[tuple[int, str]] = []
    seen: set[str] = set()
    examples = 0
    for line_number, row in _jsonl(path):
        _validate_exact_keys(row, {"id", "input", "target"}, path=path, line_number=line_number)
        example_id = _validate_identifier(row["id"], path=path, line_number=line_number)
        if example_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {example_id!r}")
        seen.add(example_id)
        if not isinstance(row["input"], str) or not isinstance(row["target"], str):
            raise TypeError(f"{path}:{line_number}: input and target must be strings")
        if not row["target"]:
            raise ValueError(f"{path}:{line_number}: target must not be empty")
        rank = int.from_bytes(
            hashlib.sha256(f"{seed}:{example_id}".encode()).digest(), "big"
        )
        item = (-rank, example_id)
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
        examples += 1
    if examples == 0:
        raise ValueError("learn requires at least one example")
    return {example_id for _, example_id in heap}


def run_learn(
    model_in: str | Path,
    examples_path: str | Path,
    model_out: str | Path,
    *,
    seed: int | None = None,
) -> None:
    source = Path(model_in).resolve()
    examples = Path(examples_path).resolve()
    destination = Path(model_out).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("model_in and model_out must be disjoint")
    temporary = new_checkpoint_directory(destination)
    connection: sqlite3.Connection | None = None
    checkpoint: BenchCheckpoint | None = None
    try:
        checkpoint = BenchCheckpoint(source)
        policy = checkpoint.raw["birth_policy"]
        births = _birth_ids(
            examples,
            int(checkpoint.raw["memory"]["capacity"]),
            int(policy["seed"]) if seed is None else seed,
        )
        connection = copy_catalog(source, temporary / "targets.sqlite3")
        next_id = int(connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0])
        for line_number, row in _jsonl(examples):
            target = row["target"]
            present = connection.execute(
                "SELECT target_id FROM targets WHERE value = ?", (target,)
            ).fetchone()
            if present is None:
                connection.execute(
                    "INSERT INTO targets(target_id, value) VALUES (?, ?)", (next_id, target)
                )
                next_id += 1
        connection.commit()
        num_classes = next_id
        if checkpoint.memory is None:
            from .bench_checkpoint import _memory_config

            memory = CategoricalAssociativeMemory(
                _memory_config(checkpoint.raw["memory"], num_classes=num_classes)
            )
        else:
            memory = checkpoint.memory
            memory.expand_classes(num_classes, reserve=max(16, 2 ** (num_classes - 1).bit_length()))
        for _, row in _jsonl(examples):
            target_id = int(
                connection.execute(
                    "SELECT target_id FROM targets WHERE value = ?", (row["target"],)
                ).fetchone()[0]
            )
            query = checkpoint.encoder.encode(row["input"])
            memory.observe_compact(
                query,
                target_id,
                origin_id=memory.step,
                insertion_mode="force" if row["id"] in births else "skip",
            )
        connection.commit()
        connection.close()
        connection = None
        write_model_state(temporary, template=checkpoint.raw, memory=memory)
        checkpoint.close()
        checkpoint = None
        commit_checkpoint(temporary, destination)
    except BaseException:
        if connection is not None:
            connection.close()
        if checkpoint is not None:
            checkpoint.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _atomic_predictions(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_infer(
    model_path: str | Path,
    requests_path: str | Path,
    predictions_path: str | Path,
) -> None:
    model = Path(model_path).resolve()
    requests = Path(requests_path).resolve()
    predictions = Path(predictions_path).resolve()
    if predictions == model or model in predictions.parents:
        raise ValueError("predictions path must be outside checkpoint")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in _jsonl(requests):
        mode = row.get("mode")
        expected = (
            {"id", "mode", "input", "seed", "max_output_tokens"}
            if mode == "generate"
            else {"id", "mode", "input", "value"}
        )
        if mode not in {"generate", "score"}:
            raise ValueError(f"{requests}:{line_number}: invalid mode")
        _validate_exact_keys(row, expected, path=requests, line_number=line_number)
        request_id = _validate_identifier(
            row["id"], path=requests, line_number=line_number
        )
        if request_id in seen:
            raise ValueError(f"{requests}:{line_number}: duplicate id {request_id!r}")
        seen.add(request_id)
        if not isinstance(row["input"], str):
            raise TypeError(f"{requests}:{line_number}: input must be a string")
        if mode == "generate":
            if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
                raise ValueError(f"{requests}:{line_number}: seed must be an integer")
            if (
                not isinstance(row["max_output_tokens"], int)
                or isinstance(row["max_output_tokens"], bool)
                or row["max_output_tokens"] <= 0
            ):
                raise ValueError(
                    f"{requests}:{line_number}: max_output_tokens must be positive"
                )
        elif not isinstance(row["value"], str):
            raise ValueError(f"{requests}:{line_number}: value must be a string")
        validated.append(row)

    def predictions_iter() -> Iterator[dict[str, Any]]:
        with BenchCheckpoint(model) as checkpoint:
            responses: list[dict[str, Any] | None] = [None] * len(validated)
            groups: dict[str, list[int]] = {}
            for index, row in enumerate(validated):
                groups.setdefault(row["input"], []).append(index)
            for input_text, indices in groups.items():
                known: list[tuple[int, int]] = []
                for index in indices:
                    row = validated[index]
                    if row["mode"] == "score":
                        target = checkpoint.target_id(row["value"])
                        if target is None:
                            responses[index] = {
                                "id": row["id"],
                                "log_probability": None,
                            }
                        else:
                            known.append((index, target))
                if checkpoint.memory is None:
                    prediction = None
                    logs: list[float] = []
                else:
                    prediction, tensor = checkpoint.memory.predict_and_log_probabilities(
                        checkpoint.encoder.encode(input_text),
                        [target for _, target in known],
                    )
                    logs = [float(value) for value in tensor]
                log_cursor = 0
                for index in indices:
                    row = validated[index]
                    if row["mode"] == "generate":
                        responses[index] = {
                            "id": row["id"],
                            "output": (
                                "" if prediction is None else checkpoint.target_value(prediction)
                            ),
                        }
                    elif responses[index] is None:
                        responses[index] = {
                            "id": row["id"],
                            "log_probability": logs[log_cursor],
                        }
                        log_cursor += 1
            for response in responses:
                assert response is not None
                yield response

    _atomic_predictions(predictions, predictions_iter())
