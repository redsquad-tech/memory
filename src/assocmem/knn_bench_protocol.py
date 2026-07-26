from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Self

import torch

from .baselines import ExactKNNBaseline
from .bench_checkpoint import (
    commit_checkpoint,
    copy_catalog,
    new_checkpoint_directory,
    open_catalog_readonly,
)
from .bench_protocol import (
    _atomic_predictions,
    _jsonl,
    _validate_exact_keys,
    _validate_identifier,
)
from .config import TextEncoderConfig
from .encoding import SignedHashTextEncoder
from .online import Prediction

FORMAT = "assocmem-exact-knn-bench"
FORMAT_VERSION = 1
CHECKPOINT_RESERVE_BYTES = 2_000_000


class KNNCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw = json.loads((self.path / "model.json").read_text(encoding="utf-8"))
        if (
            self.raw.get("format") != FORMAT
            or self.raw.get("format_version") != FORMAT_VERSION
        ):
            raise ValueError("unsupported exact-kNN checkpoint")
        self.encoder_config = TextEncoderConfig(**self.raw["encoder"])
        self.encoder = SignedHashTextEncoder(self.encoder_config)
        if self.raw["encoder_fingerprint"] != self.encoder.fingerprint:
            raise ValueError("checkpoint encoder fingerprint mismatch")
        self.num_classes = int(self.raw["num_classes"])
        catalog = self.path / "targets.sqlite3"
        if catalog.exists():
            self.catalog = open_catalog_readonly(catalog)
        elif self.num_classes == 0:
            self.catalog = sqlite3.connect(":memory:")
            self.catalog.execute(
                "CREATE TABLE targets (target_id INTEGER PRIMARY KEY, value TEXT UNIQUE)"
            )
        else:
            raise ValueError("target registry is missing")
        count = int(self.catalog.execute("SELECT COUNT(*) FROM targets").fetchone()[0])
        if count != self.num_classes:
            raise ValueError("target registry count mismatch")
        self.model = self._load_model()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.catalog.close()

    def close(self) -> None:
        self.catalog.close()

    def target_id(self, value: str) -> int | None:
        row = self.catalog.execute(
            "SELECT target_id FROM targets WHERE value = ?", (value,)
        ).fetchone()
        return None if row is None else int(row[0])

    def target_value(self, target_id: int) -> str:
        row = self.catalog.execute(
            "SELECT value FROM targets WHERE target_id = ?", (target_id,)
        ).fetchone()
        if row is None:
            raise ValueError("prediction outside target registry")
        return str(row[0])

    def _load_model(self) -> ExactKNNBaseline | None:
        if self.num_classes == 0:
            return None
        size = int(self.raw["size"])
        capacity = max(int(self.raw["capacity"]), size, 1)
        model = ExactKNNBaseline(
            self.encoder_config.dimension,
            self.num_classes,
            capacity,
            top_k=int(self.raw["top_k"]),
            key_scale=float(self.raw["key_scale"]),
            prior_mass=float(self.raw["prior_mass"]),
            key_nnz=int(self.raw["key_nnz"]),
            prior_mode="uniform",
        )
        tensors = torch.load(
            self.path / "tensors.pt", map_location="cpu", weights_only=True
        )
        expected = {
            "keys": ((size, self.encoder_config.dimension), torch.int8),
            "key_nnz": ((size,), torch.int16),
            "targets": ((size,), torch.int64),
            "counts": ((self.num_classes,), torch.int64),
        }
        for name, (shape, dtype) in expected.items():
            value = tensors.get(name)
            if value is None or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"invalid exact-kNN tensor {name}")
        model.keys[:size] = tensors["keys"]
        model.key_nnz[:size] = tensors["key_nnz"]
        model.targets[:size] = tensors["targets"]
        model.counts[:] = tensors["counts"]
        model.size = size
        model.cursor = int(self.raw["cursor"])
        return model


def _examples(path: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, row in _jsonl(path):
        _validate_exact_keys(row, {"id", "input", "target"}, path=path, line_number=line_number)
        identifier = _validate_identifier(row["id"], path=path, line_number=line_number)
        if identifier in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {identifier!r}")
        seen.add(identifier)
        if not isinstance(row["input"], str) or not isinstance(row["target"], str):
            raise TypeError(f"{path}:{line_number}: input and target must be strings")
        if not row["target"]:
            raise ValueError(f"{path}:{line_number}: target must not be empty")
        result.append(row)
    if not result:
        raise ValueError("learn requires at least one example")
    return result


def _new_model(
    checkpoint: KNNCheckpoint,
    *,
    num_classes: int,
    capacity: int,
) -> ExactKNNBaseline:
    raw = checkpoint.raw
    model = ExactKNNBaseline(
        checkpoint.encoder_config.dimension,
        num_classes,
        capacity,
        top_k=int(raw["top_k"]),
        key_scale=float(raw["key_scale"]),
        prior_mass=float(raw["prior_mass"]),
        key_nnz=int(raw["key_nnz"]),
        prior_mode="uniform",
    )
    old = checkpoint.model
    if old is not None:
        if old.size > capacity:
            raise ValueError("budget cannot hold the existing exact-kNN checkpoint")
        model.keys[: old.size] = old.keys[: old.size]
        model.key_nnz[: old.size] = old.key_nnz[: old.size]
        model.targets[: old.size] = old.targets[: old.size]
        model.counts[: checkpoint.num_classes] = old.counts
        model.size = old.size
        model.cursor = old.cursor
    return model


def _write_state(
    destination: Path,
    *,
    template: dict[str, Any],
    model: ExactKNNBaseline,
    encoder: SignedHashTextEncoder,
) -> None:
    raw = dict(template)
    raw.update(
        {
            "num_classes": model.num_classes,
            "capacity": model.capacity,
            "size": model.size,
            "cursor": model.cursor,
            "encoder_fingerprint": encoder.fingerprint,
        }
    )
    (destination / "model.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "keys": model.keys[: model.size].clone(),
            "key_nnz": model.key_nnz[: model.size].clone(),
            "targets": model.targets[: model.size].clone(),
            "counts": model.counts.clone(),
        },
        destination / "tensors.pt",
    )


def run_knn_learn(
    model_in: str | Path,
    examples_path: str | Path,
    model_out: str | Path,
    budget_path: str | Path,
) -> None:
    source = Path(model_in).resolve()
    examples_file = Path(examples_path).resolve()
    destination = Path(model_out).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("model_in and model_out must be disjoint")
    examples = _examples(examples_file)
    budget = json.loads(Path(budget_path).read_text(encoding="utf-8"))
    byte_limit = int(budget["persistent_model_bytes"])
    temporary = new_checkpoint_directory(destination)
    connection: sqlite3.Connection | None = None
    checkpoint: KNNCheckpoint | None = None
    try:
        checkpoint = KNNCheckpoint(source)
        connection = copy_catalog(source, temporary / "targets.sqlite3")
        next_id = int(connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0])
        for row in examples:
            present = connection.execute(
                "SELECT target_id FROM targets WHERE value = ?", (row["target"],)
            ).fetchone()
            if present is None:
                connection.execute(
                    "INSERT INTO targets(target_id, value) VALUES (?, ?)",
                    (next_id, row["target"]),
                )
                next_id += 1
        connection.commit()
        catalog_bytes = (temporary / "targets.sqlite3").stat().st_size
        per_atom = checkpoint.encoder_config.dimension + 2 + 8
        maximum = max(
            1,
            (byte_limit - catalog_bytes - CHECKPOINT_RESERVE_BYTES) // per_atom,
        )
        total = (checkpoint.model.size if checkpoint.model is not None else 0) + len(
            examples
        )
        capacity = min(total, maximum)
        model = _new_model(checkpoint, num_classes=next_id, capacity=capacity)
        dummy = Prediction(torch.empty(0), 0)
        for row in examples:
            target = int(
                connection.execute(
                    "SELECT target_id FROM targets WHERE value = ?", (row["target"],)
                ).fetchone()[0]
            )
            model.observe(
                checkpoint.encoder.encode(row["input"]),
                target,
                dummy,
            )
        connection.close()
        connection = None
        _write_state(
            temporary,
            template=checkpoint.raw,
            model=model,
            encoder=checkpoint.encoder,
        )
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


def run_knn_infer(
    model_path: str | Path,
    requests_path: str | Path,
    predictions_path: str | Path,
) -> None:
    requests = list(_jsonl(Path(requests_path)))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in requests:
        mode = row.get("mode")
        expected = (
            {"id", "mode", "input", "seed", "max_output_tokens"}
            if mode == "generate"
            else {"id", "mode", "input", "value"}
        )
        if mode not in {"generate", "score"}:
            raise ValueError(f"{requests_path}:{line_number}: invalid mode")
        _validate_exact_keys(
            row, expected, path=Path(requests_path), line_number=line_number
        )
        identifier = _validate_identifier(
            row["id"], path=Path(requests_path), line_number=line_number
        )
        if identifier in seen:
            raise ValueError(f"{requests_path}:{line_number}: duplicate id")
        seen.add(identifier)
        if not isinstance(row["input"], str):
            raise TypeError(f"{requests_path}:{line_number}: input must be a string")
        if mode == "score" and not isinstance(row["value"], str):
            raise TypeError(f"{requests_path}:{line_number}: value must be a string")
        rows.append(row)

    def responses():
        with KNNCheckpoint(model_path) as checkpoint:
            cache: dict[str, Prediction | None] = {}
            for row in rows:
                prediction = cache.get(row["input"])
                if row["input"] not in cache:
                    prediction = (
                        None
                        if checkpoint.model is None
                        else checkpoint.model.predict(
                            checkpoint.encoder.encode(row["input"])
                        )
                    )
                    cache[row["input"]] = prediction
                if row["mode"] == "generate":
                    yield {
                        "id": row["id"],
                        "output": (
                            ""
                            if prediction is None
                            else checkpoint.target_value(prediction.prediction)
                        ),
                    }
                    continue
                target = checkpoint.target_id(row["value"])
                probability = (
                    0.0
                    if prediction is None or target is None
                    else float(prediction.probabilities[target])
                )
                yield {
                    "id": row["id"],
                    "log_probability": (
                        None if probability == 0 else float(torch.log(torch.tensor(probability)))
                    ),
                }

    _atomic_predictions(Path(predictions_path), responses())
