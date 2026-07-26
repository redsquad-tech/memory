from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import torch

from .categorical import (
    CategoricalAssociativeMemory,
    CategoricalMemoryConfig,
    CategoricalUpdateConfig,
)
from .config import TextEncoderConfig
from .encoding import SignedHashTextEncoder

FORMAT = "assocmem-categorical-bench"
FORMAT_VERSION = 1
SCORE_LOG_FLOOR_NATS = -1024.0 * math.log(2.0)


def _memory_config(raw: dict[str, Any], *, num_classes: int) -> CategoricalMemoryConfig:
    values = dict(raw)
    values["num_classes"] = num_classes
    values["update"] = CategoricalUpdateConfig(**values.pop("update", {}))
    return CategoricalMemoryConfig(**values)


def initialize_catalog(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS targets (
            target_id INTEGER PRIMARY KEY,
            value TEXT NOT NULL UNIQUE COLLATE BINARY
        );
        """
    )
    return connection


def open_catalog_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)


class BenchCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = json.loads((self.path / "model.json").read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT or raw.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported benchmark checkpoint")
        if raw.get("score_log_floor_nats") != SCORE_LOG_FLOOR_NATS:
            raise ValueError("checkpoint score floor differs from protocol")
        self.raw = raw
        self.encoder_config = TextEncoderConfig(**raw["encoder"])
        self.encoder = SignedHashTextEncoder(self.encoder_config)
        if raw.get("encoder_fingerprint") != self.encoder.fingerprint:
            raise ValueError("checkpoint encoder fingerprint mismatch")
        self.num_classes = int(raw["num_classes"])
        if self.num_classes < 0:
            raise ValueError("negative target count")
        catalog_path = self.path / "targets.sqlite3"
        if catalog_path.exists():
            self.catalog = open_catalog_readonly(catalog_path)
            count, minimum, maximum = self.catalog.execute(
                "SELECT COUNT(*), MIN(target_id), MAX(target_id) FROM targets"
            ).fetchone()
            if int(count) != self.num_classes:
                raise ValueError("target registry count mismatch")
            if count and (int(minimum) != 0 or int(maximum) != self.num_classes - 1):
                raise ValueError("target ids are not contiguous")
        elif self.num_classes:
            raise ValueError("checkpoint target registry is missing")
        else:
            self.catalog = sqlite3.connect(":memory:")
            self.catalog.execute(
                "CREATE TABLE targets (target_id INTEGER PRIMARY KEY, value TEXT UNIQUE)"
            )
        self.memory = self._load_memory()

    def close(self) -> None:
        self.catalog.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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

    def _load_memory(self) -> CategoricalAssociativeMemory | None:
        if self.num_classes == 0:
            return None
        config = _memory_config(self.raw["memory"], num_classes=self.num_classes)
        if config.encoder_fingerprint != self.encoder.fingerprint:
            raise ValueError("memory encoder fingerprint mismatch")
        memory = CategoricalAssociativeMemory(config)
        tensors = torch.load(
            self.path / "tensors.pt", map_location="cpu", weights_only=True
        )
        size = int(self.raw["size"])
        class_capacity = int(self.raw["class_capacity"])
        if not 0 <= size <= config.capacity or class_capacity < self.num_classes:
            raise ValueError("invalid checkpoint dimensions")
        expected = {
            "W": ((size, config.dimension), torch.float32),
            "birth_label": ((size,), torch.int64),
            "usage": ((size,), torch.float32),
            "origin_id": ((size,), torch.int64),
            "atom_uid": ((size,), torch.int64),
            "base_counts": ((class_capacity,), torch.int64),
        }
        for name, (shape, dtype) in expected.items():
            value = tensors.get(name)
            if value is None or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"invalid checkpoint tensor {name}")
        memory.expand_classes(self.num_classes, reserve=class_capacity)
        with torch.no_grad():
            memory.W[:size] = tensors["W"]
            memory.birth_label[:size] = tensors["birth_label"]
            memory.usage[:size] = tensors["usage"]
            memory.origin_id[:size] = tensors["origin_id"]
            memory.atom_uid[:size] = tensors["atom_uid"]
            memory.base_counts[:class_capacity] = tensors["base_counts"]
        memory.size = size
        memory.step = int(self.raw["step"])
        memory.revision = int(self.raw["revision"])
        memory.next_atom_uid = int(self.raw["next_atom_uid"])
        memory.base_count_argmax = int(self.raw["base_count_argmax"])
        if size:
            labels = memory.birth_label[:size]
            if int(labels.min()) < 0 or int(labels.max()) >= self.num_classes:
                raise ValueError("atom label outside target registry")
            norms = memory.W[:size].abs().sum(dim=1)
            if not torch.allclose(
                norms,
                torch.full_like(norms, config.key_scale),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("checkpoint violates key norm invariant")
        return memory


def default_model_document(*, capacity: int = 4055) -> dict[str, Any]:
    encoder_config = TextEncoderConfig(dimension=8192, max_features=256)
    encoder = SignedHashTextEncoder(encoder_config)
    memory = CategoricalMemoryConfig(
        dimension=encoder_config.dimension,
        num_classes=1,
        capacity=capacity,
        top_k=16,
        key_nnz=256,
        key_scale=8.0,
        prior_mass=1.0,
        prior_mode="uniform",
        encoder_fingerprint=encoder.fingerprint,
        update=CategoricalUpdateConfig(learning_rate_key=0.2, train_keys=True),
    )
    memory_raw = asdict(memory)
    memory_raw.pop("num_classes")
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "encoder": asdict(encoder_config),
        "encoder_fingerprint": encoder.fingerprint,
        "memory": memory_raw,
        "birth_policy": {"name": "sha256_batch_top_capacity", "seed": 1729},
        "generation": "map",
        "score_log_floor_nats": SCORE_LOG_FLOOR_NATS,
        "num_classes": 0,
        "class_capacity": 0,
        "size": 0,
        "step": 0,
        "revision": 0,
        "next_atom_uid": 0,
        "base_count_argmax": 0,
    }


def write_model_state(
    destination: Path,
    *,
    template: dict[str, Any],
    memory: CategoricalAssociativeMemory,
) -> None:
    document = dict(template)
    document.update(
        {
            "num_classes": memory.config.num_classes,
            "class_capacity": memory._class_capacity,
            "size": memory.size,
            "step": memory.step,
            "revision": memory.revision,
            "next_atom_uid": memory.next_atom_uid,
            "base_count_argmax": memory.base_count_argmax,
        }
    )
    tensors = {
        "W": memory.W[: memory.size].detach().cpu().clone(),
        "birth_label": memory.birth_label[: memory.size].detach().cpu().clone(),
        "usage": memory.usage[: memory.size].detach().cpu().clone(),
        "origin_id": memory.origin_id[: memory.size].detach().cpu().clone(),
        "atom_uid": memory.atom_uid[: memory.size].detach().cpu().clone(),
        "base_counts": memory.base_counts[: memory._class_capacity].detach().cpu().clone(),
    }
    (destination / "model.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(tensors, destination / "tensors.pt")


def checkpoint_content_hash(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for entry in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(entry.relative_to(root).as_posix().encode("utf-8"))
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def new_checkpoint_directory(destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))


def commit_checkpoint(temporary: Path, destination: Path) -> None:
    for entry in temporary.iterdir():
        if entry.is_file():
            with entry.open("rb") as handle:
                os.fsync(handle.fileno())
    os.replace(temporary, destination)


def copy_catalog(source: Path, destination: Path) -> sqlite3.Connection:
    source_catalog = source / "targets.sqlite3"
    if source_catalog.exists():
        shutil.copy2(source_catalog, destination)
    return initialize_catalog(destination)
