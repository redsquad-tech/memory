from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from .encoding import TernaryQuery
from .online import OnlineExample


@dataclass(frozen=True)
class ByteEncoderConfig:
    dimension: int = 16384
    context_length: int = 64
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4, 5, 8)
    max_features: int = 256
    hash_seed: int = 0


class SignedHashByteEncoder:
    def __init__(self, config: ByteEncoderConfig):
        self.config = config
        self._key = config.hash_seed.to_bytes(16, "little", signed=True)
        self.fingerprint = hashlib.sha256(repr(config).encode()).hexdigest()

    def encode(self, context: bytes) -> TernaryQuery:
        context = context[-self.config.context_length :]
        accum: dict[int, int] = {}
        priorities: dict[int, int] = {}
        for order in self.config.ngram_orders:
            if len(context) < order:
                continue
            # Suffix n-grams are causal and position-sensitive through their order.
            feature = bytes([order]) + context[-order:]
            digest = hashlib.blake2b(feature, digest_size=16, key=self._key).digest()
            primary = int.from_bytes(digest[:8], "little")
            priority = int.from_bytes(digest[8:], "little")
            index = primary % self.config.dimension
            sign = 1 if ((primary >> 63) & 1) == 0 else -1
            accum[index] = accum.get(index, 0) + sign
            priorities[index] = min(priority, priorities.get(index, priority))
        live = [(i, 1 if v > 0 else -1, priorities[i]) for i, v in accum.items() if v]
        live.sort(key=lambda item: (item[2], item[0]))
        live = live[: self.config.max_features]
        live.sort(key=lambda item: item[0])
        return TernaryQuery(
            self.config.dimension,
            torch.tensor([x[0] for x in live], dtype=torch.int64),
            torch.tensor([x[1] for x in live], dtype=torch.float32),
            self.fingerprint,
        )


class ByteNextTokenStream:
    def __init__(
        self,
        path: str | Path,
        encoder: SignedHashByteEncoder,
        *,
        start: int,
        end: int,
    ):
        self.path = Path(path)
        self.encoder = encoder
        self.start = start
        self.end = end

    def __iter__(self) -> Iterator[OnlineExample]:
        data = self.path.read_bytes()[self.start : self.end]
        context_length = self.encoder.config.context_length
        for position in range(1, len(data)):
            context = data[max(0, position - context_length) : position]
            yield OnlineExample(
                self.encoder.encode(context),
                data[position],
                example_id=self.start + position,
            )


def byte_split(total_bytes: int) -> dict[str, tuple[int, int]]:
    train_end = int(total_bytes * 0.8)
    validation_end = int(total_bytes * 0.9)
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, total_bytes),
    }
