from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import torch

from .config import TextEncoderConfig

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class TernaryQuery:
    dimension: int
    indices: torch.Tensor
    values: torch.Tensor
    encoder_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.indices.ndim != 1 or self.values.ndim != 1:
            raise ValueError("indices and values must be one-dimensional")
        if self.indices.numel() != self.values.numel():
            raise ValueError("indices and values must have equal length")
        if self.indices.dtype != torch.int64:
            raise TypeError("indices must use torch.int64")
        if self.values.dtype != torch.float32:
            raise TypeError("values must use torch.float32")
        if self.indices.numel():
            if int(self.indices.min()) < 0 or int(self.indices.max()) >= self.dimension:
                raise ValueError("query index outside dimension")
            if not bool(torch.all((self.values == -1) | (self.values == 1))):
                raise ValueError("query values must be -1 or +1")
            if not bool(torch.all(self.indices[1:] > self.indices[:-1])):
                raise ValueError("query indices must be sorted and unique")

    @property
    def nnz(self) -> int:
        return self.indices.numel()

    @property
    def device(self) -> torch.device:
        return self.values.device

    def to(self, device: torch.device | str) -> TernaryQuery:
        return TernaryQuery(
            self.dimension,
            self.indices.to(device=device),
            self.values.to(device=device),
            self.encoder_fingerprint,
        )

    def snapshot(self, *, device: torch.device | str | None = None) -> TernaryQuery:
        target = self.device if device is None else torch.device(device)
        return TernaryQuery(
            self.dimension,
            self.indices.detach().to(target).clone(),
            self.values.detach().to(target).clone(),
            self.encoder_fingerprint,
        )

    def to_dense(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.device
        result = torch.zeros(self.dimension, dtype=torch.float32, device=target)
        if self.nnz:
            result[self.indices.to(target)] = self.values.to(target)
        return result


class SignedHashTextEncoder:
    def __init__(self, config: TextEncoderConfig):
        self.config = config
        self._key = config.hash_seed.to_bytes(16, "little", signed=True)
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        payload = repr(self.config).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _digest(self, feature: str) -> tuple[int, int]:
        raw = hashlib.blake2b(feature.encode("utf-8"), digest_size=16, key=self._key).digest()
        return int.from_bytes(raw[:8], "little"), int.from_bytes(raw[8:], "little")

    def _features(self, text: str) -> set[str]:
        normalized = unicodedata.normalize(self.config.unicode_normalization, text)
        if self.config.casefold:
            normalized = normalized.casefold()
        chars = f"^{normalized}$"
        result: set[str] = set()
        for n in range(self.config.char_ngram_min, self.config.char_ngram_max + 1):
            result.update(f"c{n}:{chars[i : i + n]}" for i in range(max(0, len(chars) - n + 1)))
        words = ["<s>", *_WORD_RE.findall(normalized), "</s>"]
        for n in range(self.config.word_ngram_min, self.config.word_ngram_max + 1):
            result.update(
                f"w{n}:{' '.join(words[i : i + n])}" for i in range(max(0, len(words) - n + 1))
            )
        return result

    def encode(self, text: str, *, task_id: str | None = None) -> TernaryQuery:
        accum: dict[int, int] = {}
        priorities: dict[int, int] = {}
        prefix = f"task:{task_id}|" if task_id is not None else ""
        for feature in self._features(text):
            primary, priority = self._digest(prefix + feature)
            index = primary % self.config.dimension
            sign = 1 if ((primary >> 63) & 1) == 0 else -1
            accum[index] = accum.get(index, 0) + sign
            priorities[index] = min(priority, priorities.get(index, priority))
        live = [
            (idx, 1 if total > 0 else -1, priorities[idx]) for idx, total in accum.items() if total
        ]
        if len(live) > self.config.max_features:
            live.sort(key=lambda item: (item[2], item[0]))
            live = live[: self.config.max_features]
        live.sort(key=lambda item: item[0])
        indices = torch.tensor([item[0] for item in live], dtype=torch.int64)
        values = torch.tensor([item[1] for item in live], dtype=torch.float32)
        return TernaryQuery(self.config.dimension, indices, values, self.fingerprint)
