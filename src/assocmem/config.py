from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class TextEncoderConfig:
    dimension: int = 8192
    hash_seed: int = 0
    hash_version: str = "blake2b-128-v1"
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    max_features: int = 256
    unicode_normalization: str = "NFKC"
    casefold: bool = True

    def __post_init__(self) -> None:
        _positive("dimension", self.dimension)
        _positive("max_features", self.max_features)
        if self.char_ngram_min > self.char_ngram_max:
            raise ValueError("invalid character n-gram range")
        if self.word_ngram_min > self.word_ngram_max:
            raise ValueError("invalid word n-gram range")
        if self.hash_version != "blake2b-128-v1":
            raise ValueError(f"unsupported hash version: {self.hash_version}")


@dataclass(frozen=True)
class UpdateConfig:
    learning_rate_key: float = 0.2
    learning_rate_value: float = 0.5
    gradient_clip_norm: float = 1.0
    value_max_norm: float = 8.0
    train_keys: bool = True
    backtracking_attempts: int = 5
    backtracking_factor: float = 0.5
    acceptance_tolerance_bits: float = 1e-8
    support_replacements: int = 1
    fixed_key_support: bool = False

    def __post_init__(self) -> None:
        _positive("learning_rate_key", self.learning_rate_key)
        _positive("learning_rate_value", self.learning_rate_value)
        _positive("gradient_clip_norm", self.gradient_clip_norm)
        _positive("value_max_norm", self.value_max_norm)
        _positive("backtracking_attempts", self.backtracking_attempts)
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1)")
        if self.support_replacements < 0:
            raise ValueError("support_replacements must be non-negative")


@dataclass(frozen=True)
class InsertionConfig:
    enabled: bool = True
    surprise_margin_from_uniform_bits: float = 2.0
    minimum_surprise_bits: float = 1.0
    min_gain_bits: float = 0.01
    value_scale: float = 2.0
    novelty_energy_ratio: float = 0.25
    min_background_responsibility: float = 0.25
    bootstrap_when_empty: bool = True

    def __post_init__(self) -> None:
        _positive("minimum_surprise_bits", self.minimum_surprise_bits)
        if self.min_gain_bits < 0:
            raise ValueError("min_gain_bits must be non-negative")
        _positive("value_scale", self.value_scale)
        if not 0 <= self.novelty_energy_ratio <= 1:
            raise ValueError("novelty_energy_ratio must be in [0, 1]")
        if not 0 <= self.min_background_responsibility <= 1:
            raise ValueError("min_background_responsibility must be in [0, 1]")

    def surprise_threshold(self, num_classes: int) -> float:
        return max(
            self.minimum_surprise_bits,
            math.log2(num_classes) - self.surprise_margin_from_uniform_bits,
        )


@dataclass(frozen=True)
class EvictionConfig:
    usage_decay: float = 0.999

    def __post_init__(self) -> None:
        if not 0 < self.usage_decay <= 1:
            raise ValueError("usage_decay must be in (0, 1]")


@dataclass(frozen=True)
class MemoryConfig:
    dimension: int
    num_classes: int
    capacity: int
    top_k: int | None = 16
    key_nnz: int = 64
    key_scale: float = 8.0
    prior_pseudocount: float = 1.0
    encoder_fingerprint: str | None = None
    update: UpdateConfig = field(default_factory=UpdateConfig)
    insertion: InsertionConfig = field(default_factory=InsertionConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)

    def __post_init__(self) -> None:
        for name in ("dimension", "num_classes", "capacity", "key_nnz"):
            _positive(name, getattr(self, name))
        if self.key_nnz > self.dimension:
            raise ValueError("key_nnz cannot exceed dimension")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        _positive("key_scale", self.key_scale)
        _positive("prior_pseudocount", self.prior_pseudocount)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MemoryConfig:
        values = dict(raw)
        values["update"] = UpdateConfig(**values.get("update", {}))
        values["insertion"] = InsertionConfig(**values.get("insertion", {}))
        values["eviction"] = EvictionConfig(**values.get("eviction", {}))
        return cls(**values)
