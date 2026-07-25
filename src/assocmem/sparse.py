from __future__ import annotations

from dataclasses import dataclass

import torch

from .encoding import TernaryQuery


@dataclass(frozen=True)
class SparseScores:
    slot_ids: torch.Tensor
    scores: torch.Tensor
    energies: torch.Tensor


class SparseExactIndex:
    """Dynamic exact inverted index for fixed-L1 sparse keys.

    The implementation is deliberately CPU-oriented and independent from
    torch.sparse autograd. Selected rows can be materialized densely for local
    learning and written back with ``set_key``.
    """

    def __init__(self, dimension: int, capacity: int, key_nnz: int, key_scale: float):
        self.dimension = dimension
        self.capacity = capacity
        self.key_nnz = key_nnz
        self.key_scale = key_scale
        self.indices = torch.full((capacity, key_nnz), -1, dtype=torch.int32)
        self.values = torch.zeros((capacity, key_nnz), dtype=torch.float32)
        self.lengths = torch.zeros(capacity, dtype=torch.int16)
        self.postings: dict[int, dict[int, float]] = {}
        self.size = 0

    def _remove(self, slot: int) -> None:
        length = int(self.lengths[slot])
        for feature in self.indices[slot, :length].tolist():
            bucket = self.postings[int(feature)]
            bucket.pop(slot, None)
            if not bucket:
                del self.postings[int(feature)]

    def set_key(self, slot: int, key: torch.Tensor) -> None:
        if key.shape != (self.dimension,):
            raise ValueError("invalid dense key shape")
        if not 0 <= slot < self.capacity:
            raise IndexError(slot)
        if slot < self.size:
            self._remove(slot)
        nonzero = torch.nonzero(key, as_tuple=False).flatten()
        if not 0 < nonzero.numel() <= self.key_nnz:
            raise ValueError("key violates sparsity invariant")
        if not torch.isclose(
            key.abs().sum(), torch.tensor(self.key_scale, dtype=key.dtype), atol=1e-5
        ):
            raise ValueError("key violates L1 invariant")
        order = torch.argsort(nonzero, stable=True)
        nonzero = nonzero[order]
        length = nonzero.numel()
        self.indices[slot].fill_(-1)
        self.values[slot].zero_()
        self.indices[slot, :length] = nonzero.to(torch.int32)
        self.values[slot, :length] = key[nonzero]
        self.lengths[slot] = length
        for feature, value in zip(nonzero.tolist(), key[nonzero].tolist(), strict=True):
            self.postings.setdefault(int(feature), {})[slot] = float(value)
        self.size = max(self.size, slot + 1)

    def dense_key(self, slot: int) -> torch.Tensor:
        result = torch.zeros(self.dimension, dtype=torch.float32)
        length = int(self.lengths[slot])
        result[self.indices[slot, :length].to(torch.int64)] = self.values[slot, :length]
        return result

    def score(self, query: TernaryQuery, top_k: int | None = None) -> SparseScores:
        if query.dimension != self.dimension:
            raise ValueError("dimension mismatch")
        scores = torch.zeros(self.size, dtype=torch.float32)
        for feature, sign in zip(query.indices.tolist(), query.values.tolist(), strict=True):
            for slot, value in self.postings.get(int(feature), {}).items():
                scores[slot] += float(sign) * value
        order = torch.argsort(scores, descending=True, stable=True)
        k = self.size if top_k is None else min(top_k, self.size)
        slots = order[:k]
        selected_scores = scores[slots]
        return SparseScores(
            slots,
            selected_scores,
            0.5 * (self.key_scale - selected_scores),
        )

    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.indices, self.values, self.lengths)
        )
