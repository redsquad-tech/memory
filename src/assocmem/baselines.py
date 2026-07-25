from __future__ import annotations

import math

import torch

from .encoding import TernaryQuery
from .online import Prediction


class FrequencyBaseline:
    def __init__(self, num_classes: int, pseudocount: float = 1.0):
        self.num_classes = num_classes
        self.pseudocount = pseudocount
        self.counts = torch.zeros(num_classes, dtype=torch.int64)

    def predict(self, query: TernaryQuery) -> Prediction:
        counts = self.counts.to(torch.float64) + self.pseudocount
        probabilities = (counts / counts.sum()).to(torch.float32)
        return Prediction(probabilities, int(probabilities.argmax()))

    def observe(
        self, query: TernaryQuery, target: int, prediction: Prediction, *, example_id: int = -1
    ) -> None:
        self.counts[target] += 1

    def allocated_bytes(self) -> int:
        return self.counts.numel() * self.counts.element_size()


class ExactKNNBaseline:
    def __init__(
        self,
        dimension: int,
        num_classes: int,
        capacity: int,
        *,
        top_k: int = 16,
        key_scale: float = 8.0,
        pseudocount: float = 1.0,
        prior_mass: float = 1.0,
        key_nnz: int | None = None,
    ):
        self.dimension = dimension
        self.num_classes = num_classes
        self.capacity = capacity
        self.top_k = top_k
        self.key_scale = key_scale
        self.pseudocount = pseudocount
        if prior_mass <= 0:
            raise ValueError("prior_mass must be positive")
        if key_nnz is not None and key_nnz <= 0:
            raise ValueError("key_nnz must be positive or None")
        self.prior_mass = prior_mass
        self.maximum_key_nnz = key_nnz
        self.keys = torch.zeros((capacity, dimension), dtype=torch.int8)
        self.key_nnz = torch.zeros(capacity, dtype=torch.int16)
        self.targets = torch.zeros(capacity, dtype=torch.int64)
        self.counts = torch.zeros(num_classes, dtype=torch.int64)
        self.size = 0
        self.cursor = 0

    def predict(self, query: TernaryQuery) -> Prediction:
        base = self.counts.to(torch.float64) + self.pseudocount
        base = base / base.sum()
        if not self.size:
            probabilities = base.to(torch.float32)
            return Prediction(probabilities, int(probabilities.argmax()))
        dot = self.keys[: self.size, query.indices].to(torch.float32) @ query.values
        key_nnz = self.key_nnz[: self.size].clamp_min(1).to(torch.float32)
        normalized_dot = self.key_scale * dot / key_nnz
        energies = 0.5 * (self.key_scale - normalized_dot)
        order = torch.argsort(energies, stable=True)[: min(self.top_k, self.size)]
        weights = torch.exp2(-energies[order])
        votes = torch.zeros(self.num_classes, dtype=torch.float64)
        votes.scatter_add_(0, self.targets[order], weights.to(torch.float64))
        probabilities = (votes + self.prior_mass * base) / (votes.sum() + self.prior_mass)
        probabilities = probabilities.to(torch.float32)
        return Prediction(probabilities, int(probabilities.argmax()))

    def observe(
        self, query: TernaryQuery, target: int, prediction: Prediction, *, example_id: int = -1
    ) -> None:
        slot = self.size if self.size < self.capacity else self.cursor
        self.keys[slot].zero_()
        limit = query.nnz if self.maximum_key_nnz is None else min(query.nnz, self.maximum_key_nnz)
        support = query.indices[:limit]
        self.keys[slot, support] = query.values[:limit].to(torch.int8)
        self.key_nnz[slot] = limit
        self.targets[slot] = target
        if self.size < self.capacity:
            self.size += 1
        else:
            self.cursor = (self.cursor + 1) % self.capacity
        self.counts[target] += 1

    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.keys, self.key_nnz, self.targets, self.counts)
        )


class CentroidBaseline:
    def __init__(self, dimension: int, num_classes: int, pseudocount: float = 1.0):
        self.centroids = torch.zeros((num_classes, dimension), dtype=torch.float32)
        self.counts = torch.zeros(num_classes, dtype=torch.int64)
        self.pseudocount = pseudocount

    def predict(self, query: TernaryQuery) -> Prediction:
        scores = self.centroids[:, query.indices] @ query.values if query.nnz else self.counts * 0
        prior = (self.counts.to(torch.float32) + self.pseudocount).log()
        probabilities = torch.softmax(scores / math.sqrt(max(query.nnz, 1)) + prior, dim=0)
        return Prediction(probabilities, int(probabilities.argmax()))

    def observe(
        self, query: TernaryQuery, target: int, prediction: Prediction, *, example_id: int = -1
    ) -> None:
        count = int(self.counts[target])
        row = self.centroids[target]
        row.mul_(count / (count + 1))
        row[query.indices].add_(query.values / (count + 1))
        self.counts[target] += 1

    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (self.centroids, self.counts)
        )


class OnlineLinearBaseline:
    def __init__(self, dimension: int, num_classes: int, learning_rate: float = 0.1):
        self.weights = torch.zeros((num_classes, dimension), dtype=torch.float32)
        self.bias = torch.zeros(num_classes, dtype=torch.float32)
        self.learning_rate = learning_rate

    def predict(self, query: TernaryQuery) -> Prediction:
        logits = self.bias.clone()
        if query.nnz:
            logits += self.weights[:, query.indices] @ query.values
        probabilities = torch.softmax(logits, dim=0)
        return Prediction(probabilities, int(probabilities.argmax()))

    def observe(
        self, query: TernaryQuery, target: int, prediction: Prediction, *, example_id: int = -1
    ) -> None:
        gradient = prediction.probabilities.clone()
        gradient[target] -= 1
        self.bias.add_(gradient, alpha=-self.learning_rate)
        if query.nnz:
            self.weights[:, query.indices] -= (
                self.learning_rate * gradient[:, None] * query.values[None, :]
            )

    def allocated_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.weights, self.bias))


class OnlineMLPBaseline:
    def __init__(
        self, dimension: int, num_classes: int, hidden_size: int = 128, learning_rate: float = 0.05
    ):
        generator = torch.Generator().manual_seed(0)
        self.first = torch.randn((hidden_size, dimension), generator=generator) * 0.01
        self.second = torch.randn((num_classes, hidden_size), generator=generator) * 0.01
        self.bias = torch.zeros(num_classes)
        self.learning_rate = learning_rate

    def _forward(self, query: TernaryQuery) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = (
            self.first[:, query.indices] @ query.values
            if query.nnz
            else torch.zeros(self.first.shape[0])
        )
        hidden = torch.relu(hidden_pre)
        return hidden, self.second @ hidden + self.bias

    def predict(self, query: TernaryQuery) -> Prediction:
        hidden, logits = self._forward(query)
        probabilities = torch.softmax(logits, dim=0)
        return Prediction(probabilities, int(probabilities.argmax()), hidden)

    def observe(
        self, query: TernaryQuery, target: int, prediction: Prediction, *, example_id: int = -1
    ) -> None:
        hidden = prediction.token
        assert isinstance(hidden, torch.Tensor)
        gradient_logits = prediction.probabilities.clone()
        gradient_logits[target] -= 1
        old_second = self.second.clone()
        self.second -= self.learning_rate * gradient_logits[:, None] * hidden[None, :]
        self.bias -= self.learning_rate * gradient_logits
        if query.nnz:
            gradient_hidden = old_second.T @ gradient_logits
            gradient_hidden *= hidden > 0
            self.first[:, query.indices] -= (
                self.learning_rate * gradient_hidden[:, None] * query.values[None, :]
            )

    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.first, self.second, self.bias)
        )
