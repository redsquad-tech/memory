"""Legacy in-process classifier experiment helpers.

The behavioral benchmark must not import this module; its public boundary is
the JSONL learn/infer process contract in ``adapters/seqbench``.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from .encoding import TernaryQuery
from .memory import AssociativeMemory


@dataclass(frozen=True)
class OnlineExample:
    query: TernaryQuery
    target: int
    example_id: int = -1
    hidden_state: int | None = None
    true_distribution: torch.Tensor | None = None


@dataclass(frozen=True)
class Prediction:
    probabilities: torch.Tensor
    prediction: int
    token: object | None = None


class OnlineModel(Protocol):
    def predict(self, query: TernaryQuery) -> Prediction: ...

    def observe(
        self,
        query: TernaryQuery,
        target: int,
        prediction: Prediction,
        *,
        example_id: int = -1,
    ) -> object: ...

    def allocated_bytes(self) -> int: ...


class MemoryOnlineModel:
    def __init__(self, memory: AssociativeMemory):
        self.memory = memory

    def predict(self, query: TernaryQuery) -> Prediction:
        read = self.memory.read(query)
        return Prediction(read.probabilities, int(read.prediction.item()), read)

    def observe(
        self,
        query: TernaryQuery,
        target: int,
        prediction: Prediction,
        *,
        example_id: int = -1,
    ) -> object:
        return self.memory.observe(  # type: ignore[arg-type]
            query, target, pre_read=prediction.token, origin_id=example_id
        )

    def allocated_bytes(self) -> int:
        return self.memory.allocated_bytes()


@dataclass
class OnlineMetrics:
    examples: int = 0
    nll_bits_sum: float = 0.0
    correct: int = 0
    brier_sum: float = 0.0
    read_seconds: float = 0.0
    update_seconds: float = 0.0
    num_classes: int | None = None
    latency_window: int = 10_000
    confusion: torch.Tensor = None  # type: ignore[assignment]
    calibration_count: torch.Tensor = None  # type: ignore[assignment]
    calibration_confidence: torch.Tensor = None  # type: ignore[assignment]
    calibration_correct: torch.Tensor = None  # type: ignore[assignment]
    read_latencies: deque[float] = None  # type: ignore[assignment]
    update_latencies: deque[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        classes = self.num_classes or 0
        self.confusion = torch.zeros((classes, classes), dtype=torch.int64)
        self.calibration_count = torch.zeros(15, dtype=torch.int64)
        self.calibration_confidence = torch.zeros(15, dtype=torch.float64)
        self.calibration_correct = torch.zeros(15, dtype=torch.int64)
        self.read_latencies = deque(maxlen=self.latency_window)
        self.update_latencies = deque(maxlen=self.latency_window)

    def update(self, prediction: Prediction, target: int, read_seconds: float) -> None:
        probabilities = prediction.probabilities.detach().cpu()
        if not self.num_classes:
            self.num_classes = int(probabilities.numel())
            self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        probability = float(probabilities[target].clamp_min(torch.finfo(torch.float32).tiny))
        one_hot = torch.zeros_like(probabilities)
        one_hot[target] = 1
        self.examples += 1
        self.nll_bits_sum += -math.log2(probability)
        self.correct += int(prediction.prediction == target)
        self.brier_sum += float(((probabilities - one_hot) ** 2).sum())
        self.read_seconds += read_seconds
        self.confusion[target, prediction.prediction] += 1
        confidence = float(probabilities.max())
        bin_id = min(int(confidence * 15), 14)
        self.calibration_count[bin_id] += 1
        self.calibration_confidence[bin_id] += confidence
        self.calibration_correct[bin_id] += int(prediction.prediction == target)
        self.read_latencies.append(read_seconds)

    def add_update_latency(self, seconds: float) -> None:
        self.update_seconds += seconds
        self.update_latencies.append(seconds)

    def _macro_f1(self) -> float:
        scores = []
        for label in range(self.confusion.shape[0]):
            tp = int(self.confusion[label, label])
            fp = int(self.confusion[:, label].sum()) - tp
            fn = int(self.confusion[label, :].sum()) - tp
            scores.append(2 * tp / max(2 * tp + fp + fn, 1))
        return sum(scores) / len(scores) if scores else 0.0

    def _ece(self, bins: int = 15) -> float:
        total = max(self.examples, 1)
        result = 0.0
        for index in range(bins):
            count = int(self.calibration_count[index])
            if count:
                accuracy = float(self.calibration_correct[index]) / count
                confidence = float(self.calibration_confidence[index]) / count
                result += count / total * abs(accuracy - confidence)
        return result

    @staticmethod
    def _percentile(values: deque[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)]

    def summary(self) -> dict[str, float | int]:
        n = max(self.examples, 1)
        return {
            "examples": self.examples,
            "prequential_nll_bits": self.nll_bits_sum / n,
            "accuracy": self.correct / n,
            "macro_f1": self._macro_f1(),
            "brier": self.brier_sum / n,
            "ece_15": self._ece(),
            "read_seconds": self.read_seconds,
            "update_seconds": self.update_seconds,
            "read_p50_seconds": self._percentile(self.read_latencies, 0.50),
            "read_p95_seconds": self._percentile(self.read_latencies, 0.95),
            "update_p50_seconds": self._percentile(self.update_latencies, 0.50),
            "update_p95_seconds": self._percentile(self.update_latencies, 0.95),
        }


def evaluate_online(
    model: OnlineModel,
    stream: Iterable[OnlineExample],
    *,
    output_dir: str | Path | None = None,
    log_every: int = 1000,
) -> dict[str, float | int]:
    metrics = OnlineMetrics()
    rows: list[dict[str, float | int]] = []
    for example in stream:
        started = time.perf_counter()
        prediction = model.predict(example.query)
        read_seconds = time.perf_counter() - started
        metrics.update(prediction, example.target, read_seconds)
        started = time.perf_counter()
        model.observe(
            example.query,
            example.target,
            prediction,
            example_id=example.example_id,
        )
        metrics.add_update_latency(time.perf_counter() - started)
        if log_every and metrics.examples % log_every == 0:
            rows.append(metrics.summary())
    summary = metrics.summary()
    summary["allocated_bytes"] = model.allocated_bytes()
    memory = getattr(model, "memory", None)
    if memory is not None:
        summary["active_atoms"] = memory.size
        summary["active_logical_bytes"] = memory.active_logical_bytes()
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (destination / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return summary
