from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
import statistics
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch

from ..baselines import ExactKNNBaseline, OnlineLinearBaseline
from ..config import InsertionConfig, MemoryConfig, TextEncoderConfig, UpdateConfig
from ..encoding import SignedHashTextEncoder, TernaryQuery
from ..memory import AssociativeMemory, ObserveReport
from ..online import MemoryOnlineModel, OnlineMetrics
from .datasets import iter_banking77

PROTOCOL_VERSION = "banking77-v2-posthoc"
BYTE_BUDGET = 128 * 1024**2
SHARED_BIRTH_SEED = 1729
OFFICIAL_TEST_PREVIOUSLY_EXPOSED = True

MODEL_ORDER = (
    "shared_learned",
    "shared_frozen",
    "natural_learned",
    "natural_frozen",
    "knn",
    "linear",
)
MODEL_LABELS = {
    "shared_learned": "Shared-birth learned-key memory",
    "shared_frozen": "Shared-birth frozen-key memory",
    "natural_learned": "Natural-insertion learned-key memory",
    "natural_frozen": "Natural-insertion frozen-key memory",
    "knn": "Exact same-energy kNN",
    "linear": "Online linear",
}
COMPARISONS = (
    ("shared_keys", "shared_learned", "shared_frozen"),
    ("natural_keys", "natural_learned", "natural_frozen"),
    ("natural_vs_knn", "natural_learned", "knn"),
    ("natural_vs_linear", "natural_learned", "linear"),
)
METRIC_FIELDS = (
    "prequential_nll_bits",
    "prequential_accuracy_pct",
    "test_nll_bits",
    "test_accuracy_pct",
    "test_macro_f1_pct",
    "test_brier",
    "test_dedup_nll_bits",
    "test_dedup_accuracy_pct",
    "test_dedup_macro_f1_pct",
    "test_dedup_brier",
    "insertion_rate_pct",
    "active_atoms",
    "retained_examples",
    "allocated_bytes",
    "model_mib",
    "run_seconds",
)
RAW_FIELDS = (
    "protocol_version",
    "model",
    "model_label",
    "seed",
    *METRIC_FIELDS,
    "train_examples",
    "test_examples",
    "test_dedup_examples",
    "excluded_duplicate_test_examples",
    "dimension",
    "max_features",
    "hash_seed",
    "capacity",
    "top_k",
    "key_nnz",
    "key_scale",
    "learning_rate_key",
    "learning_rate_value",
    "linear_learning_rate",
    "knn_prior_mass",
    "byte_budget",
    "shared_birth_seed",
    "shared_birth_count",
    "fixed_key_support",
    "official_test_previously_exposed",
    "hyperparameters_provisional",
)
SUMMARY_FIELDS = (
    "row_type",
    "model",
    "model_label",
    "comparison",
    "left_model",
    "right_model",
    "seeds",
    *METRIC_FIELDS,
    "prequential_nll_advantage_bits",
    "prequential_nll_advantage_ci95",
    "prequential_nll_favorable_seeds",
    "test_nll_advantage_bits",
    "test_nll_advantage_ci95",
    "test_nll_favorable_seeds",
    "test_accuracy_advantage_pp",
    "test_accuracy_advantage_ci95",
    "test_accuracy_favorable_seeds",
    "protocol_version",
    "official_test_previously_exposed",
    "hyperparameters_provisional",
)


@dataclass(frozen=True)
class EncodedExample:
    query: TernaryQuery
    target: int
    example_id: int


@dataclass(frozen=True)
class PaperExperimentConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    dimension: int = 8192
    max_features: int = 256
    capacity: int = 4055
    top_k: int = 16
    key_nnz: int = 256
    key_scale: float = 8.0
    learning_rate_key: float = 0.2
    learning_rate_value: float = 0.5
    linear_learning_rate: float = 0.03
    knn_prior_mass: float = 1.0
    byte_budget: int = BYTE_BUDGET
    shared_birth_seed: int = SHARED_BIRTH_SEED
    torch_threads: int = 8


def memory_allocated_bytes(config: PaperExperimentConfig, num_classes: int) -> int:
    per_atom = 4 * config.dimension + 4 * num_classes + 4 + 8 + 8
    return config.capacity * per_atom + 8 * num_classes


def knn_capacity_for_budget(
    config: PaperExperimentConfig, num_classes: int, train_examples: int
) -> int:
    per_example = config.dimension + 2 + 8
    available = max(config.byte_budget - 8 * num_classes, 0)
    return min(train_examples, available // per_example)


def _build_model(
    name: str,
    *,
    encoder: SignedHashTextEncoder,
    num_classes: int,
    train_examples: int,
    config: PaperExperimentConfig,
) -> object:
    if name.startswith(("shared_", "natural_")):
        learned = name.endswith("learned")
        update = UpdateConfig(
            learning_rate_key=config.learning_rate_key,
            learning_rate_value=config.learning_rate_value,
            train_keys=learned,
            support_replacements=0,
            fixed_key_support=True,
        )
        memory = AssociativeMemory(
            MemoryConfig(
                dimension=config.dimension,
                num_classes=num_classes,
                capacity=config.capacity,
                top_k=config.top_k,
                key_nnz=config.key_nnz,
                key_scale=config.key_scale,
                encoder_fingerprint=encoder.fingerprint,
                update=update,
                insertion=InsertionConfig(
                    surprise_margin_from_uniform_bits=2.0,
                    minimum_surprise_bits=1.0,
                    min_gain_bits=0.01,
                    value_scale=2.0,
                    novelty_energy_ratio=0.25,
                    min_background_responsibility=0.25,
                ),
            )
        )
        return MemoryOnlineModel(memory)
    if name == "knn":
        return ExactKNNBaseline(
            config.dimension,
            num_classes,
            knn_capacity_for_budget(config, num_classes, train_examples),
            top_k=config.top_k,
            key_scale=config.key_scale,
            prior_mass=config.knn_prior_mass,
            key_nnz=config.key_nnz,
        )
    if name == "linear":
        return OnlineLinearBaseline(
            config.dimension,
            num_classes,
            learning_rate=config.linear_learning_rate,
        )
    raise ValueError(f"unknown paper model: {name}")


def _encode(records: list[tuple[str, int]], encoder: SignedHashTextEncoder) -> list[EncodedExample]:
    return [
        EncodedExample(encoder.encode(text, task_id="banking77"), target, example_id)
        for example_id, (text, target) in enumerate(records)
    ]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\W+", " ", normalized, flags=re.UNICODE).strip()


def duplicate_test_ids(
    train_records: list[tuple[str, int]], test_records: list[tuple[str, int]]
) -> set[int]:
    train_text = {_normalize_text(text) for text, _ in train_records}
    return {
        example_id
        for example_id, (text, _) in enumerate(test_records)
        if _normalize_text(text) in train_text
    }


def select_shared_birth_ids(
    examples: list[EncodedExample], count: int, birth_seed: int
) -> set[int]:
    def rank(example: EncodedExample) -> bytes:
        return hashlib.sha256(f"{birth_seed}:{example.example_id}".encode("ascii")).digest()

    return {
        example.example_id for example in sorted(examples, key=rank)[: min(count, len(examples))]
    }


def _evaluate_test(
    model: object,
    examples: list[EncodedExample],
    num_classes: int,
) -> dict[str, float | int]:
    metrics = OnlineMetrics(num_classes=num_classes)
    for example in examples:
        started = time.perf_counter()
        prediction = model.predict(example.query)  # type: ignore[attr-defined]
        metrics.update(prediction, example.target, time.perf_counter() - started)
    return metrics.summary()


def _run_model(
    name: str,
    model: object,
    train: list[EncodedExample],
    test: list[EncodedExample],
    dedup_test: list[EncodedExample],
    num_classes: int,
    shared_birth_ids: set[int],
) -> dict[str, float | int]:
    started_run = time.perf_counter()
    metrics = OnlineMetrics(num_classes=num_classes)
    insertions = 0
    common_counts = torch.zeros(num_classes, dtype=torch.float64)
    for example in train:
        started = time.perf_counter()
        prediction = model.predict(example.query)  # type: ignore[attr-defined]
        metrics.update(prediction, example.target, time.perf_counter() - started)
        started = time.perf_counter()
        if name.startswith("shared_"):
            reference = (common_counts + 1.0) / (common_counts.sum() + num_classes)
            force = example.example_id in shared_birth_ids
            report = model.memory.observe(  # type: ignore[attr-defined]
                example.query,
                example.target,
                pre_read=prediction.token,
                origin_id=example.example_id,
                insertion_mode="force" if force else "skip",
                insertion_reference_probabilities=(reference.to(torch.float32) if force else None),
            )
            common_counts[example.target] += 1
        else:
            report = model.observe(  # type: ignore[attr-defined]
                example.query,
                example.target,
                prediction,
                example_id=example.example_id,
            )
        metrics.add_update_latency(time.perf_counter() - started)
        if isinstance(report, ObserveReport):
            insertions += int(report.insertion.inserted)

    train_summary = metrics.summary()
    test_summary = _evaluate_test(model, test, num_classes)
    dedup_summary = _evaluate_test(model, dedup_test, num_classes)
    memory = getattr(model, "memory", None)
    retained = getattr(model, "size", None)
    if retained is None:
        retained = memory.size if memory is not None else 0
    return {
        "prequential_nll_bits": float(train_summary["prequential_nll_bits"]),
        "prequential_accuracy_pct": 100.0 * float(train_summary["accuracy"]),
        "test_nll_bits": float(test_summary["prequential_nll_bits"]),
        "test_accuracy_pct": 100.0 * float(test_summary["accuracy"]),
        "test_macro_f1_pct": 100.0 * float(test_summary["macro_f1"]),
        "test_brier": float(test_summary["brier"]),
        "test_dedup_nll_bits": float(dedup_summary["prequential_nll_bits"]),
        "test_dedup_accuracy_pct": 100.0 * float(dedup_summary["accuracy"]),
        "test_dedup_macro_f1_pct": 100.0 * float(dedup_summary["macro_f1"]),
        "test_dedup_brier": float(dedup_summary["brier"]),
        "insertion_rate_pct": (
            100.0 * insertions / max(len(train), 1) if memory is not None else math.nan
        ),
        "active_atoms": int(memory.size) if memory is not None else 0,
        "retained_examples": int(retained),
        "allocated_bytes": int(model.allocated_bytes()),  # type: ignore[attr-defined]
        "model_mib": float(model.allocated_bytes()) / 1024**2,  # type: ignore[attr-defined]
        "run_seconds": time.perf_counter() - started_run,
    }


def _mean_std(values: list[float], digits: int = 6) -> str:
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} ± {deviation:.{digits}f}"


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(len(values), 1.96)
    margin = critical * standard_error
    return mean, mean - margin, mean + margin


def _empty_summary_row() -> dict[str, str]:
    return {field: "" for field in SUMMARY_FIELDS}


def aggregate_rows(
    per_seed: list[dict[str, float | int | str]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        rows = [row for row in per_seed if row["model"] == model]
        if not rows:
            raise ValueError(f"missing results for model {model}")
        aggregate = _empty_summary_row()
        aggregate.update(
            {
                "row_type": "model",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seeds": str(len(rows)),
                "protocol_version": PROTOCOL_VERSION,
                "official_test_previously_exposed": "true",
                "hyperparameters_provisional": "true",
            }
        )
        for field in METRIC_FIELDS:
            values = [float(row[field]) for row in rows if not math.isnan(float(row[field]))]
            aggregate[field] = _mean_std(values) if values else ""
        result.append(aggregate)

    by_model_seed = {(str(row["model"]), int(row["seed"])): row for row in per_seed}
    for comparison, left, right in COMPARISONS:
        seeds = sorted(
            {int(row["seed"]) for row in per_seed if row["model"] == left}
            & {int(row["seed"]) for row in per_seed if row["model"] == right}
        )
        row = _empty_summary_row()
        row.update(
            {
                "row_type": "comparison",
                "comparison": comparison,
                "left_model": left,
                "right_model": right,
                "seeds": str(len(seeds)),
                "protocol_version": PROTOCOL_VERSION,
                "official_test_previously_exposed": "true",
                "hyperparameters_provisional": "true",
            }
        )
        specifications = (
            (
                "prequential_nll_advantage",
                "prequential_nll_bits",
                lambda a, b: b - a,
            ),
            ("test_nll_advantage", "test_nll_bits", lambda a, b: b - a),
            (
                "test_accuracy_advantage",
                "test_accuracy_pct",
                lambda a, b: a - b,
            ),
        )
        for output, metric, advantage in specifications:
            differences = [
                advantage(
                    float(by_model_seed[(left, seed)][metric]),
                    float(by_model_seed[(right, seed)][metric]),
                )
                for seed in seeds
            ]
            mean, low, high = paired_interval(differences)
            suffix = "pp" if output == "test_accuracy_advantage" else "bits"
            row[f"{output}_{suffix}"] = f"{mean:.6f}"
            row[f"{output}_ci95"] = f"[{low:.6f}, {high:.6f}]"
            favorable_field = output.replace("_advantage", "") + "_favorable_seeds"
            row[favorable_field] = str(sum(value > 0 for value in differences))
        result.append(row)
    return result


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...] = SUMMARY_FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_paper_experiment(
    *,
    train_records: list[tuple[str, int]] | None = None,
    test_records: list[tuple[str, int]] | None = None,
    data_dir: str | Path = "data",
    output: str | Path = "banking77_results_v2.csv",
    raw_output: str | Path = "banking77_runs_v2.csv",
    config: PaperExperimentConfig | None = None,
) -> list[dict[str, str]]:
    config = config or PaperExperimentConfig()
    torch.set_num_threads(config.torch_threads)
    if train_records is None:
        train_records = list(iter_banking77(data_dir, "train"))
    if test_records is None:
        test_records = list(iter_banking77(data_dir, "test"))
    labels = {target for _, target in train_records + test_records}
    if not labels or min(labels) != 0 or max(labels) + 1 != len(labels):
        raise ValueError("class labels must be contiguous and start at zero")
    num_classes = len(labels)
    if memory_allocated_bytes(config, num_classes) > config.byte_budget:
        raise ValueError("configured memory exceeds the byte budget")

    encoder = SignedHashTextEncoder(
        TextEncoderConfig(
            dimension=config.dimension,
            max_features=config.max_features,
            hash_seed=0,
        )
    )
    encoded_train = _encode(train_records, encoder)
    encoded_test = _encode(test_records, encoder)
    excluded_ids = duplicate_test_ids(train_records, test_records)
    dedup_test = [example for example in encoded_test if example.example_id not in excluded_ids]
    shared_birth_ids = select_shared_birth_ids(
        encoded_train, config.capacity, config.shared_birth_seed
    )

    raw_rows: list[dict[str, float | int | str]] = []
    total = len(config.seeds) * len(MODEL_ORDER)
    completed = 0
    for seed in config.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        stream = list(encoded_train)
        random.Random(seed).shuffle(stream)
        for model_name in MODEL_ORDER:
            completed += 1
            print(
                f"[{completed:02d}/{total}] seed={seed} model={MODEL_LABELS[model_name]}",
                flush=True,
            )
            model = _build_model(
                model_name,
                encoder=encoder,
                num_classes=num_classes,
                train_examples=len(encoded_train),
                config=config,
            )
            measurements = _run_model(
                model_name,
                model,
                stream,
                encoded_test,
                dedup_test,
                num_classes,
                shared_birth_ids,
            )
            row: dict[str, float | int | str] = {
                "protocol_version": PROTOCOL_VERSION,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "seed": seed,
                **measurements,
                "train_examples": len(encoded_train),
                "test_examples": len(encoded_test),
                "test_dedup_examples": len(dedup_test),
                "excluded_duplicate_test_examples": len(excluded_ids),
                "dimension": config.dimension,
                "max_features": config.max_features,
                "hash_seed": 0,
                "capacity": (
                    knn_capacity_for_budget(config, num_classes, len(encoded_train))
                    if model_name == "knn"
                    else (config.capacity if model_name.startswith(("shared_", "natural_")) else 0)
                ),
                "top_k": config.top_k,
                "key_nnz": config.key_nnz,
                "key_scale": config.key_scale,
                "learning_rate_key": config.learning_rate_key,
                "learning_rate_value": config.learning_rate_value,
                "linear_learning_rate": config.linear_learning_rate,
                "knn_prior_mass": config.knn_prior_mass,
                "byte_budget": config.byte_budget,
                "shared_birth_seed": config.shared_birth_seed,
                "shared_birth_count": len(shared_birth_ids),
                "fixed_key_support": (
                    "true" if model_name.startswith(("shared_", "natural_")) else "not_applicable"
                ),
                "official_test_previously_exposed": str(OFFICIAL_TEST_PREVIOUSLY_EXPOSED).lower(),
                "hyperparameters_provisional": "true",
            }
            raw_rows.append(row)

    rows = aggregate_rows(raw_rows)
    destination = Path(output)
    raw_destination = Path(raw_output)
    _write_csv_atomic(raw_destination, raw_rows, RAW_FIELDS)
    _write_csv_atomic(destination, rows, SUMMARY_FIELDS)
    print(f"\nDone: {destination.resolve()}")
    print(f"Raw runs: {raw_destination.resolve()}")
    print(
        f"Excluded normalized train/test duplicates: {len(excluded_ids)}; "
        "official test is explicitly marked as previously exposed."
    )
    for row in rows[: len(MODEL_ORDER)]:
        print(
            f"{row['model_label']}: test accuracy {row['test_accuracy_pct']}, "
            f"prequential NLL {row['prequential_nll_bits']}"
        )
    return rows


def main() -> None:
    run_paper_experiment()


if __name__ == "__main__":
    main()
