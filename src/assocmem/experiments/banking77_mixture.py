from __future__ import annotations

import csv
import hashlib
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from ..categorical import (
    CategoricalAssociativeMemory,
    CategoricalMemoryConfig,
    CategoricalMixtureDecoder,
    CategoricalUpdateConfig,
)
from ..config import TextEncoderConfig
from ..encoding import SignedHashTextEncoder
from ..online import OnlineMetrics, Prediction
from .banking77_paper import (
    OFFICIAL_TEST_PREVIOUSLY_EXPOSED,
    EncodedExample,
    PaperExperimentConfig,
    _build_model,
    _encode,
    _normalize_text,
    _run_model,
    _write_csv_atomic,
    duplicate_test_ids,
    select_shared_birth_ids,
)
from .datasets import iter_banking77

PROTOCOL_VERSION = "banking77-mixture-v1-posthoc"
VALIDATION_SPLIT_VERSION = "banking77-mixture-val-v1"
EXPECTED_VALIDATION_HASH = "ebfb17faadb9f66177159020d8584acd5978b07758d153ac5493e65147a7fbb0"

METRIC_FIELDS = (
    "prequential_nll_bits",
    "prequential_accuracy_pct",
    "eval_nll_bits",
    "eval_accuracy_pct",
    "eval_macro_f1_pct",
    "eval_brier",
    "eval_ece_15",
    "weighted_target_purity",
    "target_retrieval_mass",
    "energy_margin",
    "recall_at_1",
    "recall_at_4",
    "recall_at_16",
    "mrr",
    "active_atoms",
    "allocated_bytes",
    "run_seconds",
)
GEOMETRY_FIELDS = (
    "weighted_target_purity",
    "target_retrieval_mass",
    "energy_margin",
    "recall_at_1",
    "recall_at_4",
    "recall_at_16",
    "mrr",
)
RAW_FIELDS = (
    "protocol_version",
    "experiment",
    "stage",
    "key_mode",
    "training_decoder",
    "eval_decoder",
    "training_prior",
    "eval_prior",
    "seed",
    *METRIC_FIELDS,
    "train_examples",
    "eval_examples",
    "dimension",
    "max_features",
    "hash_seed",
    "capacity",
    "top_k",
    "key_nnz",
    "key_scale",
    "prior_mass",
    "learning_rate_key",
    "fixed_key_support",
    "birth_seed",
    "birth_count",
    "validation_split_version",
    "validation_split_hash",
    "official_test_previously_exposed",
)
SUMMARY_FIELDS = (
    "row_type",
    "experiment",
    "stage",
    "key_mode",
    "training_decoder",
    "eval_decoder",
    "training_prior",
    "eval_prior",
    "seeds",
    *METRIC_FIELDS,
    "comparison",
    "nll_advantage_bits",
    "nll_advantage_ci95",
    "nll_favorable_seeds",
    "accuracy_advantage_pp",
    "accuracy_advantage_ci95",
    "accuracy_favorable_seeds",
    "purity_advantage",
    "target_mass_advantage",
    "margin_advantage",
    "mrr_advantage",
    "protocol_version",
)


@dataclass(frozen=True)
class MixtureExperimentConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    dimension: int = 8192
    max_features: int = 256
    capacity: int = 4055
    top_k: int = 16
    key_nnz: int = 256
    key_scale: float = 8.0
    prior_mass: float = 1.0
    learning_rate_key: float = 0.2
    birth_seed: int = 1729
    torch_threads: int = 8


def stratified_validation_split(
    records: list[tuple[str, int]],
) -> tuple[list[int], list[int], str]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for row_id, (_, target) in enumerate(records):
        by_class[target].append(row_id)
    validation: set[int] = set()
    for row_ids in by_class.values():
        ranked = sorted(
            row_ids,
            key=lambda row_id: hashlib.sha256(
                f"{VALIDATION_SPLIT_VERSION}:{row_id}".encode("ascii")
            ).digest(),
        )
        validation.update(ranked[: round(0.2 * len(ranked))])
    training = set(range(len(records))) - validation
    training_text = {_normalize_text(records[row_id][0]) for row_id in training}
    cross_duplicates = {
        row_id for row_id in validation if _normalize_text(records[row_id][0]) in training_text
    }
    validation -= cross_duplicates
    training |= cross_duplicates
    validation_ids = sorted(validation)
    split_hash = hashlib.sha256(
        ",".join(str(row_id) for row_id in validation_ids).encode("ascii")
    ).hexdigest()
    return sorted(training), validation_ids, split_hash


def _encode_ids(
    records: list[tuple[str, int]],
    row_ids: list[int],
    encoder: SignedHashTextEncoder,
) -> list[EncodedExample]:
    return [
        EncodedExample(
            encoder.encode(records[row_id][0], task_id="banking77"),
            records[row_id][1],
            row_id,
        )
        for row_id in row_ids
    ]


def _prior_probabilities(
    num_classes: int,
    mode: str,
    counts: torch.Tensor,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)
    empirical = counts.to(torch.float64) + 1.0
    return (empirical / empirical.sum()).to(torch.float32)


def _direct_mixture_prediction(
    *,
    W: torch.Tensor,
    labels: torch.Tensor,
    query,
    target: int,
    key_scale: float,
    top_k: int,
    decoder: CategoricalMixtureDecoder,
    prior: torch.Tensor,
) -> tuple[Prediction, dict[str, float]]:
    size = W.shape[0]
    if query.nnz:
        dot = W[:, query.indices] @ query.values
    else:
        dot = torch.zeros(size, dtype=torch.float32)
    energies = (0.5 * (key_scale - dot)).clamp_min(0)
    order = torch.argsort(energies, stable=True)
    selected = order[: min(top_k, size)]
    selected_energies = energies[selected]
    selected_labels = labels[selected]
    decoded = decoder.decode(selected_energies, selected_labels, prior)
    masses = torch.exp2(-selected_energies.to(torch.float64))
    correct = selected_labels.to(torch.int64) == target
    mass_sum = float(masses.sum())
    purity = float(masses[correct].sum()) / mass_sum if mass_sum else math.nan
    target_mass = float(decoded.responsibilities[correct].sum())
    ordered_labels = labels[order].to(torch.int64)
    correct_positions = torch.nonzero(ordered_labels == target, as_tuple=False).flatten()
    first_rank = int(correct_positions[0]) + 1 if correct_positions.numel() else math.inf
    correct_energies = energies[labels.to(torch.int64) == target]
    wrong_energies = energies[labels.to(torch.int64) != target]
    margin = (
        float(wrong_energies.min() - correct_energies.min())
        if correct_energies.numel() and wrong_energies.numel()
        else math.nan
    )
    diagnostics = {
        "weighted_target_purity": purity,
        "target_retrieval_mass": target_mass,
        "energy_margin": margin,
        "recall_at_1": float(first_rank <= 1),
        "recall_at_4": float(first_rank <= 4),
        "recall_at_16": float(first_rank <= 16),
        "mrr": 0.0 if math.isinf(first_rank) else 1.0 / first_rank,
    }
    return (
        Prediction(decoded.probabilities, int(decoded.probabilities.argmax())),
        diagnostics,
    )


def _evaluate_mixture(
    *,
    W: torch.Tensor,
    labels: torch.Tensor,
    examples: list[EncodedExample],
    num_classes: int,
    key_scale: float,
    top_k: int,
    prior_mass: float,
    prior: torch.Tensor,
) -> dict[str, float]:
    metrics = OnlineMetrics(num_classes=num_classes)
    diagnostics: dict[str, list[float]] = defaultdict(list)
    decoder = CategoricalMixtureDecoder(prior_mass)
    for example in examples:
        started = time.perf_counter()
        prediction, diagnostic = _direct_mixture_prediction(
            W=W,
            labels=labels,
            query=example.query,
            target=example.target,
            key_scale=key_scale,
            top_k=top_k,
            decoder=decoder,
            prior=prior,
        )
        metrics.update(prediction, example.target, time.perf_counter() - started)
        for name, value in diagnostic.items():
            if not math.isnan(value):
                diagnostics[name].append(value)
    summary = metrics.summary()
    return {
        "eval_nll_bits": float(summary["prequential_nll_bits"]),
        "eval_accuracy_pct": 100.0 * float(summary["accuracy"]),
        "eval_macro_f1_pct": 100.0 * float(summary["macro_f1"]),
        "eval_brier": float(summary["brier"]),
        "eval_ece_15": float(summary["ece_15"]),
        **{
            name: (statistics.fmean(diagnostics[name]) if diagnostics[name] else math.nan)
            for name in GEOMETRY_FIELDS
        },
    }


def _legacy_labels(memory, target_by_id: dict[int, int]) -> torch.Tensor:
    return torch.tensor(
        [target_by_id[int(row_id)] for row_id in memory.origin_id[: memory.size]],
        dtype=torch.int16,
    )


def _base_row(
    *,
    experiment: str,
    stage: str,
    key_mode: str,
    training_decoder: str,
    eval_decoder: str,
    training_prior: str,
    eval_prior: str,
    seed: int,
    config: MixtureExperimentConfig,
    train_examples: int,
    eval_examples: int,
    birth_count: int,
    split_hash: str,
) -> dict[str, float | int | str]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": experiment,
        "stage": stage,
        "key_mode": key_mode,
        "training_decoder": training_decoder,
        "eval_decoder": eval_decoder,
        "training_prior": training_prior,
        "eval_prior": eval_prior,
        "seed": seed,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "dimension": config.dimension,
        "max_features": config.max_features,
        "hash_seed": 0,
        "capacity": config.capacity,
        "top_k": config.top_k,
        "key_nnz": config.key_nnz,
        "key_scale": config.key_scale,
        "prior_mass": config.prior_mass,
        "learning_rate_key": config.learning_rate_key,
        "fixed_key_support": "true",
        "birth_seed": config.birth_seed,
        "birth_count": birth_count,
        "validation_split_version": VALIDATION_SPLIT_VERSION,
        "validation_split_hash": split_hash,
        "official_test_previously_exposed": str(OFFICIAL_TEST_PREVIOUSLY_EXPOSED).lower(),
    }


def _run_categorical_training(
    *,
    examples: list[EncodedExample],
    births: set[int],
    num_classes: int,
    learned: bool,
    config: MixtureExperimentConfig,
) -> tuple[CategoricalAssociativeMemory, dict[str, float]]:
    memory = CategoricalAssociativeMemory(
        CategoricalMemoryConfig(
            dimension=config.dimension,
            num_classes=num_classes,
            capacity=config.capacity,
            top_k=config.top_k,
            key_nnz=config.key_nnz,
            key_scale=config.key_scale,
            prior_mass=config.prior_mass,
            prior_mode="uniform",
            update=CategoricalUpdateConfig(
                learning_rate_key=config.learning_rate_key,
                train_keys=learned,
            ),
        )
    )
    metrics = OnlineMetrics(num_classes=num_classes)
    started_run = time.perf_counter()
    for example in examples:
        started = time.perf_counter()
        read = memory.read(example.query)
        prediction = Prediction(read.probabilities, int(read.prediction), read)
        metrics.update(prediction, example.target, time.perf_counter() - started)
        started = time.perf_counter()
        memory.observe(
            example.query,
            example.target,
            pre_read=read,
            origin_id=example.example_id,
            insertion_mode=("force" if example.example_id in births else "skip"),
        )
        metrics.add_update_latency(time.perf_counter() - started)
    summary = metrics.summary()
    return memory, {
        "prequential_nll_bits": float(summary["prequential_nll_bits"]),
        "prequential_accuracy_pct": 100.0 * float(summary["accuracy"]),
        "run_seconds": time.perf_counter() - started_run,
    }


def _mean_std(values: list[float]) -> str:
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.6f} ± {deviation:.6f}"


def _paired_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    critical = 2.776 if len(values) == 5 else 1.96
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def aggregate_rows(
    raw_rows: list[dict[str, float | int | str]],
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, float | int | str]]] = defaultdict(list)
    keys = (
        "experiment",
        "stage",
        "key_mode",
        "training_decoder",
        "eval_decoder",
        "training_prior",
        "eval_prior",
    )
    for row in raw_rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    result: list[dict[str, str]] = []
    for group, rows in sorted(grouped.items()):
        aggregate = {field: "" for field in SUMMARY_FIELDS}
        aggregate.update(
            {
                "row_type": "model",
                **dict(zip(keys, group, strict=True)),
                "seeds": str(len(rows)),
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        for field in METRIC_FIELDS:
            values = [
                float(row[field])
                for row in rows
                if field in row and not math.isnan(float(row[field]))
            ]
            aggregate[field] = _mean_std(values) if values else ""
        result.append(aggregate)

    for key_mode in ("learned", "frozen"):
        candidates = [
            row
            for row in raw_rows
            if row["experiment"] == "A"
            and row["key_mode"] == key_mode
            and row["eval_prior"] == "empirical"
        ]
        by_decoder_seed = {(str(row["eval_decoder"]), int(row["seed"])): row for row in candidates}
        seeds = sorted(
            {seed for decoder, seed in by_decoder_seed if decoder == "categorical_mixture"}
            & {seed for decoder, seed in by_decoder_seed if decoder == "legacy_gated_logit"}
        )
        if not seeds:
            continue
        comparison = {field: "" for field in SUMMARY_FIELDS}
        comparison.update(
            {
                "row_type": "comparison",
                "experiment": "A",
                "stage": "official_test",
                "key_mode": key_mode,
                "comparison": "categorical_mixture_vs_legacy_decoder",
                "seeds": str(len(seeds)),
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        differences = [
            float(by_decoder_seed[("legacy_gated_logit", seed)]["eval_nll_bits"])
            - float(by_decoder_seed[("categorical_mixture", seed)]["eval_nll_bits"])
            for seed in seeds
        ]
        mean, low, high = _paired_interval(differences)
        comparison["nll_advantage_bits"] = f"{mean:.6f}"
        comparison["nll_advantage_ci95"] = f"[{low:.6f}, {high:.6f}]"
        comparison["nll_favorable_seeds"] = str(sum(value > 0 for value in differences))
        differences = [
            float(by_decoder_seed[("categorical_mixture", seed)]["eval_accuracy_pct"])
            - float(by_decoder_seed[("legacy_gated_logit", seed)]["eval_accuracy_pct"])
            for seed in seeds
        ]
        mean, low, high = _paired_interval(differences)
        comparison["accuracy_advantage_pp"] = f"{mean:.6f}"
        comparison["accuracy_advantage_ci95"] = f"[{low:.6f}, {high:.6f}]"
        comparison["accuracy_favorable_seeds"] = str(sum(value > 0 for value in differences))
        result.append(comparison)

    for stage in ("validation", "official_test"):
        candidates = [
            row
            for row in raw_rows
            if row["experiment"] == "B"
            and row["stage"] == stage
            and row["eval_decoder"] == "categorical_mixture"
            and row["eval_prior"] == "uniform"
        ]
        by_key_seed = {(str(row["key_mode"]), int(row["seed"])): row for row in candidates}
        seeds = sorted(
            {seed for mode, seed in by_key_seed if mode == "learned"}
            & {seed for mode, seed in by_key_seed if mode == "frozen"}
        )
        if not seeds:
            continue
        comparison = {field: "" for field in SUMMARY_FIELDS}
        comparison.update(
            {
                "row_type": "comparison",
                "experiment": "B",
                "stage": stage,
                "comparison": "learned_vs_frozen",
                "seeds": str(len(seeds)),
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        specifications = (
            ("nll", "eval_nll_bits", lambda learned, frozen: frozen - learned),
            (
                "accuracy",
                "eval_accuracy_pct",
                lambda learned, frozen: learned - frozen,
            ),
        )
        for prefix, metric, advantage in specifications:
            differences = [
                advantage(
                    float(by_key_seed[("learned", seed)][metric]),
                    float(by_key_seed[("frozen", seed)][metric]),
                )
                for seed in seeds
            ]
            mean, low, high = _paired_interval(differences)
            suffix = "bits" if prefix == "nll" else "pp"
            comparison[f"{prefix}_advantage_{suffix}"] = f"{mean:.6f}"
            comparison[f"{prefix}_advantage_ci95"] = f"[{low:.6f}, {high:.6f}]"
            comparison[f"{prefix}_favorable_seeds"] = str(sum(value > 0 for value in differences))
        for output, metric in (
            ("purity_advantage", "weighted_target_purity"),
            ("target_mass_advantage", "target_retrieval_mass"),
            ("margin_advantage", "energy_margin"),
            ("mrr_advantage", "mrr"),
        ):
            differences = [
                float(by_key_seed[("learned", seed)][metric])
                - float(by_key_seed[("frozen", seed)][metric])
                for seed in seeds
            ]
            comparison[output] = f"{statistics.fmean(differences):.6f}"
        result.append(comparison)
    return result


def _legacy_reproduction_check(
    raw_v2: dict[tuple[str, int], dict[str, str]],
    key_mode: str,
    seed: int,
    measurements: dict[str, float | int | None],
) -> None:
    expected = raw_v2[(f"shared_{key_mode}", seed)]
    for field, tolerance in (
        ("prequential_nll_bits", 1e-5),
        ("prequential_accuracy_pct", 1e-9),
        ("test_nll_bits", 1e-5),
        ("test_accuracy_pct", 1e-9),
    ):
        if abs(float(expected[field]) - float(measurements[field])) > tolerance:
            raise RuntimeError(f"legacy reproduction mismatch for {key_mode} seed={seed} {field}")


def run_mixture_experiment(
    *,
    data_dir: str | Path = "data",
    output: str | Path = "banking77_mixture_results_v1.csv",
    raw_output: str | Path = "banking77_mixture_runs_v1.csv",
    legacy_raw: str | Path = "banking77_runs_v2.csv",
    config: MixtureExperimentConfig | None = None,
) -> list[dict[str, str]]:
    config = config or MixtureExperimentConfig()
    torch.set_num_threads(config.torch_threads)
    train_records = list(iter_banking77(data_dir, "train"))
    test_records = list(iter_banking77(data_dir, "test"))
    num_classes = len({target for _, target in train_records})
    encoder = SignedHashTextEncoder(
        TextEncoderConfig(
            dimension=config.dimension,
            max_features=config.max_features,
            hash_seed=0,
        )
    )
    encoded_full = _encode(train_records, encoder)
    encoded_test = _encode(test_records, encoder)
    duplicate_ids = duplicate_test_ids(train_records, test_records)
    encoded_test_dedup = [
        example for example in encoded_test if example.example_id not in duplicate_ids
    ]
    train_ids, validation_ids, split_hash = stratified_validation_split(train_records)
    if split_hash != EXPECTED_VALIDATION_HASH:
        raise RuntimeError("BANKING77 validation split hash mismatch")
    encoded_dev = _encode_ids(train_records, train_ids, encoder)
    encoded_validation = _encode_ids(train_records, validation_ids, encoder)
    target_by_id = {row_id: target for row_id, (_, target) in enumerate(train_records)}
    with Path(legacy_raw).open(encoding="utf-8", newline="") as handle:
        raw_v2 = {(row["model"], int(row["seed"])): row for row in csv.DictReader(handle)}
    raw_rows: list[dict[str, float | int | str]] = []
    total_training_runs = len(config.seeds) * 6
    completed = 0

    paper_config = PaperExperimentConfig(
        seeds=config.seeds,
        dimension=config.dimension,
        max_features=config.max_features,
        capacity=config.capacity,
        top_k=config.top_k,
        key_nnz=config.key_nnz,
        key_scale=config.key_scale,
        learning_rate_key=config.learning_rate_key,
        torch_threads=config.torch_threads,
    )
    full_births = select_shared_birth_ids(encoded_full, config.capacity, config.birth_seed)
    for seed in config.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        stream = list(encoded_full)
        random.Random(seed).shuffle(stream)
        for key_mode in ("learned", "frozen"):
            completed += 1
            print(
                f"[{completed:02d}/{total_training_runs}] A seed={seed} legacy-{key_mode}",
                flush=True,
            )
            model_name = f"shared_{key_mode}"
            model = _build_model(
                model_name,
                encoder=encoder,
                num_classes=num_classes,
                train_examples=len(encoded_full),
                config=paper_config,
            )
            legacy_measurements = _run_model(
                model_name,
                model,
                stream,
                encoded_test,
                encoded_test_dedup,
                num_classes,
                full_births,
            )
            _legacy_reproduction_check(raw_v2, key_mode, seed, legacy_measurements)
            base = _base_row(
                experiment="A",
                stage="official_test",
                key_mode=key_mode,
                training_decoder="legacy_gated_logit",
                eval_decoder="legacy_gated_logit",
                training_prior="empirical",
                eval_prior="empirical",
                seed=seed,
                config=config,
                train_examples=len(encoded_full),
                eval_examples=len(encoded_test),
                birth_count=len(full_births),
                split_hash=split_hash,
            )
            base.update(
                {
                    "prequential_nll_bits": legacy_measurements["prequential_nll_bits"],
                    "prequential_accuracy_pct": legacy_measurements["prequential_accuracy_pct"],
                    "eval_nll_bits": legacy_measurements["test_nll_bits"],
                    "eval_accuracy_pct": legacy_measurements["test_accuracy_pct"],
                    "eval_macro_f1_pct": legacy_measurements["test_macro_f1_pct"],
                    "eval_brier": legacy_measurements["test_brier"],
                    "eval_ece_15": math.nan,
                    "weighted_target_purity": math.nan,
                    "target_retrieval_mass": math.nan,
                    "energy_margin": math.nan,
                    "recall_at_1": math.nan,
                    "recall_at_4": math.nan,
                    "recall_at_16": math.nan,
                    "mrr": math.nan,
                    "active_atoms": int(model.memory.size),
                    "allocated_bytes": int(model.allocated_bytes()),
                    "run_seconds": legacy_measurements["run_seconds"],
                }
            )
            raw_rows.append(base)
            labels = _legacy_labels(model.memory, target_by_id)
            for prior_mode in ("uniform", "empirical"):
                prior = _prior_probabilities(num_classes, prior_mode, model.memory.base_counts)
                evaluated = _evaluate_mixture(
                    W=model.memory.W[: model.memory.size],
                    labels=labels,
                    examples=encoded_test,
                    num_classes=num_classes,
                    key_scale=config.key_scale,
                    top_k=config.top_k,
                    prior_mass=config.prior_mass,
                    prior=prior,
                )
                row = _base_row(
                    experiment="A",
                    stage="official_test",
                    key_mode=key_mode,
                    training_decoder="legacy_gated_logit",
                    eval_decoder="categorical_mixture",
                    training_prior="empirical",
                    eval_prior=prior_mode,
                    seed=seed,
                    config=config,
                    train_examples=len(encoded_full),
                    eval_examples=len(encoded_test),
                    birth_count=len(full_births),
                    split_hash=split_hash,
                )
                row.update(
                    {
                        "prequential_nll_bits": math.nan,
                        "prequential_accuracy_pct": math.nan,
                        **evaluated,
                        "active_atoms": int(model.memory.size),
                        "allocated_bytes": int(model.allocated_bytes()),
                        "run_seconds": 0.0,
                    }
                )
                raw_rows.append(row)

    for stage, train_examples, eval_examples in (
        ("validation", encoded_dev, encoded_validation),
        ("official_test", encoded_full, encoded_test),
    ):
        births = select_shared_birth_ids(train_examples, config.capacity, config.birth_seed)
        for seed in config.seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            stream = list(train_examples)
            random.Random(seed).shuffle(stream)
            for key_mode in ("learned", "frozen"):
                completed += 1
                print(
                    f"[{completed:02d}/{total_training_runs}] "
                    f"B/{stage} seed={seed} mixture-{key_mode}",
                    flush=True,
                )
                memory, prequential = _run_categorical_training(
                    examples=stream,
                    births=births,
                    num_classes=num_classes,
                    learned=key_mode == "learned",
                    config=config,
                )
                for prior_mode in ("uniform", "empirical"):
                    evaluated = _evaluate_mixture(
                        W=memory.W[: memory.size],
                        labels=memory.birth_label[: memory.size],
                        examples=eval_examples,
                        num_classes=num_classes,
                        key_scale=config.key_scale,
                        top_k=config.top_k,
                        prior_mass=config.prior_mass,
                        prior=_prior_probabilities(num_classes, prior_mode, memory.base_counts),
                    )
                    row = _base_row(
                        experiment="B",
                        stage=stage,
                        key_mode=key_mode,
                        training_decoder="categorical_mixture",
                        eval_decoder="categorical_mixture",
                        training_prior="uniform",
                        eval_prior=prior_mode,
                        seed=seed,
                        config=config,
                        train_examples=len(train_examples),
                        eval_examples=len(eval_examples),
                        birth_count=len(births),
                        split_hash=split_hash,
                    )
                    row.update(
                        {
                            **prequential,
                            **evaluated,
                            "active_atoms": memory.size,
                            "allocated_bytes": memory.allocated_bytes(),
                        }
                    )
                    raw_rows.append(row)

    summary = aggregate_rows(raw_rows)
    _write_csv_atomic(Path(raw_output), raw_rows, RAW_FIELDS)
    _write_csv_atomic(Path(output), summary, SUMMARY_FIELDS)
    print(f"\nDone: {Path(output).resolve()}")
    print(f"Raw runs: {Path(raw_output).resolve()}")
    return summary


def main() -> None:
    run_mixture_experiment()


if __name__ == "__main__":
    main()
