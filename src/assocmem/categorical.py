from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import torch

from .encoding import TernaryQuery

PriorMode = Literal["uniform", "empirical"]


@dataclass(frozen=True)
class CategoricalUpdateConfig:
    learning_rate_key: float = 0.2
    gradient_clip_norm: float = 1.0
    train_keys: bool = True
    backtracking_attempts: int = 5
    backtracking_factor: float = 0.5
    acceptance_tolerance_bits: float = 1e-8
    fixed_key_support: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate_key <= 0:
            raise ValueError("learning_rate_key must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.backtracking_attempts <= 0:
            raise ValueError("backtracking_attempts must be positive")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must be in (0, 1)")


@dataclass(frozen=True)
class CategoricalMemoryConfig:
    dimension: int
    num_classes: int
    capacity: int
    top_k: int | None = 16
    key_nnz: int = 256
    key_scale: float = 8.0
    prior_mass: float = 1.0
    prior_mode: PriorMode = "uniform"
    prior_pseudocount: float = 1.0
    encoder_fingerprint: str | None = None
    update: CategoricalUpdateConfig = field(default_factory=CategoricalUpdateConfig)
    usage_decay: float = 0.999

    def __post_init__(self) -> None:
        for name in ("dimension", "num_classes", "capacity", "key_nnz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.key_nnz > self.dimension:
            raise ValueError("key_nnz cannot exceed dimension")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if self.key_scale <= 0 or self.prior_mass <= 0 or self.prior_pseudocount <= 0:
            raise ValueError("key_scale, prior_mass, and prior_pseudocount must be positive")
        if self.prior_mode not in {"uniform", "empirical"}:
            raise ValueError("prior_mode must be uniform or empirical")
        if not 0 < self.usage_decay <= 1:
            raise ValueError("usage_decay must be in (0, 1]")
        if not self.update.fixed_key_support:
            raise ValueError("categorical MVP requires fixed_key_support")


@dataclass(frozen=True)
class DecoderOutput:
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    responsibilities: torch.Tensor
    background_responsibility: torch.Tensor


class CategoricalMixtureDecoder:
    def __init__(self, prior_mass: float = 1.0):
        if not math.isfinite(prior_mass) or prior_mass <= 0:
            raise ValueError("prior_mass must be finite and positive")
        self.prior_mass = prior_mass

    def decode(
        self,
        energies: torch.Tensor,
        birth_labels: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> DecoderOutput:
        if energies.ndim != 1 or birth_labels.ndim != 1:
            raise ValueError("energies and birth_labels must be one-dimensional")
        if energies.numel() != birth_labels.numel():
            raise ValueError("energies and birth_labels must have equal length")
        if prior_probabilities.ndim != 1 or prior_probabilities.numel() == 0:
            raise ValueError("prior_probabilities must be a non-empty vector")
        if not bool(torch.isfinite(energies).all()) or bool((energies < 0).any()):
            raise ValueError("energies must be finite and non-negative")
        if birth_labels.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("birth_labels must have an integer dtype")
        if birth_labels.numel() and (
            int(birth_labels.min()) < 0 or int(birth_labels.max()) >= prior_probabilities.numel()
        ):
            raise ValueError("birth label outside class space")
        if (
            not bool(torch.isfinite(prior_probabilities).all())
            or bool((prior_probabilities <= 0).any())
            or not torch.isclose(
                prior_probabilities.sum(),
                torch.ones((), device=prior_probabilities.device),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise ValueError("prior_probabilities must be a positive distribution")

        device = energies.device
        prior = prior_probabilities.to(device=device, dtype=torch.float64)
        masses = torch.exp2(-energies.to(torch.float64))
        denominator = self.prior_mass + masses.sum()
        responsibilities = masses / denominator
        background = (
            torch.as_tensor(self.prior_mass, dtype=torch.float64, device=device) / denominator
        )
        votes = torch.zeros(prior.numel(), dtype=torch.float64, device=device)
        if masses.numel():
            votes.scatter_add_(0, birth_labels.to(device=device, dtype=torch.int64), masses)
        probabilities = (self.prior_mass * prior + votes) / denominator
        probabilities = probabilities.to(torch.float32)
        return DecoderOutput(
            probabilities=probabilities,
            log_probabilities=probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
            responsibilities=responsibilities.to(torch.float32),
            background_responsibility=background.to(torch.float32),
        )


@dataclass(frozen=True)
class CategoricalReadResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    prediction: torch.Tensor
    slot_ids: torch.Tensor
    atom_uids: torch.Tensor
    origin_ids: torch.Tensor
    birth_labels: torch.Tensor
    energies: torch.Tensor
    responsibilities: torch.Tensor
    background_responsibility: torch.Tensor
    query: TernaryQuery
    prior_mode: PriorMode
    state_revision: int
    owner_token: str
    query_signature: str


@dataclass(frozen=True)
class CompactReadResult:
    prediction: int
    slot_ids: torch.Tensor
    atom_uids: torch.Tensor
    origin_ids: torch.Tensor
    birth_labels: torch.Tensor
    energies: torch.Tensor
    masses: torch.Tensor
    responsibilities: torch.Tensor
    background_responsibility: torch.Tensor
    denominator: torch.Tensor
    query: TernaryQuery
    prior_mode: PriorMode
    state_revision: int
    owner_token: str
    query_signature: str


@dataclass(frozen=True)
class CategoricalLearnReport:
    applied: bool
    attempts: int
    selected_slots: tuple[int, ...]
    key_gradient_norm: float
    loss_before_bits: float
    loss_after_bits: float
    support_added: int
    support_removed: int


@dataclass(frozen=True)
class CategoricalInsertReport:
    attempted: bool
    inserted: bool
    slot_id: int | None
    atom_uid: int | None
    evicted_atom_uid: int | None
    surprise_bits: float
    gain_bits: float
    reason: str


@dataclass(frozen=True)
class CategoricalObserveReport:
    read_before: CategoricalReadResult
    read_after: CategoricalReadResult
    learn: CategoricalLearnReport
    insertion: CategoricalInsertReport
    target: int


@dataclass(frozen=True)
class CategoricalCompactObserveReport:
    prediction_before: int
    prediction_after: int
    learn: CategoricalLearnReport
    insertion: CategoricalInsertReport
    target: int


def _query_signature(query: TernaryQuery) -> str:
    payload = (
        query.dimension,
        query.encoder_fingerprint,
        query.indices.detach().cpu().tolist(),
        query.values.detach().cpu().tolist(),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def _loss_bits(read: CategoricalReadResult, target: int) -> float:
    probability = read.probabilities[target].clamp_min(torch.finfo(torch.float32).tiny)
    return float((-torch.log2(probability)).detach().cpu())


def _clip_rows(gradient: torch.Tensor, maximum: float) -> torch.Tensor:
    norms = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return gradient * (maximum / norms).clamp_max(1.0)


class CategoricalAssociativeMemory:
    def __init__(
        self,
        config: CategoricalMemoryConfig,
        *,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)
        self.decoder = CategoricalMixtureDecoder(config.prior_mass)
        self.W = torch.zeros(
            (config.capacity, config.dimension), dtype=torch.float32, device=self.device
        )
        self.birth_label = torch.full((config.capacity,), -1, dtype=torch.int64, device=self.device)
        self.usage = torch.zeros(config.capacity, dtype=torch.float32, device=self.device)
        self.origin_id = torch.full((config.capacity,), -1, dtype=torch.int64, device=self.device)
        self.atom_uid = torch.full((config.capacity,), -1, dtype=torch.int64, device=self.device)
        self.base_counts = torch.zeros(config.num_classes, dtype=torch.int64, device=self.device)
        self._class_capacity = config.num_classes
        self.base_count_argmax = 0
        self.size = 0
        self.step = 0
        self.revision = 0
        self.next_atom_uid = 0
        self._owner_token = uuid.uuid4().hex

    def expand_classes(self, new_num_classes: int, *, reserve: int | None = None) -> None:
        """Grow the categorical value space without changing existing class ids."""
        if new_num_classes < self.config.num_classes:
            raise ValueError("class space cannot shrink")
        if new_num_classes == self.config.num_classes and (reserve or 0) <= self._class_capacity:
            return
        requested = max(new_num_classes, reserve or 0)
        if requested > self._class_capacity:
            capacity = max(requested, 16, 2 * self._class_capacity)
            counts = torch.zeros(capacity, dtype=torch.int64, device=self.device)
            counts[: self.config.num_classes] = self.base_counts[: self.config.num_classes]
            self.base_counts = counts
            self._class_capacity = capacity
        if new_num_classes != self.config.num_classes:
            self.config = replace(self.config, num_classes=new_num_classes)
            self.revision += 1

    def _prior_probability(self, target: int, mode: PriorMode) -> torch.Tensor:
        if not 0 <= target < self.config.num_classes:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        if mode == "uniform":
            return torch.as_tensor(
                1.0 / self.config.num_classes, dtype=torch.float64, device=self.device
            )
        pseudo = self.config.prior_pseudocount
        numerator = self.base_counts[target].to(torch.float64) + pseudo
        denominator = self.step + pseudo * self.config.num_classes
        return numerator / denominator

    def _retrieve_evidence(
        self,
        query: TernaryQuery,
        *,
        prior_mode: PriorMode | None = None,
    ) -> CompactReadResult:
        query = self._validate_query(query)
        mode = self.config.prior_mode if prior_mode is None else prior_mode
        if mode not in {"uniform", "empirical"}:
            raise ValueError("prior_mode must be uniform or empirical")
        if query.nnz:
            dot = self.W[: self.size, query.indices] @ query.values
        else:
            dot = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        energies_all = (0.5 * (self.config.key_scale - dot)).clamp_min(0)
        order = torch.argsort(energies_all, stable=True)
        k = self.size if self.config.top_k is None else min(self.config.top_k, self.size)
        slots = order[:k]
        energies = energies_all[slots]
        labels = self.birth_label[slots]
        masses = torch.exp2(-energies.to(torch.float64))
        denominator = torch.as_tensor(
            self.config.prior_mass, dtype=torch.float64, device=self.device
        ) + masses.sum()
        responsibilities = masses / denominator
        background = torch.as_tensor(
            self.config.prior_mass, dtype=torch.float64, device=self.device
        ) / denominator
        prior_argmax = 0 if mode == "uniform" else self.base_count_argmax
        candidates = sorted({prior_argmax, *(int(value) for value in labels.detach().cpu())})
        prediction = candidates[0]
        best = -1.0
        for candidate in candidates:
            vote = masses[labels == candidate].sum()
            probability = (
                self.config.prior_mass * self._prior_probability(candidate, mode) + vote
            ) / denominator
            value = float(probability)
            if value > best:
                prediction, best = candidate, value
        return CompactReadResult(
            prediction=prediction,
            slot_ids=slots.detach(),
            atom_uids=self.atom_uid[slots].detach().clone(),
            origin_ids=self.origin_id[slots].detach().clone(),
            birth_labels=labels.detach().clone(),
            energies=energies.detach(),
            masses=masses.detach(),
            responsibilities=responsibilities.to(torch.float32).detach(),
            background_responsibility=background.to(torch.float32).detach(),
            denominator=denominator.detach(),
            query=query,
            prior_mode=mode,
            state_revision=self.revision,
            owner_token=self._owner_token,
            query_signature=_query_signature(query),
        )

    def _compact_probability(self, read: CompactReadResult, target: int) -> torch.Tensor:
        vote = read.masses[read.birth_labels == target].sum()
        numerator = (
            self.config.prior_mass * self._prior_probability(target, read.prior_mode) + vote
        )
        return numerator / read.denominator

    def log_probabilities_for(
        self,
        query: TernaryQuery,
        target_ids: Sequence[int],
        *,
        prior_mode: PriorMode | None = None,
    ) -> torch.Tensor:
        read = self._retrieve_evidence(query, prior_mode=prior_mode)
        values = [
            self._compact_probability(read, int(target)).clamp_min(torch.finfo(torch.float64).tiny)
            for target in target_ids
        ]
        if not values:
            return torch.empty(0, dtype=torch.float64, device=self.device)
        return torch.stack(values).log()

    def predict_class(
        self, query: TernaryQuery, *, prior_mode: PriorMode | None = None
    ) -> int:
        return self._retrieve_evidence(query, prior_mode=prior_mode).prediction

    def _validate_query(self, query: TernaryQuery) -> TernaryQuery:
        if query.dimension != self.config.dimension:
            raise ValueError("query and memory dimensions differ")
        expected = self.config.encoder_fingerprint
        if expected is not None and query.encoder_fingerprint != expected:
            raise ValueError("query encoder fingerprint does not match memory")
        return query.snapshot(device=self.device)

    def _base_probabilities(self, mode: PriorMode) -> torch.Tensor:
        if mode == "uniform":
            return torch.full(
                (self.config.num_classes,),
                1.0 / self.config.num_classes,
                dtype=torch.float32,
                device=self.device,
            )
        counts = self.base_counts[: self.config.num_classes].to(torch.float64)
        pseudo = self.config.prior_pseudocount
        return ((counts + pseudo) / (counts.sum() + pseudo * self.config.num_classes)).to(
            torch.float32
        )

    def read(
        self,
        query: TernaryQuery,
        *,
        prior_mode: PriorMode | None = None,
    ) -> CategoricalReadResult:
        query = self._validate_query(query)
        mode = self.config.prior_mode if prior_mode is None else prior_mode
        if mode not in {"uniform", "empirical"}:
            raise ValueError("prior_mode must be uniform or empirical")
        if query.nnz:
            dot = self.W[: self.size, query.indices] @ query.values
        else:
            dot = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        energies_all = (0.5 * (self.config.key_scale - dot)).clamp_min(0)
        order = torch.argsort(energies_all, stable=True)
        k = self.size if self.config.top_k is None else min(self.config.top_k, self.size)
        slots = order[:k]
        energies = energies_all[slots]
        labels = self.birth_label[slots]
        decoded = self.decoder.decode(energies, labels, self._base_probabilities(mode))
        return CategoricalReadResult(
            decoded.log_probabilities.detach(),
            decoded.probabilities.detach(),
            decoded.probabilities.argmax().detach(),
            slots.detach(),
            self.atom_uid[slots].detach().clone(),
            self.origin_id[slots].detach().clone(),
            labels.detach().clone(),
            energies.detach(),
            decoded.responsibilities.detach(),
            decoded.background_responsibility.detach(),
            query,
            mode,
            self.revision,
            self._owner_token,
            _query_signature(query),
        )

    def _project_keys(self, keys: torch.Tensor, old_keys: torch.Tensor) -> torch.Tensor:
        projected = torch.zeros_like(keys)
        for row in range(keys.shape[0]):
            support = old_keys[row] != 0
            if not bool(support.any()):
                raise FloatingPointError("fixed-support projection has no support")
            values = keys[row, support].clone()
            tiny = values.abs() < 1e-8
            values[tiny] = old_keys[row, support][tiny].sign() * 1e-8
            values *= self.config.key_scale / values.abs().sum()
            projected[row, support] = values
        return projected

    def _local_key_gradient(
        self, read: CategoricalReadResult, target: int
    ) -> tuple[torch.Tensor, float]:
        keys = self.W[read.slot_ids].detach().clone().requires_grad_(True)
        query = read.query
        dot = keys[:, query.indices] @ query.values if query.nnz else keys.sum(dim=1) * 0
        energies = 0.5 * (keys.abs().sum(dim=1) - dot)
        masses = torch.exp2(-energies.to(torch.float64))
        votes = torch.zeros(self.config.num_classes, dtype=torch.float64, device=self.device)
        votes.scatter_add_(0, read.birth_labels.to(torch.int64), masses)
        prior = self._base_probabilities(read.prior_mode).to(torch.float64)
        denominator = self.config.prior_mass + masses.sum()
        probability = (self.config.prior_mass * prior[target] + votes[target]) / denominator
        loss = -probability.clamp_min(torch.finfo(torch.float64).tiny).log()
        gradient = torch.autograd.grad(loss, keys)[0]
        return gradient, float(gradient.norm().detach().cpu())

    def _learn(
        self, read: CategoricalReadResult, target: int
    ) -> tuple[CategoricalLearnReport, CategoricalReadResult]:
        before = _loss_bits(read, target)
        slots = read.slot_ids
        if slots.numel() == 0 or not self.config.update.train_keys:
            return (
                CategoricalLearnReport(
                    False, 0, tuple(int(v) for v in slots.cpu()), 0.0, before, before, 0, 0
                ),
                read,
            )
        gradient, gradient_norm = self._local_key_gradient(read, target)
        gradient = _clip_rows(gradient, self.config.update.gradient_clip_norm)
        old_keys = self.W[slots].clone()
        accepted: CategoricalReadResult | None = None
        attempts = 0
        with torch.no_grad():
            for attempt in range(self.config.update.backtracking_attempts):
                attempts = attempt + 1
                factor = self.config.update.backtracking_factor**attempt
                try:
                    candidate = self._project_keys(
                        old_keys - self.config.update.learning_rate_key * factor * gradient,
                        old_keys,
                    )
                except FloatingPointError:
                    continue
                if not bool(torch.isfinite(candidate).all()):
                    continue
                self.W[slots] = candidate
                candidate_read = self.read(read.query, prior_mode=read.prior_mode)
                if (
                    _loss_bits(candidate_read, target)
                    <= before + self.config.update.acceptance_tolerance_bits
                ):
                    accepted = candidate_read
                    break
                self.W[slots] = old_keys
            if accepted is None:
                self.W[slots] = old_keys
        final = (
            accepted if accepted is not None else self.read(read.query, prior_mode=read.prior_mode)
        )
        final_keys = self.W[slots]
        return (
            CategoricalLearnReport(
                accepted is not None,
                attempts,
                tuple(int(v) for v in slots.cpu()),
                gradient_norm,
                before,
                _loss_bits(final, target),
                int(((old_keys == 0) & (final_keys != 0)).sum()),
                int(((old_keys != 0) & (final_keys == 0)).sum()),
            ),
            final,
        )

    def _update_usage(self, read: CategoricalReadResult | CompactReadResult) -> None:
        with torch.no_grad():
            self.usage[: self.size].mul_(self.config.usage_decay)
            if read.slot_ids.numel():
                self.usage[read.slot_ids].add_(read.responsibilities)

    def _compact_loss_bits(self, read: CompactReadResult, target: int) -> float:
        probability = self._compact_probability(read, target).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        return float((-torch.log2(probability)).detach().cpu())

    def _local_key_gradient_compact(
        self, read: CompactReadResult, target: int
    ) -> tuple[torch.Tensor, float]:
        keys = self.W[read.slot_ids].detach().clone().requires_grad_(True)
        query = read.query
        dot = keys[:, query.indices] @ query.values if query.nnz else keys.sum(dim=1) * 0
        energies = 0.5 * (keys.abs().sum(dim=1) - dot)
        masses = torch.exp2(-energies.to(torch.float64))
        vote = masses[read.birth_labels == target].sum()
        denominator = self.config.prior_mass + masses.sum()
        probability = (
            self.config.prior_mass * self._prior_probability(target, read.prior_mode) + vote
        ) / denominator
        loss = -probability.clamp_min(torch.finfo(torch.float64).tiny).log()
        gradient = torch.autograd.grad(loss, keys)[0]
        return gradient, float(gradient.norm().detach().cpu())

    def _learn_compact(
        self, read: CompactReadResult, target: int
    ) -> tuple[CategoricalLearnReport, CompactReadResult]:
        before = self._compact_loss_bits(read, target)
        slots = read.slot_ids
        if slots.numel() == 0 or not self.config.update.train_keys:
            return (
                CategoricalLearnReport(
                    False, 0, tuple(int(v) for v in slots.cpu()), 0.0, before, before, 0, 0
                ),
                read,
            )
        gradient, gradient_norm = self._local_key_gradient_compact(read, target)
        gradient = _clip_rows(gradient, self.config.update.gradient_clip_norm)
        old_keys = self.W[slots].clone()
        accepted: CompactReadResult | None = None
        attempts = 0
        with torch.no_grad():
            for attempt in range(self.config.update.backtracking_attempts):
                attempts = attempt + 1
                factor = self.config.update.backtracking_factor**attempt
                try:
                    candidate = self._project_keys(
                        old_keys - self.config.update.learning_rate_key * factor * gradient,
                        old_keys,
                    )
                except FloatingPointError:
                    continue
                if not bool(torch.isfinite(candidate).all()):
                    continue
                self.W[slots] = candidate
                candidate_read = self._retrieve_evidence(
                    read.query, prior_mode=read.prior_mode
                )
                if (
                    self._compact_loss_bits(candidate_read, target)
                    <= before + self.config.update.acceptance_tolerance_bits
                ):
                    accepted = candidate_read
                    break
                self.W[slots] = old_keys
            if accepted is None:
                self.W[slots] = old_keys
        final = accepted or self._retrieve_evidence(read.query, prior_mode=read.prior_mode)
        final_keys = self.W[slots]
        return (
            CategoricalLearnReport(
                accepted is not None,
                attempts,
                tuple(int(v) for v in slots.cpu()),
                gradient_norm,
                before,
                self._compact_loss_bits(final, target),
                int(((old_keys == 0) & (final_keys != 0)).sum()),
                int(((old_keys != 0) & (final_keys == 0)).sum()),
            ),
            final,
        )

    def _insert_compact(
        self,
        query: TernaryQuery,
        target: int,
        baseline: CompactReadResult,
        origin_id: int,
    ) -> tuple[CategoricalInsertReport, CompactReadResult]:
        if query.nnz == 0:
            return (
                CategoricalInsertReport(
                    True,
                    False,
                    None,
                    None,
                    None,
                    self._compact_loss_bits(baseline, target),
                    0.0,
                    "empty-query",
                ),
                baseline,
            )
        support = query.indices[: self.config.key_nnz]
        signs = query.values[: self.config.key_nnz]
        candidate = torch.zeros(self.config.dimension, dtype=torch.float32, device=self.device)
        candidate[support] = signs * (self.config.key_scale / support.numel())
        free = self.size < self.config.capacity
        slot = self.size if free else int(torch.argmin(self.usage[: self.size]).item())
        evicted_uid = None if free else int(self.atom_uid[slot].item())
        uid = self.next_atom_uid
        with torch.no_grad():
            if free:
                self.size += 1
            self.W[slot] = candidate
            self.birth_label[slot] = target
            self.usage[slot] = 0
            self.origin_id[slot] = origin_id
            self.atom_uid[slot] = uid
        after = self._retrieve_evidence(query, prior_mode=baseline.prior_mode)
        selected_at = torch.nonzero(after.slot_ids == slot, as_tuple=False)
        if selected_at.numel():
            with torch.no_grad():
                self.usage[slot] = after.responsibilities[int(selected_at[0, 0])]
        gain = self._compact_loss_bits(baseline, target) - self._compact_loss_bits(after, target)
        self.next_atom_uid += 1
        return (
            CategoricalInsertReport(
                True,
                True,
                slot,
                uid,
                evicted_uid,
                self._compact_loss_bits(baseline, target),
                gain,
                "forced",
            ),
            after,
        )

    def _insert(
        self,
        query: TernaryQuery,
        target: int,
        baseline: CategoricalReadResult,
        origin_id: int,
    ) -> tuple[CategoricalInsertReport, CategoricalReadResult]:
        if query.nnz == 0:
            return (
                CategoricalInsertReport(
                    True,
                    False,
                    None,
                    None,
                    None,
                    _loss_bits(baseline, target),
                    0.0,
                    "empty-query",
                ),
                baseline,
            )
        support = query.indices[: self.config.key_nnz]
        signs = query.values[: self.config.key_nnz]
        candidate = torch.zeros(self.config.dimension, dtype=torch.float32, device=self.device)
        candidate[support] = signs * (self.config.key_scale / support.numel())
        free = self.size < self.config.capacity
        slot = self.size if free else int(torch.argmin(self.usage[: self.size]).item())
        evicted_uid = None if free else int(self.atom_uid[slot].item())
        uid = self.next_atom_uid
        with torch.no_grad():
            if free:
                self.size += 1
            self.W[slot] = candidate
            self.birth_label[slot] = target
            self.usage[slot] = 0
            self.origin_id[slot] = origin_id
            self.atom_uid[slot] = uid
        after = self.read(query, prior_mode=baseline.prior_mode)
        selected_at = torch.nonzero(after.slot_ids == slot, as_tuple=False)
        if selected_at.numel():
            with torch.no_grad():
                self.usage[slot] = after.responsibilities[int(selected_at[0, 0])]
        gain = _loss_bits(baseline, target) - _loss_bits(after, target)
        self.next_atom_uid += 1
        return (
            CategoricalInsertReport(
                True,
                True,
                slot,
                uid,
                evicted_uid,
                _loss_bits(baseline, target),
                gain,
                "forced",
            ),
            after,
        )

    def observe(
        self,
        query: TernaryQuery,
        target: int,
        *,
        pre_read: CategoricalReadResult | None = None,
        origin_id: int = -1,
        insertion_mode: Literal["force", "skip"] = "skip",
    ) -> CategoricalObserveReport:
        if insertion_mode not in {"force", "skip"}:
            raise ValueError("categorical memory supports only force or skip insertion")
        if not 0 <= target < self.config.num_classes:
            raise ValueError("target outside class space")
        query = self._validate_query(query)
        before = self.read(query) if pre_read is None else pre_read
        if before.owner_token != self._owner_token:
            raise RuntimeError("ReadResult belongs to another memory")
        if before.state_revision != self.revision:
            raise RuntimeError("stale ReadResult")
        if before.query_signature != _query_signature(query):
            raise ValueError("pre_read belongs to another query")
        rollback_slots = before.slot_ids
        if insertion_mode == "force" and self.size >= self.config.capacity:
            # Usage is updated before eviction and can change which slot is
            # least-used. Snapshot every active key for this rare destructive
            # path so observe remains atomic even if insertion later fails.
            rollback_slots = torch.arange(self.size, device=self.device)
        snapshot = (
            rollback_slots,
            self.W[rollback_slots].clone(),
            self.birth_label.clone(),
            self.usage.clone(),
            self.origin_id.clone(),
            self.atom_uid.clone(),
            int(self.base_counts[target]),
            self.size,
            self.step,
            self.revision,
            self.next_atom_uid,
            self.base_count_argmax,
        )
        try:
            learn, after_learn = self._learn(before, target)
            self._update_usage(before)
            if insertion_mode == "force":
                insertion, _ = self._insert(query, target, after_learn, origin_id)
            else:
                insertion = CategoricalInsertReport(
                    False,
                    False,
                    None,
                    None,
                    None,
                    _loss_bits(after_learn, target),
                    0.0,
                    "skipped",
                )
            self.base_counts[target] += 1
            current = int(self.base_counts[target])
            best = int(self.base_counts[self.base_count_argmax])
            if current > best or (current == best and target < self.base_count_argmax):
                self.base_count_argmax = target
            self.step += 1
            self.revision += 1
        except BaseException:
            with torch.no_grad():
                self.W[snapshot[0]] = snapshot[1]
                self.birth_label[:] = snapshot[2]
                self.usage[:] = snapshot[3]
                self.origin_id[:] = snapshot[4]
                self.atom_uid[:] = snapshot[5]
                self.base_counts[target] = snapshot[6]
            (
                self.size,
                self.step,
                self.revision,
                self.next_atom_uid,
                self.base_count_argmax,
            ) = snapshot[7:]
            raise
        after = self.read(query, prior_mode=before.prior_mode)
        return CategoricalObserveReport(before, after, learn, insertion, target)

    def observe_compact(
        self,
        query: TernaryQuery,
        target: int,
        *,
        origin_id: int = -1,
        insertion_mode: Literal["force", "skip"] = "skip",
    ) -> CategoricalCompactObserveReport:
        if insertion_mode not in {"force", "skip"}:
            raise ValueError("categorical memory supports only force or skip insertion")
        if not 0 <= target < self.config.num_classes:
            raise ValueError("target outside class space")
        query = self._validate_query(query)
        before = self._retrieve_evidence(query)
        rollback_slots = before.slot_ids
        if insertion_mode == "force" and self.size >= self.config.capacity:
            rollback_slots = torch.arange(self.size, device=self.device)
        snapshot = (
            rollback_slots,
            self.W[rollback_slots].clone(),
            self.birth_label.clone(),
            self.usage.clone(),
            self.origin_id.clone(),
            self.atom_uid.clone(),
            int(self.base_counts[target]),
            self.size,
            self.step,
            self.revision,
            self.next_atom_uid,
            self.base_count_argmax,
        )
        try:
            learn, after_learn = self._learn_compact(before, target)
            self._update_usage(before)
            if insertion_mode == "force":
                insertion, _ = self._insert_compact(query, target, after_learn, origin_id)
            else:
                insertion = CategoricalInsertReport(
                    False,
                    False,
                    None,
                    None,
                    None,
                    self._compact_loss_bits(after_learn, target),
                    0.0,
                    "skipped",
                )
            self.base_counts[target] += 1
            current = int(self.base_counts[target])
            best = int(self.base_counts[self.base_count_argmax])
            if current > best or (current == best and target < self.base_count_argmax):
                self.base_count_argmax = target
            self.step += 1
            self.revision += 1
        except BaseException:
            with torch.no_grad():
                self.W[snapshot[0]] = snapshot[1]
                self.birth_label[:] = snapshot[2]
                self.usage[:] = snapshot[3]
                self.origin_id[:] = snapshot[4]
                self.atom_uid[:] = snapshot[5]
                self.base_counts[target] = snapshot[6]
            (
                self.size,
                self.step,
                self.revision,
                self.next_atom_uid,
                self.base_count_argmax,
            ) = snapshot[7:]
            raise
        after = self._retrieve_evidence(query, prior_mode=before.prior_mode)
        return CategoricalCompactObserveReport(
            before.prediction,
            after.prediction,
            learn,
            insertion,
            target,
        )

    def _restore(self, other: CategoricalAssociativeMemory) -> None:
        if self._class_capacity < other._class_capacity:
            self.expand_classes(
                other.config.num_classes,
                reserve=other._class_capacity,
            )
        with torch.no_grad():
            for name in ("W", "birth_label", "usage", "origin_id", "atom_uid"):
                getattr(self, name).copy_(getattr(other, name))
            self.base_counts[: other._class_capacity].copy_(
                other.base_counts[: other._class_capacity]
            )
        self.size = other.size
        self.step = other.step
        self.revision = other.revision
        self.next_atom_uid = other.next_atom_uid
        self.base_count_argmax = other.base_count_argmax

    def clone(self) -> CategoricalAssociativeMemory:
        other = CategoricalAssociativeMemory(self.config, device=self.device)
        other._restore(self)
        return other

    def allocated_bytes(self) -> int:
        tensors = (
            self.W,
            self.birth_label,
            self.usage,
            self.origin_id,
            self.atom_uid,
            self.base_counts,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def active_logical_bytes(self) -> int:
        per_atom = (
            self.config.dimension * self.W.element_size()
            + self.birth_label.element_size()
            + self.usage.element_size()
            + self.origin_id.element_size()
            + self.atom_uid.element_size()
        )
        return (
            self.size * per_atom
            + self.config.num_classes * self.base_counts.element_size()
        )
