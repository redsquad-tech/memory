from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass

import torch

from .config import MemoryConfig
from .encoding import TernaryQuery


@dataclass(frozen=True)
class ReadResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    prediction: torch.Tensor
    slot_ids: torch.Tensor
    atom_uids: torch.Tensor
    origin_ids: torch.Tensor
    energies: torch.Tensor
    responsibilities: torch.Tensor
    background_responsibility: torch.Tensor
    query: TernaryQuery
    state_revision: int
    owner_token: str
    query_signature: str


@dataclass(frozen=True)
class LearnReport:
    applied: bool
    attempts: int
    selected_slots: tuple[int, ...]
    key_gradient_norm: float
    value_gradient_norm: float
    loss_before_bits: float
    loss_after_bits: float
    support_added: int
    support_removed: int


@dataclass(frozen=True)
class InsertReport:
    attempted: bool
    inserted: bool
    slot_id: int | None
    atom_uid: int | None
    evicted_atom_uid: int | None
    surprise_bits: float
    gain_bits: float
    reason: str


@dataclass(frozen=True)
class ObserveReport:
    read_before: ReadResult
    read_after: ReadResult
    learn: LearnReport
    insertion: InsertReport
    target: int


def _loss_bits(read: ReadResult, target: int) -> float:
    probability = read.probabilities[target].clamp_min(torch.finfo(torch.float32).tiny)
    return float((-torch.log2(probability)).detach().cpu())


def _clip_rows(gradient: torch.Tensor, maximum: float) -> torch.Tensor:
    norms = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return gradient * (maximum / norms).clamp_max(1.0)


def _query_signature(query: TernaryQuery) -> str:
    payload = (
        query.dimension,
        query.encoder_fingerprint,
        query.indices.detach().cpu().tolist(),
        query.values.detach().cpu().tolist(),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


class AssociativeMemory:
    def __init__(self, config: MemoryConfig, *, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.W = torch.zeros(
            (config.capacity, config.dimension), dtype=torch.float32, device=self.device
        )
        self.V = torch.zeros(
            (config.capacity, config.num_classes), dtype=torch.float32, device=self.device
        )
        self.usage = torch.zeros(config.capacity, dtype=torch.float32, device=self.device)
        self.origin_id = torch.full((config.capacity,), -1, dtype=torch.int64, device=self.device)
        self.atom_uid = torch.full((config.capacity,), -1, dtype=torch.int64, device=self.device)
        self.base_counts = torch.zeros(config.num_classes, dtype=torch.int64, device=self.device)
        self.size = 0
        self.step = 0
        self.revision = 0
        self.next_atom_uid = 0
        self._owner_token = uuid.uuid4().hex

    def _validate_query(self, query: TernaryQuery) -> TernaryQuery:
        if query.dimension != self.config.dimension:
            raise ValueError("query and memory dimensions differ")
        expected = self.config.encoder_fingerprint
        if expected is not None and query.encoder_fingerprint != expected:
            raise ValueError("query encoder fingerprint does not match memory")
        return query.snapshot(device=self.device)

    def _base_logits(self) -> torch.Tensor:
        counts = self.base_counts.to(torch.float64)
        pseudo = self.config.prior_pseudocount
        probabilities = (counts + pseudo) / (counts.sum() + pseudo * self.config.num_classes)
        return probabilities.log().to(torch.float32)

    def read(self, query: TernaryQuery) -> ReadResult:
        query = self._validate_query(query)
        base_logits = self._base_logits()
        if self.size == 0:
            probabilities = torch.softmax(base_logits, dim=0)
            empty_float = torch.empty(0, dtype=torch.float32, device=self.device)
            empty_int = torch.empty(0, dtype=torch.int64, device=self.device)
            return ReadResult(
                base_logits,
                probabilities,
                probabilities.argmax(),
                empty_int,
                empty_int,
                empty_int,
                empty_float,
                empty_float,
                torch.ones((), dtype=torch.float32, device=self.device),
                query,
                self.revision,
                self._owner_token,
                _query_signature(query),
            )
        if query.nnz:
            dot = self.W[: self.size, query.indices] @ query.values
        else:
            dot = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        energies_all = (0.5 * (self.config.key_scale - dot)).clamp_min(0)
        order = torch.argsort(energies_all, stable=True)
        k = self.size if self.config.top_k is None else min(self.config.top_k, self.size)
        slots = order[:k]
        energies = energies_all[slots]
        log_mass = -math.log(2.0) * energies
        log_z = torch.logsumexp(torch.cat([torch.zeros(1, device=self.device), log_mass]), dim=0)
        responsibilities = torch.exp(log_mass - log_z)
        background = torch.exp(-log_z)
        logits = base_logits + (responsibilities[:, None] * self.V[slots]).sum(dim=0)
        probabilities = torch.softmax(logits, dim=0)
        return ReadResult(
            logits.detach(),
            probabilities.detach(),
            probabilities.argmax().detach(),
            slots.detach(),
            self.atom_uid[slots].detach().clone(),
            self.origin_id[slots].detach().clone(),
            energies.detach(),
            responsibilities.detach(),
            background.detach(),
            query,
            self.revision,
            self._owner_token,
            _query_signature(query),
        )

    def _project_keys(
        self,
        keys: torch.Tensor,
        *,
        old_keys: torch.Tensor | None = None,
        gradients: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = torch.zeros_like(keys)
        for row in range(keys.shape[0]):
            candidate = keys[row].clone()
            if self.config.update.fixed_key_support:
                support = old_keys[row] != 0 if old_keys is not None else candidate != 0
                if not bool(support.any()):
                    raise FloatingPointError("fixed-support projection has no support")
                values = candidate[support]
                if old_keys is not None:
                    tiny = values.abs() < 1e-8
                    values[tiny] = old_keys[row, support][tiny].sign() * 1e-8
                values = values * (self.config.key_scale / values.abs().sum())
                projected[row, support] = values
                continue
            if (
                old_keys is not None
                and gradients is not None
                and self.config.update.support_replacements
            ):
                old_support = torch.nonzero(old_keys[row] != 0, as_tuple=False).flatten()
                outside = old_keys[row] == 0
                if old_support.numel() and bool(outside.any()):
                    outside_scores = gradients[row].abs().masked_fill(~outside, -1)
                    replacements = min(
                        self.config.update.support_replacements,
                        int(old_support.numel()),
                        int(outside.sum().item()),
                    )
                    victims = old_support[
                        torch.argsort(old_keys[row, old_support].abs(), stable=True)[:replacements]
                    ]
                    additions = torch.argsort(outside_scores, descending=True, stable=True)[
                        :replacements
                    ]
                    useful = outside_scores[additions] > 0
                    victims, additions = victims[useful], additions[useful]
                    if additions.numel():
                        magnitude = old_keys[row, victims].abs().clamp_min(1e-6)
                        candidate[victims] = 0
                        candidate[additions] = -gradients[row, additions].sign() * magnitude
            order = torch.argsort(candidate.abs(), descending=True, stable=True)
            keep = order[: self.config.key_nnz]
            values = candidate[keep]
            nonzero = values.abs() > 0
            if not bool(nonzero.any()):
                raise FloatingPointError("key projection produced an empty active key")
            keep = keep[nonzero]
            values = values[nonzero]
            values = values * (self.config.key_scale / values.abs().sum())
            projected[row, keep] = values
        return projected

    def _project_values(self, values: torch.Tensor) -> torch.Tensor:
        values = values - values.mean(dim=1, keepdim=True)
        norms = values.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return values * (self.config.update.value_max_norm / norms).clamp_max(1.0)

    def _local_gradients(
        self, read: ReadResult, target: int
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        slots = read.slot_ids
        keys = self.W[slots].detach().clone().requires_grad_(self.config.update.train_keys)
        values = self.V[slots].detach().clone().requires_grad_(True)
        query = read.query
        dot = keys[:, query.indices] @ query.values if query.nnz else keys.sum(dim=1) * 0
        energies = 0.5 * (keys.abs().sum(dim=1) - dot)
        log_mass = -math.log(2.0) * energies
        log_z = torch.logsumexp(torch.cat([torch.zeros(1, device=self.device), log_mass]), dim=0)
        responsibilities = torch.exp(log_mass - log_z)
        logits = self._base_logits() + (responsibilities[:, None] * values).sum(dim=0)
        loss = torch.nn.functional.cross_entropy(
            logits[None, :], torch.tensor([target], device=self.device)
        )
        inputs = [values]
        if self.config.update.train_keys:
            inputs.insert(0, keys)
        gradients = torch.autograd.grad(loss, inputs)
        if self.config.update.train_keys:
            key_gradient, value_gradient = gradients
        else:
            key_gradient = torch.zeros_like(keys)
            value_gradient = gradients[0]
        return (
            key_gradient,
            value_gradient,
            float(key_gradient.norm().detach().cpu()),
            float(value_gradient.norm().detach().cpu()),
        )

    def _learn(self, read: ReadResult, target: int) -> tuple[LearnReport, ReadResult]:
        before = _loss_bits(read, target)
        slots = read.slot_ids
        if slots.numel() == 0:
            report = LearnReport(False, 0, (), 0.0, 0.0, before, before, 0, 0)
            return report, read
        key_grad, value_grad, key_norm, value_norm = self._local_gradients(read, target)
        key_grad = _clip_rows(key_grad, self.config.update.gradient_clip_norm)
        value_grad = _clip_rows(value_grad, self.config.update.gradient_clip_norm)
        old_keys = self.W[slots].clone()
        old_values = self.V[slots].clone()
        accepted: ReadResult | None = None
        attempts = 0
        with torch.no_grad():
            for attempt in range(self.config.update.backtracking_attempts):
                attempts = attempt + 1
                factor = self.config.update.backtracking_factor**attempt
                try:
                    candidate_keys = old_keys
                    if self.config.update.train_keys:
                        candidate_keys = self._project_keys(
                            old_keys - self.config.update.learning_rate_key * factor * key_grad,
                            old_keys=old_keys,
                            gradients=key_grad,
                        )
                    candidate_values = self._project_values(
                        old_values - self.config.update.learning_rate_value * factor * value_grad
                    )
                except FloatingPointError:
                    continue
                if not bool(
                    torch.isfinite(candidate_keys).all() and torch.isfinite(candidate_values).all()
                ):
                    continue
                self.W[slots] = candidate_keys
                self.V[slots] = candidate_values
                candidate_read = self.read(read.query)
                after = _loss_bits(candidate_read, target)
                if after <= before + self.config.update.acceptance_tolerance_bits:
                    accepted = candidate_read
                    break
                self.W[slots] = old_keys
                self.V[slots] = old_values
            if accepted is None:
                self.W[slots] = old_keys
                self.V[slots] = old_values
        final_read = accepted if accepted is not None else self.read(read.query)
        final_keys = self.W[slots]
        added = int(((old_keys == 0) & (final_keys != 0)).sum().item())
        removed = int(((old_keys != 0) & (final_keys == 0)).sum().item())
        report = LearnReport(
            accepted is not None,
            attempts,
            tuple(int(v) for v in slots.detach().cpu()),
            key_norm,
            value_norm,
            before,
            _loss_bits(final_read, target),
            added,
            removed,
        )
        return report, final_read

    def _update_usage(self, read: ReadResult) -> None:
        with torch.no_grad():
            self.usage[: self.size].mul_(self.config.eviction.usage_decay)
            if read.slot_ids.numel():
                self.usage[read.slot_ids].add_(read.responsibilities)

    def _insert(
        self,
        query: TernaryQuery,
        target: int,
        baseline: ReadResult,
        origin_id: int,
        *,
        force: bool = False,
        reference_probabilities: torch.Tensor | None = None,
    ) -> tuple[InsertReport, ReadResult]:
        surprise = _loss_bits(baseline, target)
        cfg = self.config.insertion
        if not force and not cfg.enabled:
            return InsertReport(False, False, None, None, None, surprise, 0.0, "disabled"), baseline
        if not force and surprise <= cfg.surprise_threshold(self.config.num_classes):
            return InsertReport(
                False, False, None, None, None, surprise, 0.0, "below-threshold"
            ), baseline
        if query.nnz == 0:
            return InsertReport(
                True, False, None, None, None, surprise, 0.0, "empty-query"
            ), baseline
        if not force and self.size == 0 and not cfg.bootstrap_when_empty:
            return InsertReport(
                True, False, None, None, None, surprise, 0.0, "bootstrap-disabled"
            ), baseline
        if not force and self.size:
            if (
                baseline.energies.numel() == 0
                or float(baseline.energies.min()) < cfg.novelty_energy_ratio * self.config.key_scale
            ):
                return InsertReport(
                    True, False, None, None, None, surprise, 0.0, "not-novel-energy"
                ), baseline
            if float(baseline.background_responsibility) < cfg.min_background_responsibility:
                return InsertReport(
                    True, False, None, None, None, surprise, 0.0, "not-novel-background"
                ), baseline

        support = query.indices[: self.config.key_nnz]
        signs = query.values[: self.config.key_nnz]
        candidate_key = torch.zeros(self.config.dimension, dtype=torch.float32, device=self.device)
        candidate_key[support] = signs * (self.config.key_scale / support.numel())
        free = self.size < self.config.capacity
        slot = self.size if free else int(torch.argmin(self.usage[: self.size]).item())
        evicted_uid = None if free else int(self.atom_uid[slot].item())
        old = (
            self.W[slot].clone(),
            self.V[slot].clone(),
            self.usage[slot].clone(),
            self.origin_id[slot].clone(),
            self.atom_uid[slot].clone(),
        )
        temporary_uid = self.next_atom_uid
        try:
            with torch.no_grad():
                if free:
                    self.size += 1
                self.W[slot] = candidate_key
                self.V[slot].zero_()
                self.usage[slot] = 0
                self.origin_id[slot] = origin_id
                self.atom_uid[slot] = temporary_uid
                zero_read = self.read(query)
                probabilities = (
                    zero_read.probabilities
                    if reference_probabilities is None
                    else reference_probabilities
                )
                residual = -probabilities
                residual = residual.clone()
                residual[target] += 1
                candidate_value = cfg.value_scale * residual
                candidate_value = self._project_values(candidate_value[None, :])[0]
                self.V[slot] = candidate_value
                candidate_read = self.read(query)
                selected = bool((candidate_read.slot_ids == slot).any())
                gain = _loss_bits(baseline, target) - _loss_bits(candidate_read, target)
                accepted = force or (
                    selected and math.isfinite(gain) and gain >= cfg.min_gain_bits
                )
                if accepted:
                    if selected:
                        selected_at = torch.nonzero(
                            candidate_read.slot_ids == slot, as_tuple=False
                        )[0, 0]
                        self.usage[slot] = candidate_read.responsibilities[selected_at]
                    self.next_atom_uid += 1
                    return (
                        InsertReport(
                            True,
                            True,
                            slot,
                            temporary_uid,
                            evicted_uid,
                            surprise,
                            gain,
                            "forced" if force else "accepted",
                        ),
                        candidate_read,
                    )
        except BaseException:
            with torch.no_grad():
                (
                    self.W[slot],
                    self.V[slot],
                    self.usage[slot],
                    self.origin_id[slot],
                    self.atom_uid[slot],
                ) = old
                if free:
                    self.size -= 1
            raise
        with torch.no_grad():
            (
                self.W[slot],
                self.V[slot],
                self.usage[slot],
                self.origin_id[slot],
                self.atom_uid[slot],
            ) = old
            if free:
                self.size -= 1
        reason = "not-retrieved" if not selected else "insufficient-gain"
        return InsertReport(True, False, slot, None, None, surprise, gain, reason), baseline

    def observe(
        self,
        query: TernaryQuery,
        target: int,
        *,
        pre_read: ReadResult | None = None,
        origin_id: int = -1,
        insertion_mode: str = "auto",
        insertion_reference_probabilities: torch.Tensor | None = None,
    ) -> ObserveReport:
        if not 0 <= target < self.config.num_classes:
            raise ValueError("target outside class space")
        if not isinstance(origin_id, int) or not -(2**63) <= origin_id < 2**63:
            raise ValueError("origin_id must fit int64")
        if insertion_mode not in {"auto", "skip", "force"}:
            raise ValueError("insertion_mode must be auto, skip, or force")
        if insertion_reference_probabilities is not None:
            reference = insertion_reference_probabilities.to(
                device=self.device, dtype=torch.float32
            )
            if reference.shape != (self.config.num_classes,):
                raise ValueError("insertion reference has wrong class dimension")
            if (
                not bool(torch.isfinite(reference).all())
                or bool((reference < 0).any())
                or not torch.isclose(reference.sum(), torch.ones((), device=self.device))
            ):
                raise ValueError("insertion reference must be a probability distribution")
            insertion_reference_probabilities = reference
        if insertion_mode != "force" and insertion_reference_probabilities is not None:
            raise ValueError("insertion reference is only valid for forced insertion")
        query = self._validate_query(query)
        before = self.read(query) if pre_read is None else pre_read
        if before.owner_token != self._owner_token:
            raise RuntimeError("ReadResult belongs to another memory")
        if before.query_signature != _query_signature(before.query):
            raise RuntimeError("ReadResult query snapshot was mutated")
        if before.state_revision != self.revision:
            raise RuntimeError("stale ReadResult")
        if (
            before.query.dimension != query.dimension
            or not torch.equal(before.query.indices, query.indices)
            or not torch.equal(before.query.values, query.values)
        ):
            raise ValueError("pre_read belongs to another query")
        if before.query_signature != _query_signature(query):
            raise ValueError("pre_read belongs to another query")
        slots = before.slot_ids
        snapshot = (
            self.W[slots].clone(),
            self.V[slots].clone(),
            self.usage.clone(),
            self.origin_id.clone(),
            self.atom_uid.clone(),
            self.base_counts.clone(),
            self.size,
            self.step,
            self.revision,
            self.next_atom_uid,
        )
        try:
            learn_report, after_learn = self._learn(before, target)
            self._update_usage(before)
            if insertion_mode == "skip":
                insertion_report = InsertReport(
                    False,
                    False,
                    None,
                    None,
                    None,
                    _loss_bits(after_learn, target),
                    0.0,
                    "skipped",
                )
                after = after_learn
            else:
                insertion_report, after = self._insert(
                    query,
                    target,
                    after_learn,
                    origin_id,
                    force=insertion_mode == "force",
                    reference_probabilities=insertion_reference_probabilities,
                )
            with torch.no_grad():
                self.base_counts[target] += 1
            self.step += 1
            self.revision += 1
        except BaseException:
            with torch.no_grad():
                self.W[slots], self.V[slots] = snapshot[0], snapshot[1]
                self.usage[:] = snapshot[2]
                self.origin_id[:] = snapshot[3]
                self.atom_uid[:] = snapshot[4]
                self.base_counts[:] = snapshot[5]
            self.size, self.step, self.revision, self.next_atom_uid = snapshot[6:]
            raise
        after = self.read(query)
        return ObserveReport(before, after, learn_report, insertion_report, target)

    def clone(self) -> AssociativeMemory:
        other = AssociativeMemory(self.config, device=self.device)
        with torch.no_grad():
            for name in ("W", "V", "usage", "origin_id", "atom_uid", "base_counts"):
                getattr(other, name).copy_(getattr(self, name))
        other.size = self.size
        other.step = self.step
        other.revision = self.revision
        other.next_atom_uid = self.next_atom_uid
        return other

    def allocated_bytes(self) -> int:
        tensors = (self.W, self.V, self.usage, self.origin_id, self.atom_uid, self.base_counts)
        return sum(t.numel() * t.element_size() for t in tensors)

    def active_logical_bytes(self) -> int:
        per_atom = (
            self.config.dimension * self.W.element_size()
            + self.config.num_classes * self.V.element_size()
            + self.usage.element_size()
            + self.origin_id.element_size()
            + self.atom_uid.element_size()
        )
        return self.size * per_atom + self.base_counts.numel() * self.base_counts.element_size()
