from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from ..encoding import SignedHashTextEncoder
from ..online import OnlineExample


@dataclass(frozen=True)
class SyntheticConfig:
    examples: int = 20_000
    num_states: int = 64
    num_outputs: int = 16
    templates_per_state: int = 12
    noise_vocabulary: int = 128
    drift_at_fraction: float | None = 0.5
    random_labels: bool = False
    seed: int = 0


class PredictiveSemanticStream:
    def __init__(self, config: SyntheticConfig, encoder: SignedHashTextEncoder):
        self.config = config
        self.encoder = encoder
        rng = random.Random(config.seed)
        self._templates: list[list[str]] = []
        for state in range(config.num_states):
            family = state // 4
            templates = [
                f"surface_{state}_{variant} bridge_{family} {{noise}}"
                for variant in range(config.templates_per_state)
            ]
            self._templates.append(templates)
        generator = torch.Generator().manual_seed(config.seed)
        concentration = torch.full((config.num_outputs,), 0.25)
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            self._before = torch.distributions.Dirichlet(concentration).sample((config.num_states,))
        self._after = self._before.clone()
        affected = torch.randperm(config.num_states, generator=generator)[: config.num_states // 3]
        for state in affected:
            self._after[state] = self._before[state].roll(shifts=3)
        self._rng = rng
        self._torch_generator = generator

    def __iter__(self) -> Iterator[OnlineExample]:
        cfg = self.config
        drift_at = (
            cfg.examples + 1
            if cfg.drift_at_fraction is None
            else int(cfg.examples * cfg.drift_at_fraction)
        )
        for index in range(cfg.examples):
            state = self._rng.randrange(cfg.num_states)
            template = self._rng.choice(self._templates[state])
            noise = " ".join(
                f"noise_{self._rng.randrange(cfg.noise_vocabulary)}"
                for _ in range(self._rng.randrange(1, 4))
            )
            text = template.format(noise=noise)
            distribution = self._before[state] if index < drift_at else self._after[state]
            if cfg.random_labels:
                distribution = torch.full_like(distribution, 1 / cfg.num_outputs)
            target = int(torch.multinomial(distribution, 1, generator=self._torch_generator))
            yield OnlineExample(
                self.encoder.encode(text),
                target,
                example_id=index,
                hidden_state=state,
                true_distribution=distribution,
            )
