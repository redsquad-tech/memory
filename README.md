# assocmem

`assocmem` is a CPU-first reference implementation of online predictive
associative memory. It contains the original gated-logit memory and a stricter
categorical mixture memory used to isolate whether local key learning improves
predictive retrieval geometry.

The implementation intentionally uses a fixed ternary encoder and exact
retrieval. Active keys have a fixed non-zero L1 norm, preventing a zero key
from becoming a universal perfect match.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,experiments]'
```

## Library

```python
from assocmem import (
    AssociativeMemory,
    MemoryConfig,
    SignedHashTextEncoder,
    TextEncoderConfig,
)

encoder = SignedHashTextEncoder(TextEncoderConfig(dimension=8192))
memory = AssociativeMemory(
    MemoryConfig(
        dimension=8192,
        num_classes=77,
        capacity=2048,
        encoder_fingerprint=encoder.fingerprint,
    )
)

query = encoder.encode("my transfer is still pending")
prediction = memory.read(query)
report = memory.observe(query, target=12, pre_read=prediction)
```

`read` is side-effect free. `observe` rejects stale, foreign, or encoder-incompatible reads and atomically
performs local learning, optional insertion/eviction, usage accounting, and
the final prior update. Insertion requires both residual surprise and geometric
novelty; key updates may replace a bounded number of support coordinates and
are accepted only when the full-read loss does not increase.

## Experiments

The paper experiment is one command:

```bash
./run_banking77.sh
```

It runs the decoder-correction protocol on BANKING77 for seeds 0–4. There are
two causal experiments:

- Experiment A reproduces the old shared-birth learned/frozen memories, then
  evaluates the same final keys with one-hot birth labels and an arithmetic
  categorical mixture. This isolates the decoder without retraining keys.
- Experiment B trains learned and frozen keys directly under mixture NLL.
  Both branches receive the same 4,055 scheduled births; natural insertion and
  value learning are disabled.

The primary prior is uniform with mass `1`; empirical prior is an evaluation
sensitivity only. The fixed protocol uses `top_k=16`, key L1 norm `8`, 256
fixed support coordinates, and key learning rate `0.2`. Model selection is
reported on a frozen deterministic 80/20 split of the official train set.
Evaluation on the official test is exploratory because it was inspected by
the earlier v2 work.

The command writes two directly usable artifacts in the repository root:

- `banking77_mixture_results_v1.csv`: aggregate model rows and paired
  learned-minus-frozen effects with 95% t intervals.
- `banking77_mixture_runs_v1.csv`: numeric rows for every seed, stage, decoder,
  key mode, and prior sensitivity.

In addition to NLL, accuracy, macro-F1, Brier, and ECE, the new files report
weighted target purity, target retrieval mass, energy margin, Recall@1/4/16,
and MRR. The earlier v1/v2 artifacts and the old
`assocmem.experiments.banking77_paper` entry point are preserved for audit.

### Mixture v1 result

The frozen validation comparison is positive on all five seeds. Training keys
under mixture NLL reduces validation NLL from `2.0241` to `1.4929` bits and
raises accuracy from `73.17%` to `79.67%`; the paired advantages are `0.5312`
bits (95% t interval `[0.5165, 0.5458]`) and `6.50` percentage points
(`[6.24, 6.76]`). Weighted target purity rises from `0.431` to `0.575`, and
mean energy margin from `0.222` to `0.662`.

On the previously exposed official test, the corresponding learned model
reaches `1.4625` bits and `80.59%`. This exceeds the old v2 exact kNN accuracy
(`79.10%`) despite retaining only 4,055 scheduled atoms instead of all 10,003
examples, but remains below the old online linear reference (`88.41%`).

## Validation

```bash
.venv/bin/pytest
```

The dense memory is the correctness oracle. `SparseExactIndex` remains an
isolated post-gate prototype and is not used by the experiment suite. Byte-level
utilities live in `assocmem.byte` and do not carry context across
train/validation/test boundaries.
