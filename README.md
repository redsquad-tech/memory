# assocmem

`assocmem` is a CPU-first reference implementation of an online predictive
associative memory. It learns only the atoms retrieved for the current query,
can insert a new atom after surprising feedback, and evaluates every event
before updating on it.

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

It downloads and verifies BANKING77 and runs the complete v2 comparison on
seeds 0–4. The encoder and its hash seed are identical for every model and
seed; only the order of the online training stream changes.

The experiment contains two complementary learned-key versus frozen-key
comparisons:

- `shared_*`: both branches receive exactly the same deterministic set of atom
  births, with identical initial keys and values. This isolates key learning.
- `natural_*`: both branches use the normal surprise/novelty insertion policy.
  This measures the end-to-end system, including interactions between learned
  geometry and insertion.

Keys have 256 fixed support coordinates: the learned branch can change their
weights, while the frozen branch cannot change keys. Exact kNN uses the same
energy, `top_k=16`, the empirical online class prior, and retains all 10,003
training examples while remaining within the same 128 MiB byte budget. The
online linear baseline uses the provisional post-hoc learning rate `0.03`.

The command writes two directly usable artifacts in the repository root:

- `banking77_results_v2.csv`: six aggregate model rows plus paired comparison
  rows with 95% t intervals and favorable-seed counts.
- `banking77_runs_v2.csv`: one numeric row per model and seed, including all
  parameters and protocol flags.

Both the full official test result and a sensitivity result excluding the 25
normalized train/test text duplicates are reported. The v2 files explicitly
mark that this official test set was previously inspected and that the chosen
hyperparameters are provisional/post-hoc; they are not an untouched
confirmatory result. The earlier run is preserved as
`banking77_results_v1.csv`.

## Validation

```bash
.venv/bin/pytest
```

The dense memory is the correctness oracle. `SparseExactIndex` remains an
isolated post-gate prototype and is not used by the experiment suite. Byte-level
utilities live in `assocmem.byte` and do not carry context across
train/validation/test boundaries.
