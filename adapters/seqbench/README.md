# assocmem adapter for seqbench

This directory is the only integration boundary between `assocmem` and the
external `redsquad-tech/seqbench` project.

```bash
seqbench run /path/to/seqbench/specs/runs/full_v1.yaml \
  --algorithm adapters/seqbench/algorithm.yaml \
  --tasks /path/to/full.csv \
  --output runs/seqbench
```

The adapter implements the named process operations `learn` and `infer`.
Unknown output strings have exact probability zero and are transported as
`log_probability: null`.
