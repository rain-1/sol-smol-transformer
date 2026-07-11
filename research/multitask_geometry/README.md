# Multi-task representation geometry

Run from the repository root:

```bash
python research/multitask_geometry/analyze.py
```

The analysis is deterministic for a fixed checkpoint and seed. It evaluates
paired examples: every task receives the same digit strings at each length.
See `../../reports/multitask-representations.md` for interpretation and
`results.json` for all measurements.
