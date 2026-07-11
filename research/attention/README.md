# Attention research

`analyze.py` trains the four preset-sized models with a fixed seed, evaluates
256 fresh sequences at every digit length 2–8, and writes aggregate head
statistics to `results.json`. It deliberately avoids selecting illustrative
examples. Reproduce with:

```bash
python research/attention/analyze.py --steps 2000 --examples 256 --seed 42
```
