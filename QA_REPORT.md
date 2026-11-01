# Independent extraction checks

Performed on 2026-11-01: `python -m pytest -q tests tests_il` in a fresh extraction of the candidate code archive, with one BLAS thread: **99 passed, 1 skipped**, no test errors. A separate smoke invocation of `python benchmark/benchmark_decision_latency.py --root . --out benchmark/local --cycles 1` completed. This smoke test is not a rerun of all 36 archived timing observations; the reported quantitative ratio of medians comes only from the archived full benchmark and is platform-specific.

This artifact deliberately does not claim all code, physical model assumptions or tugboat safety have been externally validated.
