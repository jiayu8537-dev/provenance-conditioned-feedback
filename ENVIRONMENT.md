# Computational environment

## Locked software

- Python 3.12
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.15.3
- Numba 0.61.2
- PyTorch 2.2.2 (CPU execution)
- PyYAML 6.0.3
- Matplotlib 3.10.8
- Pillow 11.3.0
- pytest 8.4.2

Install from `requirements.txt` in a new virtual environment. The package has
no network dependency after Python packages and the separately licensed
PixelRec source files have been obtained.

## Determinism

NumPy, PyTorch, assignment, response, negative-selection, user-panel,
candidate-pool, and bootstrap seeds are fixed. `SEED_MANIFEST.csv` identifies
each stream and its derivation. Gzip files written by the preparation script
use a zero modification timestamp. Small last-decimal differences in BLAS- or
PyTorch-dependent floating-point operations can occur across architectures;
the validation checks use exact values for delivered tables and tolerances
only for recomputed numerical summaries.

The archived summary log records NumPy 2.2.6, whereas the supported
end-to-end environment pins NumPy 1.26.4 for compatibility with the macOS
x86_64 PyTorch 2.2.2 wheel. This changes only the multinomial draw sequence
used for percentile-bootstrap interval endpoints. `run_pipeline.py audit`
requires point estimates and conclusions to agree exactly and bounds the
largest interval-endpoint difference at 0.002; the exact reported intervals
remain preserved in the checksum-locked delivered tables.

## Resource expectations

The package is CPU-first. The archived run used Python 3.12 on macOS x86_64.
The recorded shard timings sum to approximately 2.4 CPU-hours for the main and
original sensitivity grids, 2.7 CPU-hours for the LightGCN/horizon extensions,
and under 15 minutes for the targeted schedule, strict-core, and choice
analyses. Wall time varies with core count, BLAS implementation, and disk.
Run panel shards concurrently only when sufficient memory is available.

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` may be set to
control oversubscription. The analysis code caps PyTorch CPU threads at eight.

## Two validation modes

- Archive audit: tests, aggregate-only regeneration, and package verification;
  no PixelRec data are required.
- Full rerun: reconstruct licensed inputs, set `JIIS_FORCE_RETRAIN=1` if static
  initializations should be refitted instead of loading the delivered fitted
  artifacts, and run `scripts/run_pipeline.py full`.
