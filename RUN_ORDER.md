# Reproduction order

All commands are run from the package root with the clean environment
described in `ENVIRONMENT.md`.

## A. Verify the delivered archive

```bash
.venv/bin/python scripts/run_pipeline.py audit
```

This is the fastest route for editors and reviewers. It verifies the immutable
archive, runs 23 tests, and recreates the tables and figures from the included
unit- and round-level outputs in a disposable copy.

## B. Reconstruct the source-dependent inputs

```bash
.venv/bin/python scripts/prepare_pixelrec_inputs.py \
  --interaction /path/to/interaction.csv \
  --item-info /path/to/item_info.csv \
  --output-dir data/derived
```

The command must end with `"status": "passed"`. Do not continue after a hash
or dimension failure.

## C. Complete rerun

```bash
export JIIS_DATA_ROOT="$PWD/data/derived"
export JIIS_FORCE_RETRAIN=1
.venv/bin/python scripts/run_pipeline.py full
```

The orchestrator executes, in order: static BPR; five confirmatory panels;
candidate-size, online-update, and oracle-weight sensitivity grids; CPU
LightGCN; 6/12/24-round event-online and history-replay trajectories;
round-synchronous BPR; strict 15u/5i replication; position/outside-option
sensitivity; aggregation; figures; tests; and package validation.

For cluster scheduling, the same commands can be issued independently by
following the subcommands shown by `python run_all_cpu.py --help`,
`python run_extension.py --help`, `python run_robustness.py --help`, and
`python run_choice_process_sensitivity.py --help`. Corresponding panel shards
must not be mixed across different configuration files or reconstructed input
hashes.
