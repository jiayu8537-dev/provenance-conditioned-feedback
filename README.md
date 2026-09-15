# JDSA Reproducibility Package

This package accompanies the manuscript **“Provenance-Conditioned Feedback and
Learning-Mediated Exposure Change in Online Recommender Systems: A
Matched-Trajectory Simulation Study”** by
Yu Jia and Shaojie Zhang. It is the complete reproducibility package for the
International Journal of Data Science and Analytics submission, not the earlier
BPR-only analysis folder. The confirmatory BPR module is joined by the upstream
data-construction workflow and the LightGCN, long-horizon, replay,
update-schedule, strict-core, choice-process, and intervention analyses reported
in the manuscript and Supplementary Information. It also contains the added
lower-feedback and roundwise candidate-refresh analyses.

The package supports three levels of verification:

1. `python scripts/verify_package.py` audits the delivered files, hashes,
   dimensions, experimental cells, numerical anchors, and disclosure rules.
2. `python scripts/run_pipeline.py audit` runs the tests and regenerates all
   summary tables and figures in a disposable copy, leaving the archive and
   its checksum manifest unchanged.
3. `python scripts/run_pipeline.py full` reruns the complete simulation suite
   after the licensed PixelRec inputs have been reconstructed.

## Package map

- `scripts/prepare_pixelrec_inputs.py`: deterministic reconstruction from the
  two official PixelRec50K CSV files;
- `run_all_cpu.py`: five-panel confirmatory BPR design and the candidate-size,
  update-count, and oracle-weight sensitivity analyses;
- `run_extension.py`: CPU LightGCN, 6/12/24-round BPR, and fixed-budget history
  replay;
- `run_robustness.py`: round-synchronous updating and strict 15u/5i
  within-dataset replication;
- `run_choice_process_sensitivity.py`: position and outside-option analyses;
- `run_feedback_candidate_sensitivity.py`: lower-feedback calibration and
  roundwise candidate-refresh analyses;
- `src/`: choice, ranking, online learning, intervention, metric, and inference
  implementations;
- `configs/`: locked analysis specifications;
- `data/protocol/`: non-identifying assignment-search diagnostics and the
  protocol arrays required to reconstruct the selected assignment;
- `artifacts/`: fitted static BPR and LightGCN initializations and their
  manifests;
- `raw/`: compressed, anonymized simulation outputs underlying the tables;
- `tables/`: main-text and extension results;
- `publication_assets/`: exact main-manuscript figures, supplementary figures,
  source tables, and a
  regeneration script;
- `logs/`: run and validation records;
- `tests/`: deterministic unit and integration tests.

## Quick audit in a clean environment

Python 3.12 is the locked interpreter family. From the extracted package root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_pipeline.py audit
```

The audit route uses only packaged simulation outputs; it does not require the
licensed source data. It requires exact agreement for point estimates,
dimensions, labels, and inferential conclusions. Percentile-bootstrap interval
endpoints are compared within 0.002 because the archived summary environment
used a later NumPy multinomial implementation than the PyTorch-compatible
end-to-end environment. See `ENVIRONMENT.md` and `RUN_ORDER.md`.

## Reconstructing licensed inputs

PixelRec data are not redistributed. After obtaining `interaction.csv` and
`item_info.csv` from the official PixelRec release, run:

```bash
.venv/bin/python scripts/prepare_pixelrec_inputs.py \
  --interaction /path/to/interaction.csv \
  --item-info /path/to/item_info.csv \
  --output-dir data/derived
```

The command refuses unrecognized downloads by default, reconstructs the
iterative 5u/3i, 10u/5i, and 15u/5i cores, applies the leave-last-two split,
rebuilds the model-aware assignment file, and verifies exact semantic hashes.
The delivered validation record is `logs/data_preparation_validation.json`.

## Scope and interpretation

The 5 × 10 × 10 BPR grid is the primary design. The expanded lower-feedback
comparison uses five panels, five assignments, and five response streams, with
the original high-feedback condition rerun on the same cells. Other targeted
extensions use the locked 2 × 5 × 5 reduced crossed design and are not pooled
with the confirmatory estimates. The strict 15u/5i analysis is a within-dataset structural
replication, not an external-dataset validation. The oracle correction uses
the known simulated response parameters and is a mechanism benchmark, not a
deployable estimator.

The 10%, 20%, and 40% acceptance targets are design conditions, not empirical
calibration claims. Candidate refresh and lower feedback are varied separately:
the former retains `u0 = 0.5`, and the latter retains fixed candidate pools.
The five-panel lower-feedback outputs and their matched high-feedback reference
are stored in the `lower_acceptance_expanded_*` files. The paired file defines
each record as \(\lvert\mathrm{LMC}_{\text{target},j}\rvert-
\lvert\mathrm{LMC}_{\text{high},j}\rvert\) within matched cell \(j\), before
crossed-bootstrap aggregation over panels, assignments, and response streams.
It supplies Online Resource 1, Table S2, Panel C. All nine point estimates are
negative; eight 95% intervals exclude zero, while the 40% AI-appreciation
interval includes zero. `lower_acceptance_endpoints.csv` is retained as a
compatibility view of the current five-panel target-condition estimates, not
as a separate two-panel analysis.

All paths are relative to the package root. No source-data row, personal path,
credential, or machine-specific mount point is included.

## License

Repository software is released under the MIT License. Original documentation,
result tables, and figures are released under CC BY 4.0. PixelRec is not
redistributed and remains governed by the terms of its authors and official
data provider. See `LICENSE.md` for the complete licensing notice.
