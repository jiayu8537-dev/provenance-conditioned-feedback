# Manuscript-to-package crosswalk

| Manuscript component | Code/configuration | Primary auditable output |
|---|---|---|
| Data and temporal construction | `scripts/prepare_pixelrec_inputs.py` | `logs/data_preparation_validation.json`; `publication_assets/tables/Table1_data_temporal_panel_statistics.csv` |
| Static BPR and initialization | `src/bpr_training.py`; `configs/bpr.yaml` | `artifacts/bpr_manifest.json`; `logs/bpr_training.csv`; Table 2 source |
| Attribution assignment and balance | `src/data_loading.py`; `data/protocol/` | `raw/qualified_assignment_metadata_10.csv`; Table 2 source |
| Choice and matched trajectories | `src/choice_model.py`; `src/simulation.py`; `src/online_bpr.py` | `raw/main/`; `raw/main_corrected_round_level.csv.gz` |
| Prespecified response scenarios | `configs/main.yaml`; `src/choice_model.py` | Main-manuscript Table 3 |
| Neutral-drift equivalence assessment | `src/bootstrap.py` | Section 5.1: legacy-named source file `tables/Table3_zero_effect_equivalence.csv` |
| Recommendation/distribution outcomes | `src/exposure_metrics.py`; `src/bootstrap.py` | Fig. 3 and Section 5.2: `tables/Fig3_roundwise_AA_data.csv` and the legacy-named endpoint file `tables/Table4_BPR_algorithmic_amplification.csv` |
| Oracle/quota/combined controls | `src/interventions.py`; `src/quota_reranker.py` | Main-manuscript Table 4: legacy-named source file `tables/Table5_intervention_tradeoffs.csv` |
| Candidate, update, and oracle sensitivity | `run_all_cpu.py`; `configs/sensitivity.yaml` | three sensitivity CSV files in `tables/` |
| Choice-process sensitivity | `run_choice_process_sensitivity.py` | `tables/choice_process/` |
| LightGCN extension | `src/lightgcn_cpu.py`; `src/lightgcn_simulation.py` | `tables/extension/lightgcn_*` |
| Long horizons and history replay | `run_extension.py`; `configs/extension.yaml` | `tables/extension/long_horizon_*` |
| Round-synchronous updating | `run_robustness.py` | `tables/robustness/synchronous_schedule_differences.csv` and `update_accounting.csv` |
| Strict 15u/5i replication | `run_robustness.py`; `configs/robustness.yaml` | `tables/robustness/strict_*` and `robustness_endpoints.csv` |
| Main-manuscript Figs. 1–4 | `publication_assets/figures/main/` | exact PDF files used to compile the submitted manuscript |
| Online Resource 1 supplementary figures | `publication_assets/figures/supplementary/`; `publication_assets/make_jiis_final_figures.py` | SFig. 1 sensitivity analysis and SFig. 2 targeted validation, plus data-driven regeneration outputs |

The confirmatory BPR grid contains 5 panels × 10 attribution assignments ×
10 response streams. Targeted extensions use 2 × 5 × 5. These grids remain
separate in both code and tables.

The three legacy table filenames are retained to avoid breaking the audited
aggregation pipeline. Their current manuscript destinations are defined above;
the submitted main manuscript uses Table 3 for response scenarios and has no
Table 5.
