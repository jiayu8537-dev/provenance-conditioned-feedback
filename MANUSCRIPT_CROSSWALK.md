# Manuscript-to-package crosswalk

| Manuscript component | Code/configuration | Primary auditable output |
|---|---|---|
| Data and temporal construction | `scripts/prepare_pixelrec_inputs.py` | `logs/data_preparation_validation.json`; `publication_assets/tables/Table1_data_temporal_panel_statistics.csv` |
| Static BPR and initialization | `src/bpr_training.py`; `configs/bpr.yaml` | `artifacts/bpr_manifest.json`; `logs/bpr_training.csv`; Table 2 source |
| Attribution assignment and balance | `src/data_loading.py`; `data/protocol/` | `raw/qualified_assignment_metadata_10.csv`; Table 2 source |
| Choice and matched trajectories | `src/choice_model.py`; `src/simulation.py`; `src/online_bpr.py` | `raw/main/`; `raw/main_corrected_round_level.csv.gz` |
| Prespecified response scenarios | `configs/main.yaml`; `src/choice_model.py` | Main-manuscript Table 3 |
| Neutral-drift equivalence assessment | `src/bootstrap.py` | Section 5.1: legacy-named source file `tables/Table3_zero_effect_equivalence.csv` |
| Primary LMC trajectories | `src/exposure_metrics.py`; `src/bootstrap.py` | Main-manuscript Fig. 2 and Section 5.1: `tables/Fig3_roundwise_AA_data.csv` and the legacy-named endpoint file `tables/Table4_BPR_algorithmic_amplification.csv` |
| Oracle/quota/combined benchmarks | `src/interventions.py`; `src/quota_reranker.py` | Main-manuscript Fig. 4 and Table 5: legacy-named source file `tables/Table5_intervention_tradeoffs.csv` |
| Candidate, update, and oracle sensitivity | `run_all_cpu.py`; `configs/sensitivity.yaml` | three sensitivity CSV files in `tables/` |
| Choice-process sensitivity | `run_choice_process_sensitivity.py` | `tables/choice_process/` |
| Lower-feedback sensitivity | `run_feedback_candidate_sensitivity.py`; `configs/feedback_candidate_sensitivity.yaml` | Five-panel outputs `tables/added_sensitivity/lower_acceptance_expanded_endpoints.csv` and `lower_acceptance_expanded_paired_vs_high.csv`; archived round-level output; main-manuscript Fig. 3(a); Online Resource 1, Table S2, Panels A–C |
| Roundwise candidate refresh | `run_feedback_candidate_sensitivity.py`; `configs/feedback_candidate_sensitivity.yaml` | `tables/added_sensitivity/dynamic_candidates_endpoints.csv`; `dynamic_minus_fixed_paired_LMC.csv`; archived round-level output; main-manuscript Fig. 3(b); Online Resource 1, Table S2, Panel D |
| LightGCN extension | `src/lightgcn_cpu.py`; `src/lightgcn_simulation.py` | `tables/extension/lightgcn_*` |
| Long horizons and history replay | `run_extension.py`; `configs/extension.yaml` | `tables/extension/long_horizon_*` |
| Round-synchronous updating | `run_robustness.py` | `tables/robustness/synchronous_schedule_differences.csv` and `update_accounting.csv` |
| Strict 15u/5i replication | `run_robustness.py`; `configs/robustness.yaml` | `tables/robustness/strict_*` and `robustness_endpoints.csv` |
| Main-manuscript Figs. 1–4 | `publication_assets/figures/main/` | exact PDF files corresponding to the submitted manuscript |
| Supplementary figures | `publication_assets/figures/supplementary/`; `publication_assets/make_jdsa_final_figures.py` | Fig. S1 sensitivity analysis and Fig. S2 targeted validation, plus data-driven regeneration outputs |

The confirmatory BPR grid contains 5 panels × 10 attribution assignments ×
10 response streams. The expanded lower-feedback comparison uses 5 × 5 × 5,
including a high-feedback reference on the same cells. Other targeted extensions
use 2 × 5 × 5. These grids remain separate in both code and tables.

The three legacy table filenames, including `AA` in two names, are retained to avoid breaking the audited
aggregation pipeline. Their current manuscript destinations are defined above;
the submitted main manuscript uses Table 3 for response scenarios, Table 4 for
horizon and replay results, and Table 5 for benchmark outcomes.
