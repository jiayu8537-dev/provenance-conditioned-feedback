# Data dictionary

## Reconstructed inputs (`data/derived/`)

- Main split: `row_id`, `user_id`, `item_id`, `timestamp`, `split`,
  `warm_item`.
- User/item indexes: consecutive internal index plus the official identifier.
- Assignment file: official item identifier; observed count/covariate fields;
  latent cluster; popularity stratum; model-aware block; selected provenance
  code. Provenance codes are 0 = human, 1 = AI-assisted, and 2 = AI-generated.

## Simulation outputs (`raw/`)

Common identifiers are `panel`, `assignment`, `response_seed`, `scenario`,
`intervention`, `round`, and `branch`. The branch is `frozen` or `closed`.
Extension files additionally identify `model`, `horizon`, `update_regime`,
or structural `core` where applicable.

Core outcome fields:

- `D`: signed human-versus-AI exposure contrast;
- `abs_D`: absolute exposure contrast;
- `exposure_tv`: total-variation distance between displayed provenance shares
  and their candidate-pool baseline;
- `ctr`: accepted recommendations divided by displayed recommendations;
- `candidate_anchored_utility`: accepted-item relevance normalized against the
  trajectory's candidate set;
- `coverage`: share of catalog items exposed by the stated round;
- `gini`: item-level exposure concentration;
- `accepted_interactions`: accepted events accumulated through the round;
- `event_gradient_evaluations`: event-level BPR gradient evaluations;
- `parameter_update_operations`: actual parameter-application operations.

## Summary outputs (`tables/`)

`estimate`, `ci_low`, and `ci_high` report the hierarchical crossed-bootstrap
mean and 95% interval unless the filename or column states otherwise. `AA` is
the paired closed-minus-frozen change in the signed exposure contrast;
`relative_ratio_percent` equals `100 * (exp(AA) - 1)`. `n` or
`simulation_cells` counts crossed design cells, not displayed events.

Empty cells in the no-control rows of the legacy-named Table 5 source file,
which now supports main-manuscript Table 4, are structurally inapplicable
paired-control contrasts, not missing computations.
