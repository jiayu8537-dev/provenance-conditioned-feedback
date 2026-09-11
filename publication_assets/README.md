# Publication assets

`figures/main/` contains the exact three PDF files used in the submitted main
manuscript: the matched-trajectory protocol, the six-round LMC trajectories,
and the benchmark comparison. `figures/supplementary/` contains the two figures
reported in the Supplementary Information. Their hashes are covered by the
package manifest.

`tables/` contains the data and initialization sources for main-manuscript
Tables 1 and 2. Remaining outcome tables and all extension tables are in the
package-level `tables/` directory; their current manuscript mapping is recorded
in `MANUSCRIPT_CROSSWALK.md`.

Run `make_jdsa_final_figures.py` to create vector and raster regeneration
outputs under `publication_assets/generated/`. The regeneration products are
scientifically equivalent to the locked submission assets. Minor
rasterization or font-metric differences can occur across Matplotlib versions
and operating-system font stacks, so the exact submitted PDFs are retained
separately in `figures/main/`.

Run `make_added_sensitivity_figure.py` to recreate the two-panel figure for the
lower-feedback and candidate-refresh analyses. It reads only the archived
endpoint tables in `tables/added_sensitivity/` and writes a vector PDF, caption,
and panel-level source-data copies under `publication_assets/generated/`. The
600-dpi PNG distributed with the package is a raster rendering of that PDF; no
experimental run is required to regenerate the plotted estimates.
