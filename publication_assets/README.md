# Publication assets

`figures/main/` contains the exact four PDF files used to compile the submitted
LaTeX manuscript. In particular, Fig. 2 is the final matched-trajectory diagram
and Fig. 3 is the single roundwise amplification panel used in the 25-page main
manuscript. `figures/supplementary/` contains the two figures reported in Online
Resource 1. Their hashes are covered by the package manifest.

`tables/` contains the data and initialization sources for main-manuscript
Tables 1 and 2. Remaining outcome tables and all extension tables are in the
package-level `tables/` directory; their current manuscript mapping is recorded
in `MANUSCRIPT_CROSSWALK.md`.

Run `make_jiis_final_figures.py` to create vector and raster regeneration
outputs under `publication_assets/generated/`. The regeneration products are
scientifically equivalent to the locked submission assets. Minor
rasterization or font-metric differences can occur across Matplotlib versions
and operating-system font stacks, so the exact submitted PDFs are retained
separately in `figures/main/`.
