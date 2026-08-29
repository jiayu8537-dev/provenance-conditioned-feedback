# Data access and reconstruction

## Official source

The study uses PixelRec50K. Obtain the official release from the PixelRec
repository or the access location designated by its authors. The dataset
authors request that modified copies not be redistributed; consequently, this
archive contains data-processing code and validation hashes rather than the
official or derived interaction files.

Expected source filenames and SHA-256 checksums for the locked release are:

| File | Bytes | SHA-256 |
|---|---:|---|
| `interaction.csv` | 28,124,439 | `638b53ec100f760cb9bd540c361f6d6e3617c81b1c054ced63fffa41da909e4d` |
| `item_info.csv` | 24,973,166 | `a073c2c65900f215a8137929b27dc57cf6f4f8fa11453a5c74fa8ff3a730a04e` |

## Deterministic reconstruction

Run `scripts/prepare_pixelrec_inputs.py` as documented in `README.md`. The
script uses the original row order as `row_id`, repeatedly filters users and
items until the requested k-core stabilizes, sorts by
`user_id, timestamp, row_id`, and assigns each user's final two interactions
to validation and test. It also rebuilds the item metadata and attaches the
locked model-aware blocks and selected provenance labels by exact sorted item
identifier.

The expected core dimensions are:

| Core | Interactions | Users | Items |
|---|---:|---:|---:|
| 5u/3i | 956,817 | 49,993 | 59,781 |
| 10u/5i | 816,905 | 38,921 | 44,923 |
| 15u/5i | 567,403 | 20,132 | 37,576 |

`data/derived/` is deliberately empty in the distributed package. Generated
files stay under the control of the person who obtained the source dataset.
Use `JIIS_DATA_ROOT=/another/path` to point all analysis entry points to a
different generated-input directory.

## Semantic locks

Compressed byte streams can vary with gzip metadata. The preparation script
therefore validates uncompressed CSV bytes for compressed indexes and the main
split, and validates ordered item/block/label vectors for the reconstructed
assignment. The exact expected values are embedded in the script and recorded
in `logs/data_preparation_validation.json`.

