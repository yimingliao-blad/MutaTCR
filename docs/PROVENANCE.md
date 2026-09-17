# Where the data came from

This repository was consolidated on 2026-09-17 from three working trees, which are upstream and
read-only; nothing flows back to them.

| what | source |
|---|---|
| raw inputs, unified tables, model scores | `tcrp-benchmark-v4` (the newest of seven predecessor trees; source edits to 2026-05-28) |
| de-duplication removed-row IDs | derived from that tree's `data/deduplicated/exact_match/` |
| training-overlap result | a separate overlap analysis run, 2026-08-11 |
| the three analyses (`analysis/exp1-3.py`) and `manuscript/main.tex` | the delivered manuscript archive `tcr_benchmarking_journal.zip` |
| the benchmark library (`src/`) | `tcrp-benchmark-v4/src` |

`data/PROVENANCE.tsv` records, for every canonical data file, its source path, sha256, size and
source mtime.

## What was verified when it was consolidated

- Stage 1 rebuilds all three unified tables **identically** from the raw inputs.
- The score tables reconstruct each model's prediction file exactly (probabilities byte-identical).
- The merged tables in the predecessor were an inner join across models and therefore lossy
  (3,610 of 3,612 fingerprinting rows); this repo keeps the full outer join instead.
- The rebuilt `fp.db` is numerically identical to the delivered one (see docs/GAPS.md).
- The analyses reproduce the delivered figure CSVs to 8.9e-16.
- The compiled manuscript is pixel-identical to the delivered PDF on all 17 pages.
