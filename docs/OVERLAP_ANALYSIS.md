# Test/Training Overlap Analysis

How much of each benchmark source dataset the 8 models had already seen during training.
Exact match only. Analysis date 2026-08-11.

- **Peptide overlap** — how many of the dataset's distinct peptides appear in the model's
  training data, and what share of rows those peptides account for.
- **CDR3b + peptide** — rows whose exact `(CDR3b, Peptide)` pair appears in training.
  This is the rule the benchmark ships (`src/data_processing/deduplicate.py`, `_exact_dedup`),
  so these are the rows actually removed before scoring.

---

## Results

### FingerPrinting — 3,612 rows, 172 distinct peptides

| Model | Peptide overlap | Row % | CDR3b+peptide rows | Row % |
|---|---|---|---|---|
| ERGO | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| ERGO2 | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| NetTCR | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| NetTCR22 | 1/172 (0.6%) | 21 (0.6%) | 2 | 0.1% |
| TITAN * | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| EPACT † | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| PanPep | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| SCEPTR * | 0/172 (0.0%) | 0 (0.0%) | 0 | 0.0% |

### IMMREP23 — 3,484 rows, 20 distinct peptides

| Model | Peptide overlap | Row % | CDR3b+peptide rows | Row % |
|---|---|---|---|---|
| ERGO | 14/20 (70.0%) | 2,616 (75.1%) | 0 | 0.0% |
| ERGO2 | 14/20 (70.0%) | 2,616 (75.1%) | 0 | 0.0% |
| PanPep | 10/20 (50.0%) | 2,016 (57.9%) | 34 | 1.0% |
| NetTCR22 | 7/20 (35.0%) | 1,542 (44.3%) | 146 | 4.2% |
| NetTCR | 3/20 (15.0%) | 928 (26.6%) | 109 | 3.1% |
| EPACT † | 1/20 (5.0%) | 144 (4.1%) | 0 | 0.0% |
| TITAN * | 0/20 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| SCEPTR * | 0/20 (0.0%) | 0 (0.0%) | 0 | 0.0% |

### TetTCR-SeqHD — 51,351 rows, 246 distinct peptides

| Model | Peptide overlap | Row % | CDR3b+peptide rows | Row % |
|---|---|---|---|---|
| ERGO2 | 23/246 (9.3%) | 9,276 (18.1%) | 350 | 0.7% |
| ERGO | 21/246 (8.5%) | 9,379 (18.3%) | 302 | 0.6% |
| NetTCR22 | 7/246 (2.8%) | 4,427 (8.6%) | 0 | 0.0% |
| PanPep | 7/246 (2.8%) | 4,337 (8.4%) | 186 | 0.4% |
| NetTCR | 5/246 (2.0%) | 1,465 (2.9%) | 0 | 0.0% |
| EPACT † | 1/246 (0.4%) | 394 (0.8%) | 12 | 0.0% |
| TITAN * | 0/246 (0.0%) | 0 (0.0%) | 0 | 0.0% |
| SCEPTR * | 0/246 (0.0%) | 0 (0.0%) | 0 | 0.0% |

**\* These zeros are not real.** TITAN's configured training file
(`TITAN/datasets/full_data+covid.csv`) holds numeric IDs, not sequences, so no training set
is built. SCEPTR's sets are hardcoded empty at `deduplicate.py:171-172`. Neither model is
deduplicated. Using TITAN's real sequence file (`allinfo_full_data+covid.csv`) its true
overlap is 21 rows / 0 pairs on FingerPrinting, 2,472 rows / 0 pairs on IMMREP23, and
6,239 rows / **138 pairs** on TetTCR-SeqHD.

**† EPACT is understated.** It uses a 228-row sample file covering a single epitope
(`EPACT/sample/VDJdb-GLCTLVAML.csv`), not its full training set.

**Reading the two peptide columns.** They diverge because seen peptides carry more rows
than average. ERGO on IMMREP23 matches 70% of peptides but 75% of rows; on TetTCR-SeqHD it
matches 8.5% of peptides but 18.3% of rows, since common epitopes like GILGFVFTL and
NLVPMVATV have far more TCRs tested against them.

**What this shows.** Peptide-level overlap is large while pair-level overlap is near zero.
Because deduplication keys on the pair, models are scored on peptides they trained on,
paired with TCRs they have not seen. That supports a claim of generalizing to new TCRs for
a known epitope, but not to unseen epitopes. FingerPrinting is clean at both levels.

All 24 model x dataset figures were recomputed from the training files and cross-checked
against the stored `*_deduplicated.csv`; all 24 matched.

---

## Folder contents

```
tmp_tcrp_output/
├── OVERLAP_ANALYSIS.md          this file
├── overlap.py                   analysis script
├── overlap_summary.csv          results, 24 rows
├── source/                      3 unified input files
│   ├── fingerprinting_unified.csv    3,612 rows
│   ├── immrep23_unified.csv          3,484 rows
│   └── tettcr_unified.csv           51,351 rows
└── output/                      raw predictions, 8 models x 3 datasets
    ├── EPACT/  ERGO/  ERGO2/  NetTCR/
    └── NetTCR22/  PanPep/  SCEPTR/  TITAN/
```

`output/` holds each model's raw predictions *before* the cross-model join, so row counts
differ slightly between models. Join on `ID` to compare against `source/`.

Columns in `overlap_summary.csv`: `dataset`, `model`, `total_rows`, `pep_rows`, `pep_pct`,
`pep_uniq`, `pair_rows`, `pair_pct`, `pair_uniq`, `verify`.

Rerun with:

```bash
cd ~/Projects/tmp_tcrp_output
~/Projects/tcrp-benchmark-fresh/tcrp-benchmark-v4/tools/miniconda3/bin/python \
    overlap.py overlap_summary.csv
```
