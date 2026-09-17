# Known gaps

Things a reader should know that the code cannot tell them.

## Not reproducible inside this repo
- **Model inference** needs ~10 GB of published weights and eight environments; the scores are
  committed instead (docs/INFERENCE.md). Same for the de-duplication re-screen and the
  training-overlap analysis, whose results are committed.

## Drawings, not generated figures
- `manuscript/figures/framework.png` — the framework overview in Figure 1. Drawn by hand; no source
  file was found in any predecessor tree.
- `assets/Refined Flow.svg` and `assets/new_process.png` — panel (a) of the supplementary compound
  figure and the pipeline overview. Also drawings.

Everything else in the manuscript is generated from data by stages 6 and 7.

## Unused inputs left behind
Two CSVs in the earlier working trees are produced by no script and read by none
(`fig3c_heatmap.csv`, `fig3c_position_blocks.csv`); they look like leftovers from a cut figure and
are not carried here. Three files in the predecessor's `raw/` directory were read by no code and are
not carried either: `TCR_pMHC_annotated.csv` (26 MB), `tettcr_seqhd_unified_new.csv` (10.7 MB) and
`solutions.csv` (a byte-identical duplicate of `immref23.csv`).

## Formatting, not numbers
The delivered `fp.db` and the one stage 4 rebuilds differ in the text of `log2foldchange` for 1,375
of 3,612 rows: the original runner re-serialised the float with fewer digits. Numerically the two
agree within 1e-12, and every published number reproduces.

## What the training-overlap screen compared

Table 2 reports overlap between the benchmark and each model's training data. Two of the eight
models are not used as released pretrained predictors, which is why their rows are produced
differently (owner, 2026-09-17):

- **SCEPTR** is not a pretrained classifier here. `src/model_runners/sceptr_runner.py` embeds TCRs
  with the pretrained encoder and then **fits a classifier** (KNN or random forest) at run time on
  VDJdb data — 6 peptides x 300 samples, with negatives generated 1:5. There is no released training
  set to screen against, so `deduplicate.py` carries empty sets for it deliberately.
- **EPACT** was **retrained** for this work, so the file the screen uses is the data that model was
  trained on rather than a third-party corpus. The retrained weights are not distributed; the
  training code, configuration and seed are (docs/INFERENCE.md), so the model is reproduced rather
  than downloaded.

The other six models were screened against the training files their repositories ship, all of which
are present in the clones that produced `data/scores/`.

## Scope
The benchmark is one HLA-A*02:01-restricted epitope (YLQPRTFLL) and 21 TCRs, single substitutions
only.

This repository carries **only the FingerPrinting mutation panel**. The predecessor pipeline also
ran IMMREP23 and TetTCR-SeqHD; their raw data, unified tables and scores were removed on 2026-09-17
because the manuscript does not use them. Removed with them:

- `src/visualization/compound_figure.py` — the earlier preprint's multi-panel figure; its panel (b)
  pooled the two other datasets.
- `src/visualization/auc_boxplot.py` — per-peptide boxplots for those datasets.
- `assets/` — the two drawings only the compound figure used.

`src/` is the upstream benchmark library and still contains three-dataset support (`preprocess_raw.py`,
the model runners); the pipeline in `stages/` is scoped to one dataset, and `checks/check_repo.py`
(A5, A6) fails if that quietly changes.

## Licensing
- The **code** is MIT (`LICENSE`), chosen by the authors on 2026-09-17. The predecessor working tree
  carried GPL-3.0 at its root with no per-file copyright headers; this release is a deliberate
  relicensing by the copyright holders. If any part of `src/` turns out to derive from GPL'd model
  code rather than being the authors' own, that decision has to be revisited.
- The **data** is not covered by that licence: `data/raw/` comes from published studies and keeps its
  original terms (docs/DATA.md). Confirm redistribution is permitted before publishing.
- **No remote yet.** Whether the manuscript and data travel together to a public host is the
  authors' decision.
