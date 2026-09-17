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

## Scope
The benchmark is one HLA-A*02:01-restricted epitope (YLQPRTFLL) and 21 TCRs, single substitutions
only. The IMMREP23 and TetTCR-SeqHD datasets are included in full because the pipeline and the
supplementary compound figure cover them, but the manuscript's analyses use the FingerPrinting
mutation panel.

## Licensing
- The **code** is MIT (`LICENSE`), chosen by the authors on 2026-09-17. The predecessor working tree
  carried GPL-3.0 at its root with no per-file copyright headers; this release is a deliberate
  relicensing by the copyright holders. If any part of `src/` turns out to derive from GPL'd model
  code rather than being the authors' own, that decision has to be revisited.
- The **data** is not covered by that licence: `data/raw/` comes from published studies and keeps its
  original terms (docs/DATA.md). Confirm redistribution is permitted before publishing.
- **No remote yet.** Whether the manuscript and data travel together to a public host is the
  authors' decision.
