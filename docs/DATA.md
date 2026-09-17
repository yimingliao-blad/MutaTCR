# Data sources and terms

The MIT licence in `LICENSE` covers the code. It does **not** cover the experimental data:
`data/raw/` holds files from previously published studies, redistributed here so the pipeline runs
offline. Each remains under its original publisher's terms, and this repository claims no rights
over them. **Confirm redistribution is permitted before publishing the repository**, and cite the
original studies rather than this repo for the measurements.

## Raw inputs

| file | source | what it is |
|---|---|---|
| `data/raw/FingerPrinting/Fingerprinting_TCR_Data_Combined_final.xlsx`, `finger_print.csv`, `finger_print.xlsx` | Malone MJ, Huang C, Zhang Y, Qi Y, Williams L, Su LF, Lou J, Jiang N. *Resistance potential of the HLA-A2-restricted immunodominant SARS-CoV-2-specific CD8+ T cell receptor repertoire to antigenic drift.* Nature Communications. | The TCR fingerprinting measurements: the YLQPRTFLL panel (reference + 171 single-substitution variants) against 21 TCRs. **This is the manuscript's benchmark.** |
| `data/raw/TetTCR-SeqHD/41590_2021_1073_MOESM12_ESM.xlsx` | Supplementary material of the TetTCR-SeqHD study (Nature Immunology, 2021; file name is the publisher's supplementary identifier). | High-throughput TCR–antigen binding measurements. |
| `data/raw/TetTCR-SeqHD/TCR_antigen_binding_sheet.csv`, `Kevin's Publication TCRs Updated.xlsx`, `BINDING_CLASSIFICATION_RULES.md` | Prepared by the authors from the above. | The binding sheet stage 1 reads, the TCR annotation table, and the written classification rules. |
| `data/raw/IMMREP23/immref23.csv` | IMMREP23 community benchmark — Nielsen M *et al.*, *Lessons learned from the IMMREP23 TCR-epitope prediction challenge*, ImmunoInformatics. | The IMMREP23 evaluation set. |

## Derived data in this repository

Produced by the pipeline from the inputs above: `data/unified/`, `data/scores/`,
`data/dedup/removed_ids.csv`, `data/overlap/overlap_summary.csv` and everything under `results/`.
These are the authors' own output, but they are derivatives of the source measurements, so anyone
reusing them should observe the source terms and cite the original studies.

`data/PROVENANCE.tsv` records every canonical file with its sha256, size and origin.

## Model weights

Not in this repository. The eight predictors are third-party, each under its own licence, and each
is obtained from its own repository (listed in `config/benchmark_config.yaml`). See
`docs/INFERENCE.md`.
