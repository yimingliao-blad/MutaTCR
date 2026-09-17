# Manuscript: what still needs an edit

Line numbers refer to `manuscript/main.tex` as of 2026-09-17. Items are grouped by what they need
from you. Nothing here was changed in the text — the manuscript still renders exactly as delivered.

---

## A. Blanks I can now fill (the information exists; say the word)

### A1. Table 1's "Version" column is empty — line 442 declares it, all eight rows leave it blank

The versions are now verified: `setup/model_sources.tsv` records the commit each model was actually
at when it produced `data/scores/`, read from the original clones, and every one is still fetchable.
Ready to paste, in the table's row order:

| Model | Version cell |
|---|---|
| ERGO | `5c6fc37` (2020-08-26) |
| ERGO-II | `85d320a` (2022-04-01) |
| NetTCR-2.0 | `4c9acd2` (2022-03-29) |
| NetTCR-2.2 | `244bb88` (2024-03-27) |
| TITAN | `1576fa3` (2025-07-12) |
| PanPep | `b44ffb1` (2023-12-03) |
| SCEPTR | `4700c79` (2025-03-13) |
| EPACT | `a907ac5` (2025-09-23) |

Two footnotes worth attaching: ERGO-II additionally needs TCR autoencoder weights whose original
source (`github.com/IdoSpringer/TCR_Autoencoder`) now 404s, and two one-line environment edits were
applied to run the models here (`setup/patches/`: `torch.load(..., map_location='cpu')` for ERGO-II,
GPU index for EPACT). Neither changes model behaviour, but they are part of the runs.

### A2. Table 1 caption, line 512 — an unfinished promise

> "The exact software or checkpoint version used for each model will be added after implementation
> verification."

Replace once A1 is filled; the versions are no longer pending.

---

## B. Needs a decision or a URL from you

### B3. Abstract, line 70 — availability statement
> "Code and processed benchmark data will be made publicly available upon publication."

Needs the repository URL once it exists. The repository has no remote yet.

### B4. There is no Data / Code Availability section
Most journals require one. It would need to say: the code and the processed benchmark are in this
repository (URL); the experimental measurements come from Malone et al. and are redistributed here
under the source's terms (`docs/DATA.md`); the model weights are third-party and fetched from their
own repositories (`docs/INFERENCE.md`).

### B5. Author Contributions, Competing Interests, Funding / Acknowledgements are all absent
The earlier arXiv version of this work carried `Author Contributions` and `Competing Interests`
sections; this manuscript has none of them, and no funding statement.

### B6. Author affiliations and the corresponding author are commented out — lines 48–54
`\author{Yiming Liao, Yiheng Li, Ning Jiang, Bo Li, Keke Chen}` renders, but the `\affil{...}`
lines, `\author*` and both `\email{}`s sit inside `\begin{comment}...\end{comment}`, so the compiled
PDF shows no affiliations and no corresponding author. That has to be restored in the journal's own
template. `\date{}` is also empty (line 55).

---

## C. Claims worth a second look before submission

### C7. Table 2 — two of the eight rows mean something different from the others
- **EPACT** was screened against `sample/VDJdb-GLCTLVAML.csv`, the *sample* of training data the
  model publishes, not its full training set, which is not in the public release. Its "0/172" row
  therefore means "no overlap with that sample".
- **SCEPTR** has no training file to screen at all: `src/data_processing/deduplicate.py` hard-codes
  empty sets for it, so its "0/172" is by construction rather than by measurement.

The Methods (§"Model-Specific Training–Benchmark Overlap Control") currently describe one uniform
procedure. Consider one sentence distinguishing these two.

### C8. What happened to the overlapping rows is described loosely
Line ~538: the NetTCR-2.2 overlap "was recorded explicitly and taken into account during
leakage-aware evaluation." Concretely, the two matching rows were **excluded from NetTCR-2.2's
scoring only** — it is evaluated on 3,610 of the 3,612 interactions, the other seven models on all
3,612. Stating that makes the per-model denominators explicit.

### C9. Model versions in the Methods text
§"Evaluated TCR–Epitope Prediction Models" says the "publicly released pretrained implementation"
was used for each model. With A1 filled, you can point at the exact commits.

---

## D. Checked and fine — no action

- Every `\ref` and `\cite` resolves; the build log has no undefined references or citations.
- Every number in the text, the captions, the figures and Table 2 is generated from the data
  (`results/figures/values.tex`), and `checks/check_repo.py` fails if any of them drifts.
- IMMREP22/IMMREP23 appear only as citations and as the source of the AUC$_{0.1}$ convention, so
  narrowing the repository to the FingerPrinting benchmark creates no inconsistency with the text.
- The bibliography no longer carries the Zotero `file =` fields that held an author's local paths.

---

Not covered here: the target journal's own author guidelines, the completeness of individual
bibliography entries, and the science and prose themselves — none of which I reviewed.
