# Reproducibility: is anything here random?

Short answer: **no step that produces a published number draws a random value.** The seed is set
anyway, and there are checks that would catch it if that ever changed.

## Where randomness could live, and what is actually there

| stage | randomness | notes |
|---|---|---|
| 1 preprocess | none | reading workbooks, joining, labelling |
| 2 de-duplicate | none in the shipped path | it applies a committed list of removed row IDs. `--recompute` re-screens against training data, which is exact set matching, also not random |
| 3 inference | **yes, inside the models** | not run here; see below |
| 4 assemble | none | joins and a SQLite write |
| 5 analyses | none | partial AUC, means, Spearman correlations — deterministic arithmetic |
| 6 tables | none | formatting |
| 7 figures | none | formatting; pgfplots draws from the source tables |
| 8 supplementary | seeded jitter | `np.random.seed(42)` before the scatter jitter in three plotting modules; affects dot placement only, never a number |

`stages/common.py` defines `SEED = 42` and `set_seeds()`, which seeds Python's `random`, NumPy, and
— when present — PyTorch, including the cuDNN determinism flags. Every stage calls it first. Nothing
in stages 1–7 consumes it today; it is there so that anything added later (a bootstrap, a sampled
control, a permutation test) is repeatable by default instead of by luck.

## Stage 3 is the honest exception

The eight predictors are third-party and some of them sample internally:

- `panpep_runner.py` and `sceptr_runner.py` draw negative examples and sample rows —
  both seeded (`np.random.seed(42)`, `sample(random_state=42)`).
- `titan_runner.py` sets `torch.manual_seed(123456)`.
- `epact_runner.py` passes `seed: 42` to the model.

Those seeds make a re-run repeatable **on the same hardware, with the same library and CUDA
versions**. A fixed seed does not make GPU inference bitwise reproducible on its own: kernel
selection, batch composition and library versions all move the last digits. That is why the model
scores are committed as data (`data/scores/`) rather than treated as something to regenerate — the
published numbers rest on those scores, and everything downstream of them is exact.

There was one real hazard here, now removed: `create_runner()` used to fall back to
`DummyModelRunner` — which fills `Prediction_Prob` with `np.random.random()` — when a model key was
unknown or a runner failed to import. That would have produced random numbers indistinguishable from
predictions. It now raises instead, the dummy runner is seeded, labels its output
(`IS_RANDOM_DUMMY_OUTPUT`), and `checks/check_repo.py` (check F2) fails if the fallback returns.

## The evidence, not just the argument

Run `python3 checks/check_repo.py`:

- **check E** runs stage 5 twice in separate processes and compares all 22 analysis tables — they are
  byte-identical.
- **check F** greps the stages and the three analysis scripts for any random draw
  (`np.random.*`, `random.*`, `.sample(`, `.shuffle(`, `default_rng(`) and fails if one appears.
  Verified to fire when a draw is planted.
- **check B** re-runs stages 6 and 7 and compares the rendered LaTeX with the committed copies.

End to end: a fresh `git clone` plus `./run_all.sh` reproduces the analysis tables, the figure and
table sources and the LaTeX byte for byte; compiled against them, the manuscript rendered
pixel-identical on all 17 pages to the delivered original.

## One thing that is deliberately not byte-reproducible

A PDF embeds its creation timestamp, so the manuscript PDF is never byte-identical between two
builds even when nothing changed. Compare rendered pages instead — rasterise both and diff, which is
how the "pixel-identical on all 17 pages" result above was measured. The same applies to the figure
PDFs in `results/figures/`. Everything that is *data* — the analysis tables, the source tables, the
rendered LaTeX — is byte-reproducible, and that is what the checks compare.

## Library versions

The committed results were produced with Python 3.9.24, pandas 2.3.3, numpy 2.0.2, scipy 1.13.1,
scikit-learn 1.6.1 (full list in `requirements.txt`). The same pipeline under pandas 3.0.1,
scipy 1.17.1 and scikit-learn 1.8.0 gave the same numbers to **8.9e-16**, and a manuscript whose
pages render identically, so the results do not rest on one version of anything. `checks/check_dependencies.py`
prints what you have beside what was used.

The one place a version *does* show through is float formatting, not arithmetic: the original model
runner wrote `log2foldchange` with fewer digits than the rebuilt pipeline does, so 1,375 of 3,612
cells differ as text while agreeing within 1e-12 (`docs/GAPS.md`).
