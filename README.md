# MutaTCR

A mutation-aware benchmark for TCR–epitope prediction models: all test peptides are
single-amino-acid variants of one reference antigen (HLA-A\*02:01–restricted SARS-CoV-2 epitope
YLQPRTFLL), and eight published predictors are scored against experimentally measured retention or
loss of TCR recognition — 171 substitutions across 21 TCRs, 3,612 interactions.

This repository holds the data, the pipeline and the generated figures and tables for the paper
*Benchmarking TCR–Epitope Recognition Models across Antigen Mutations*. **The manuscript itself is a
separate repository**; it includes the LaTeX fragments and numbers this one produces under
`results/`, so every figure, table and quoted value in the paper comes from the data here.

## Reproduce

```bash
pip install -r requirements.txt       # or: conda env create -f environment.yml
python3 checks/check_dependencies.py  # what you have vs what the results were made with
./run_all.sh                          # rebuilds everything, then checks it
```

About a minute. It runs the stages in `stages/`:

| stage | | |
|---|---|---|
| 1 | preprocess | `data/raw/` → the unified benchmark table |
| 2 | de-duplicate | removes rows overlapping each model's training data |
| 3 | inference | the eight predictors — **not run here**: needs their weights, so their scores ship in `data/scores/` ([docs/INFERENCE.md](docs/INFERENCE.md)) |
| 4 | assemble | unified + scores → per-model predictions, a full outer join, `fp.db` |
| 5 | analyse | benchmark metrics, mutation/position/severity, epitope and TCR tables |
| 6–7 | tables and figures | each one as a small source table → generated LaTeX → image |

Then `checks/check_repo.py` verifies the result: nothing reads outside the repo, re-running
reproduces `results/` byte for byte, no figure carries typed-in numbers, and the claims that depend
on an argmax still hold. `checks/smoke_test.py` is the 7-second version.

To run the models yourself: `./setup/fetch_models.sh` clones them at the exact commits used
([docs/INFERENCE.md](docs/INFERENCE.md)). No step that produces a published number uses randomness
([docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).

## Licence

MIT — see [LICENSE](LICENSE). It covers the code only. The experimental data in `data/raw/` comes
from previously published studies and keeps its original terms ([docs/DATA.md](docs/DATA.md)); the
model weights are third-party ([NOTICE](NOTICE)).
