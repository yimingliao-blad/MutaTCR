# Model inference (stage 3): fetching the models and running them

Eight published predictors score every benchmark row: ERGO, ERGO-II, NetTCR-2.0, NetTCR-2.2, TITAN,
PanPep, SCEPTR, EPACT. Their code and weights are third-party and live in their own public
repositories, so they are **fetched, not vendored**. Their outputs are committed
(`data/scores/<dataset>_model_scores.csv`), which is why the rest of the manuscript rebuilds offline.

An empty cell in a score table means that model has no score for that row because training-overlap
de-duplication removed it (`data/dedup/removed_ids.csv` records which).

## Fetch

```bash
./setup/fetch_models.sh --list     # the sources and pinned commits
./setup/fetch_models.sh            # clone all of them into models/   (several GB)
./setup/fetch_models.sh NetTCR     # or one at a time
./setup/fetch_epact_data.sh        # EPACT's checkpoints (2.4 GB) + data (171 MB) from Zenodo
pip install sceptr                 # SCEPTR is a package, not a repository
python3 setup/validate_models.py   # what is ready and what is still missing
```

`setup/model_sources.tsv` holds one row per model: repository, pinned commit and notes.
`fetch_models.sh` checks out the pin and records what it actually got in `models/FETCHED.tsv`.
A failed clone stops the script rather than leaving a half-populated `models/`.

Two corrections found on 2026-09-17 while testing these scripts, both now in the sources file:

- EPACT's repository moved. The URL used during development, `GaoLabXDU/EPACT`, returns 404; the
  live one is `zhangyumeng1sjtu/EPACT`.
- EPACT's assets are Zenodo record **10996150** (as its README says), not 7779016, which now
  redirects to an unrelated record. The two archives are checksum-verified against the md5s Zenodo
  publishes before they are unpacked — `EPACT-data.zip` into `data/`, `EPACT-model-checkpoints.zip`
  into `checkpoints/`, which is where the runner looks.

Verified here: `NetTCR` and `EPACT` clone at their pinned commits, and `EPACT-data.zip` downloads,
passes its checksum and unpacks to `data/hla_library.json` + `data/binding/…`. The other five
repositories resolve (`git ls-remote`) but were not cloned in full.

## Pinned commits, honestly

The pins were resolved on 2026-09-17 — they are **not** the commits that produced the committed
scores, which were never recorded. A re-run from these pins may differ in the last digits. If you
re-run inference for publication, repin deliberately and note it.

## Environments and running

Each model needs its own Python environment; they have conflicting dependencies. Point the pipeline
at your models and conda, then run the stage:

```bash
export TCRP_MODELS_PATH=/path/to/models
export TCRP_CONDA_PATH=/path/to/miniconda3
python3 stages/stage3_inference.py --run
```

`src/model_runners/` holds the per-model runners. Stage 3 refuses to pretend: without the weights it
stops and says what is missing. `create_runner()` raises on an unknown model or a failed import — it
used to fall back to a runner that emits random numbers (see docs/REPRODUCIBILITY.md).

## The same requirement applies to two other steps

- **De-duplication** (`stages/stage2_dedup.py --recompute`) re-screens the benchmark against each
  model's training data. Without the models it materialises the tables from the committed
  `data/dedup/removed_ids.csv` instead.
- **The training-overlap analysis** (`analysis/overlap.py`) needs the same training files. Its
  result is committed at `data/overlap/overlap_summary.csv` and is what Table 2 is built from:

      TCRJ_MODELS=/path/to/models python3 analysis/overlap.py

  What each model was screened against — and the two caveats in that comparison — is in
  `docs/GAPS.md`.
