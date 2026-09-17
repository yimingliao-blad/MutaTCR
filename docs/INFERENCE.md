# Model inference (stage 3) — what it needs, and why its output ships with the repo

Eight published predictors score every benchmark row: ERGO, ERGO-II, NetTCR-2.0, NetTCR-2.2, TITAN,
PanPep, SCEPTR, EPACT. Running them needs, per model, the published weights/checkpoint and its own
Python environment — about 10 GB in total, plus GPU time. None of that belongs in a manuscript
repository, so **the scores are committed instead**:

    data/scores/<dataset>_model_scores.csv     row ID x one probability column per model

Every later stage is rebuilt from those, so the manuscript regenerates offline. An empty cell means
that model has no score for that row because training-overlap de-duplication removed it
(`data/dedup/removed_ids.csv` says which).

## To run inference yourself

1. Obtain the eight model repositories and their weights. `config/benchmark_config.yaml` lists each
   model's upstream repository, its environment name and its training-data file.
2. Create the environments (one per model; they have conflicting dependencies).
3. Point the pipeline at them:

       export TCRP_MODELS_PATH=/path/to/models
       export TCRP_CONDA_PATH=/path/to/miniconda3
       python3 stages/stage3_inference.py --run

`src/model_runners/` holds the per-model runners, `src/run_benchmark.py` the original end-to-end
driver. Stage 3 refuses to pretend: without the weights it stops and says what is missing rather
than producing empty output.

## The same applies to two other steps

- **De-duplication** (`stage2_dedup.py --recompute`) needs each model's *training* data to re-screen
  the benchmark. Without it, the stage materialises the tables from the committed removed-row IDs.
- **The training-overlap analysis** (`analysis/overlap.py`) needs the same training data. Its result
  is committed at `data/overlap/overlap_summary.csv` and is what Table 2 is built from. To recompute:

      TCRJ_MODELS=/path/to/models python3 analysis/overlap.py
