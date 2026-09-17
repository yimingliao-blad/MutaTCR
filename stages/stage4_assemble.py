#!/usr/bin/env python3
"""Stage 4 - assemble the analysis inputs from unified tables + committed model scores.

Produces, under build/:
  predictions/<MODEL>/<dataset>_predictions.csv   what that model scored (its de-duplicated rows)
  merged/<dataset>_all_models.csv                 FULL OUTER JOIN: every unified row, one column per
                                                  model, empty where that model has no score, plus
                                                  n_models_scored and all_models_scored flags
  fp.db                                           SQLite, one table per model (the analysis input)

Data policy: nothing is dropped here. The merged table keeps every row of the unified dataset; rows
that not all models scored are flagged, not removed. Narrowing to complete rows happens only in the
figure/table stage, where the count is recorded.

  python3 stages/stage4_assemble.py                     # all datasets
  python3 stages/stage4_assemble.py --dataset fingerprinting
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import (B_MERGED, B_PRED, DATASETS, DEDUP_IDS, FP_DB, MAIN_DATASET,  # noqa: E402
                           MODELS, SCORES, UNIFIED, banner, die, need, write_provenance, set_seeds)

# Column order of the delivered prediction files / fp.db tables.
PRED_COLS = ["ID", "Peptide", "HLA", "CDR3a", "CDR3b", "Va", "Ja", "Vb", "Jb", "Label",
             "log2foldchange", "TCR_Group", "Epitope_Group", "Prediction_Prob"]


def main():
    set_seeds()   # see stages/common.py: the chain is deterministic; this keeps it that way
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, action="append",
                    help="limit to one dataset (repeatable); default: all")
    args = ap.parse_args()
    datasets = args.dataset or DATASETS

    banner("stage 4", f"unified + scores -> predictions, merged (outer join), fp.db  [{', '.join(datasets)}]")
    removed = pd.read_csv(need(DEDUP_IDS, "removed-id list"), dtype=str)
    fp_tables = {}

    for ds in datasets:
        u = pd.read_csv(need(UNIFIED / f"{ds}_unified.csv", f"unified {ds}"),
                        dtype=str, keep_default_na=False)
        sc = pd.read_csv(need(SCORES / f"{ds}_model_scores.csv", f"scores {ds}"), dtype=str,
                         keep_default_na=False, na_values=[""])
        if len(sc) != len(u) or set(sc.ID) != set(u.ID):
            die(f"{ds}: score table and unified table cover different rows "
                f"({len(sc)} vs {len(u)})")

        # ---- per-model prediction tables
        for m in MODELS:
            col = f"{m}_Prob"
            drop = set(removed[(removed.dataset == ds) & (removed.model == m)].removed_id)
            scored = sc[sc[col].notna()]
            if set(scored.ID) & drop:
                die(f"{ds}/{m}: {len(set(scored.ID) & drop)} rows are both scored and de-duplicated")
            df = u.merge(scored[["ID", col]], on="ID", how="inner").rename(columns={col: "Prediction_Prob"})
            have = [c for c in PRED_COLS if c in df.columns]
            df = df[have]
            out_dir = B_PRED / m
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{ds}_predictions.csv"
            df.to_csv(out, index=False)
            if ds == MAIN_DATASET:
                fp_tables[m] = df
            write_provenance(out, [
                f"stage 4: {ds} predictions for {m}",
                f"sources: data/unified/{ds}_unified.csv + data/scores/{ds}_model_scores.csv",
                f"rows: {len(df)} of {len(u)} unified rows "
                f"({len(u) - len(df)} without a score for this model: de-duplicated or not run)",
            ])

        # ---- merged: full outer join, nothing dropped
        mg = u.merge(sc, on="ID", how="left")
        prob_cols = [f"{m}_Prob" for m in MODELS]
        mg["n_models_scored"] = mg[prob_cols].notna().sum(axis=1)
        mg["all_models_scored"] = (mg.n_models_scored == len(MODELS)).map({True: "1", False: "0"})
        B_MERGED.mkdir(parents=True, exist_ok=True)
        out = B_MERGED / f"{ds}_all_models.csv"
        mg.to_csv(out, index=False)
        incomplete = int((mg.n_models_scored < len(MODELS)).sum())
        write_provenance(out, [
            f"stage 4: {ds} merged across all {len(MODELS)} models (FULL OUTER JOIN)",
            f"rows: {len(mg)} - every row of the unified dataset is kept",
            f"rows scored by all {len(MODELS)} models: {len(mg) - incomplete}",
            f"rows missing at least one model's score: {incomplete} (flagged, NOT dropped)",
            "Downstream figures that need aligned models drop the incomplete rows themselves and",
            "record the count in their own provenance file.",
        ])
        print(f"  {ds}: merged {len(mg)} rows; {incomplete} row(s) missing at least one model's score "
              f"(kept and flagged)")

    # ---- fp.db for the manuscript's analyses
    if MAIN_DATASET in datasets:
        FP_DB.parent.mkdir(parents=True, exist_ok=True)
        if FP_DB.exists():
            FP_DB.unlink()
        conn = sqlite3.connect(FP_DB)
        try:
            # table order as in the delivered database
            for m in ["EPACT", "ERGO2", "NetTCR22", "SCEPTR", "ERGO", "NetTCR", "PanPep", "TITAN"]:
                df = fp_tables[m]
                cols = ", ".join(f'"{c}" TEXT' for c in df.columns)
                conn.execute(f'CREATE TABLE "{m}" ({cols})')
                conn.executemany(f'INSERT INTO "{m}" VALUES ({", ".join("?" * len(df.columns))})',
                                 df.astype(str).values.tolist())
            conn.commit()
        finally:
            conn.close()
        print(f"  fp.db: 8 tables, "
              + ", ".join(f"{m}={len(fp_tables[m])}" for m in ["ERGO", "NetTCR22"]) + " rows")
    print("\nstage 4 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
