#!/usr/bin/env python3
"""Stage 5 - the analyses: benchmark metrics, mutation/position/severity, epitope and TCR tables.

Inputs: build/fp.db (stage 4) and build/merged/*.csv (stage 4), plus the committed overlap result.
Outputs: results/analysis/*.csv - the intermediate tables every table and figure is built from.

Three parts:
  A. the three published analyses (analysis/exp1-3.py) -> overall / per-mutation / severity tables
  B. epitope-level and TCR-level tables, computed here from the merged outer join
  C. dataset and coverage summaries, plus the overlap result copied in for traceability

Nothing is dropped: tables carry every row they can, and each one says how many rows lacked a
score or lacked both classes (which ROC-based metrics need) instead of quietly omitting them.

  python3 stages/stage5_analysis.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import (B_MERGED, FP_DB, MAIN_DATASET, MODELS, OVERLAP, R_ANALYSIS,  # noqa: E402
                           REPO, banner, die, need, write_provenance, set_seeds)

MAX_FPR = 0.1


def partial_auc(labels, scores):
    """Standardised partial AUC at FPR<=0.1; None when a class is missing."""
    y = np.asarray(labels, dtype=int)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(scores, dtype=float), max_fpr=MAX_FPR))


def run_published_analyses():
    banner("stage 5A", "published analyses (exp1, exp2, exp3)")
    need(FP_DB, "fp.db", "run stages/stage4_assemble.py first")
    env = dict(os.environ, TCRJ_DB=str(FP_DB), TCRJ_OUT=str(R_ANALYSIS), PYTHONPATH=str(REPO))
    for script in ("exp1.py", "exp2.py", "exp3.py"):
        log = R_ANALYSIS / "logs" / f"{script}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(REPO / "analysis" / script)],
                           capture_output=True, text=True, env=env, cwd=str(REPO))
        log.write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            die(f"{script} failed (log: {log.relative_to(REPO)})\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
        print(f"  {script}: ok")


def epitope_and_tcr_tables():
    banner("stage 5B", "epitope-level and TCR-level tables")
    mg = pd.read_csv(need(B_MERGED / f"{MAIN_DATASET}_all_models.csv", "merged table"))
    prob = {m: f"{m}_Prob" for m in MODELS}

    # --- per TCR x model: discrimination and agreement with the experimental signal
    rows = []
    for tcr, g in mg.groupby("TCR_Group", sort=True):
        for m in MODELS:
            sub = g[g[prob[m]].notna()]
            auc = partial_auc(sub.Label, sub[prob[m]]) if len(sub) else None
            rho = p = np.nan
            if len(sub) > 2 and sub[prob[m]].nunique() > 1:
                rho, p = spearmanr(sub[prob[m]], sub.log2foldchange)
            rows.append({"TCR_Group": tcr, "Model": m, "N_Rows": len(g), "N_Scored": len(sub),
                         "N_Positive": int(sub.Label.sum()), "N_Negative": int((sub.Label == 0).sum()),
                         "AUC01": auc, "Spearman_Rho_vs_log2FC": rho, "Spearman_P_vs_log2FC": p})
    tcr_tbl = pd.DataFrame(rows)
    out = R_ANALYSIS / "tcr_model_analysis.csv"
    tcr_tbl.to_csv(out, index=False)
    ineligible = int(tcr_tbl.AUC01.isna().sum())
    write_provenance(out, [
        "stage 5B: per TCR x model discrimination and agreement with log2 fold-change",
        f"source: build/merged/{MAIN_DATASET}_all_models.csv",
        f"rows: {len(tcr_tbl)} ({tcr_tbl.TCR_Group.nunique()} TCRs x {len(MODELS)} models) - all kept",
        f"AUC01 empty for {ineligible} cells where that TCR's scored rows are all one class",
        "N_Scored < N_Rows where de-duplication removed a row for that model.",
    ])
    print(f"  tcr_model_analysis.csv: {tcr_tbl.TCR_Group.nunique()} TCRs x {len(MODELS)} models "
          f"({ineligible} cells without both classes)")

    # --- per epitope group (R-5 vs the rest) x model
    rows = []
    for grp, g in mg.groupby("Epitope_Group", sort=True):
        for m in MODELS:
            sub = g[g[prob[m]].notna()]
            rows.append({"Epitope_Group": grp, "Model": m, "N_Rows": len(g), "N_Scored": len(sub),
                         "N_Positive": int(sub.Label.sum()), "Binder_Fraction": sub.Label.mean() if len(sub) else np.nan,
                         "AUC01": partial_auc(sub.Label, sub[prob[m]]) if len(sub) else None,
                         "Mean_Prediction": sub[prob[m]].mean() if len(sub) else np.nan})
    ep = pd.DataFrame(rows)
    out = R_ANALYSIS / "epitope_group_analysis.csv"
    ep.to_csv(out, index=False)
    write_provenance(out, [
        "stage 5B: per epitope group x model",
        f"source: build/merged/{MAIN_DATASET}_all_models.csv",
        f"groups: {', '.join(map(str, mg.Epitope_Group.unique()))}",
        f"rows: {len(ep)} - all kept; AUC01 empty where a group has a single experimental class",
    ])
    print(f"  epitope_group_analysis.csv: {ep.Epitope_Group.nunique()} groups x {len(MODELS)} models")

    # --- per peptide (mutation) x model, pooled over TCRs, on the same merged table
    rows = []
    for pep, g in mg.groupby("Peptide", sort=True):
        for m in MODELS:
            sub = g[g[prob[m]].notna()]
            rows.append({"Peptide": pep, "Model": m, "N_Rows": len(g), "N_Scored": len(sub),
                         "N_Positive": int(sub.Label.sum()), "N_Negative": int((sub.Label == 0).sum()),
                         "AUC01": partial_auc(sub.Label, sub[prob[m]]) if len(sub) else None})
    pep_tbl = pd.DataFrame(rows)
    out = R_ANALYSIS / "peptide_model_analysis.csv"
    pep_tbl.to_csv(out, index=False)
    write_provenance(out, [
        "stage 5B: per peptide x model",
        f"source: build/merged/{MAIN_DATASET}_all_models.csv",
        f"rows: {len(pep_tbl)} ({pep_tbl.Peptide.nunique()} peptides x {len(MODELS)} models) - all kept",
        f"AUC01 empty for {int(pep_tbl.AUC01.isna().sum())} cells without both experimental classes",
        "The reference peptide is included here; the manuscript's mutation analyses exclude it.",
    ])
    print(f"  peptide_model_analysis.csv: {pep_tbl.Peptide.nunique()} peptides x {len(MODELS)} models")


def coverage_and_overlap():
    banner("stage 5C", "coverage summary and training-overlap result")
    rows = []
    for f in sorted(B_MERGED.glob("*_all_models.csv")):
        ds = f.name.replace("_all_models.csv", "")
        mg = pd.read_csv(f)
        for m in MODELS:
            c = mg[f"{m}_Prob"].notna()
            rows.append({"dataset": ds, "model": m, "rows": len(mg), "scored": int(c.sum()),
                         "unscored": int((~c).sum()),
                         "positives": int(mg.loc[c, "Label"].sum()),
                         "negatives": int((mg.loc[c, "Label"] == 0).sum())})
    cov = pd.DataFrame(rows)
    out = R_ANALYSIS / "coverage_summary.csv"
    cov.to_csv(out, index=False)
    write_provenance(out, [
        "stage 5C: rows scored per dataset x model",
        "source: build/merged/*_all_models.csv",
        "unscored = rows removed by training-overlap de-duplication for that model.",
    ])
    print("  coverage_summary.csv: " + ", ".join(
        f"{ds}={int(g.rows.iloc[0])} rows" for ds, g in cov.groupby("dataset")))
    shutil.copy2(need(OVERLAP, "overlap summary"), R_ANALYSIS / "overlap_summary.csv")
    print("  overlap_summary.csv: copied from data/overlap (needs the models' training data to recompute)")


def main():
    set_seeds()   # see stages/common.py: the chain is deterministic; this keeps it that way
    R_ANALYSIS.mkdir(parents=True, exist_ok=True)
    run_published_analyses()
    epitope_and_tcr_tables()
    coverage_and_overlap()
    n = len(list(R_ANALYSIS.glob("*.csv")))
    print(f"\nstage 5 done: {n} tables in results/analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
