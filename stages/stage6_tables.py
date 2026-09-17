#!/usr/bin/env python3
"""Stage 6 - tables, in the same two reviewable steps as the figures.

  --prepare   results/analysis/*.csv -> results/tables/<id>_source.csv (+ .provenance.txt)
  --render    results/tables/<id>_source.csv -> results/tables/<id>.tex (a complete tabular)

Table 2 is the one in the manuscript; the rest are the supplementary tables behind the figures,
emitted in the same shape so they can be dropped into the paper or an appendix as they are.
A table is rendered from its own source CSV only - never from the raw results.

  python3 stages/stage6_tables.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import (LABEL, MODELS, PROSE_NAME, R_ANALYSIS, R_TABLES, banner,  # noqa: E402
                           canon, need, write_provenance, set_seeds)

NPEP = 172  # peptide species in the panel: the reference plus 171 substitutions


def read(name):
    return pd.read_csv(need(R_ANALYSIS / name, f"analysis table {name}",
                            "run stages/stage5_analysis.py first"))


def fmt(v, spec):
    if pd.isna(v):
        return "--"
    if spec == "int":
        return f"{int(round(float(v))):d}"
    if spec.startswith("f"):
        return f"{float(v):.{int(spec[1:])}f}"
    return str(v)


def prepare():
    banner("stage 6 --prepare", "analysis tables -> per-table source tables")
    R_TABLES.mkdir(parents=True, exist_ok=True)
    specs = {}

    def save(tid, df, lines, headers, formats, align, caption):
        out = R_TABLES / f"{tid}_source.csv"
        df.to_csv(out, index=False)
        write_provenance(out, lines)
        specs[tid] = {"headers": headers, "formats": formats, "align": align, "caption": caption}
        print(f"  {out.name}: {len(df)} rows x {len(df.columns)} cols")

    # --- Table 2 (manuscript): training/benchmark overlap
    ovl = read("overlap_summary.csv")
    ovl = ovl[ovl.dataset == "fingerprinting"].copy()
    ovl["model"] = ovl.model.map(canon)
    order = ["ERGO", "ERGO2", "NetTCR", "NetTCR22", "TITAN", "EPACT", "PanPep", "SCEPTR"]
    ovl = ovl.set_index("model").reindex(order).reset_index()
    t2 = pd.DataFrame({
        "model": [PROSE_NAME[m] for m in ovl.model],
        "peptide_overlap": [f"{int(u)}/{NPEP} ({u / NPEP * 100:.1f}\\%)" for u in ovl.pep_uniq],
        "rows_peptide_overlap": [f"{int(r)} ({r / t * 100:.1f}\\%)"
                                 for r, t in zip(ovl.pep_rows, ovl.total_rows)],
        "rows_exact_pair": [f"{int(r)} ({r / t * 100:.1f}\\%)"
                            for r, t in zip(ovl.pair_rows, ovl.total_rows)],
    })
    save("table2_overlap", t2,
         ["Table 2: training/benchmark overlap per model (the manuscript's Table 2)",
          "source: results/analysis/overlap_summary.csv (overlap analysis; needs the models'",
          "  training data to recompute, so the result ships with the repo)",
          f"denominator for peptide overlap: {NPEP} peptide species (reference + 171 substitutions)",
          "row percentages use each model's own total benchmark rows"],
         ["Model", "Peptide overlap", "Rows with peptide overlap", "Exact CDR3$\\beta$--peptide rows"],
         ["str", "str", "str", "str"], "lccc",
         "Training--benchmark overlap between the Fingerprinting dataset and the training data "
         "used by the evaluated prediction models.")

    # --- S1: overall benchmark metrics per model
    ov = read("experiment1_overall_metrics.csv")
    ov["Model"] = ov.Model.map(canon)
    s1 = pd.DataFrame({"model": [LABEL[m] for m in ov.Model], "n_pairs": ov.N_Pairs,
                       "n_positive": ov.N_Positive_Pairs, "n_negative": ov.N_Negative_Pairs,
                       "n_eligible_mutations": ov.N_Eligible_Mutations,
                       "macro_auc01": ov["Macro_AUC0.1"], "macro_auprc": ov.Macro_AUPRC,
                       "pooled_auc01": ov["Pooled_AUC0.1"], "pooled_auprc": ov.Pooled_AUPRC}
                      ).sort_values("macro_auc01", ascending=False)
    save("tableS1_overall_metrics", s1,
         ["Supplementary: overall benchmark metrics per model",
          "source: results/analysis/experiment1_overall_metrics.csv (exp1.py)",
          "sorted by Macro AUC0.1, descending; every model keeps all of its rows"],
         ["Model", "Pairs", "Pos.", "Neg.", "Mutations", "Macro AUC$_{0.1}$", "Macro AUPRC",
          "Pooled AUC$_{0.1}$", "Pooled AUPRC"],
         ["str", "int", "int", "int", "int", "f3", "f3", "f3", "f3"], "lrrrrcccc",
         "Overall benchmark metrics for the eight evaluated models.")

    # --- S2: experimental tolerance per peptide position
    pos = read("experiment2_position_summary.csv")
    s2 = pos.rename(columns={"Mutation_Position": "position", "N_Mutations": "n_mutations",
                             "Mean_Binder_Fraction": "mean_binder_fraction",
                             "SD_Binder_Fraction": "sd_binder_fraction",
                             "Mean_log2FC": "mean_log2fc", "Mean_Delta_log2FC": "mean_delta_log2fc"}
                    )[["position", "n_mutations", "mean_binder_fraction", "sd_binder_fraction",
                       "mean_log2fc", "mean_delta_log2fc"]]
    save("tableS2_position_summary", s2,
         ["Supplementary: experimental tolerance per peptide position",
          "source: results/analysis/experiment2_position_summary.csv (exp2.py)",
          "all nine positions, P1 included (P1 supports no ROC metric, but its experimental",
          "  tolerance is the informative observation)"],
         ["Position", "Mutations", "Mean binder fraction", "SD", "Mean log$_2$FC", "Mean $\\Delta$log$_2$FC"],
         ["int", "int", "f3", "f3", "f3", "f3"], "lrcccc",
         "Experimentally observed tolerance at each peptide position.")

    # --- S3: R5 vs non-R5, experiment and prediction side by side
    r5b = read("experiment2_r5_summary.csv")
    r5a = read("r5_vs_nonr5_auc_summary.csv")
    r5a["Model"] = r5a.Model.map(canon)
    piv = r5a.pivot(index="Model", columns="R5_Status", values="Mean_AUC01").reindex(MODELS)
    s3 = pd.DataFrame({"model": [LABEL[m] for m in piv.index],
                       "mean_auc01_r5": piv["R5"].values,
                       "mean_auc01_non_r5": piv["non-R5"].values,
                       "difference": (piv["R5"] - piv["non-R5"]).values})
    save("tableS3_r5_vs_nonr5", s3,
         ["Supplementary: mean AUC0.1 for R5 substitutions vs all other positions",
          "source: results/analysis/r5_vs_nonr5_auc_summary.csv (exp3.py)",
          f"experimental binder fraction for reference: R5 "
          f"{float(r5b.loc[r5b.R5_Group == 'R5', 'Mean_Binder_Fraction'].iloc[0]):.3f} vs non-R5 "
          f"{float(r5b.loc[r5b.R5_Group == 'non-R5', 'Mean_Binder_Fraction'].iloc[0]):.3f}",
          "negative difference = the model does worse on R5 substitutions"],
         ["Model", "R5", "non-R5", "Difference"], ["str", "f3", "f3", "f3"], "lccc",
         "Prediction performance for R5 substitutions compared with mutations elsewhere.")

    # --- S4: per-TCR discrimination and agreement with the experiment
    tcr = read("tcr_model_analysis.csv")
    wide = tcr.pivot(index="TCR_Group", columns="Model", values="AUC01").reindex(columns=MODELS)
    s4 = wide.reset_index().rename(columns={"TCR_Group": "tcr", **{m: LABEL[m] for m in MODELS}})
    save("tableS4_tcr_auc", s4,
         ["Supplementary: AUC0.1 per TCR x model",
          "source: results/analysis/tcr_model_analysis.csv (stage 5B)",
          "empty cell = that TCR's scored rows are all one experimental class, so AUC0.1 is undefined",
          "the per-TCR Spearman correlations with log2FC are in the source analysis table"],
         ["TCR"] + [LABEL[m] for m in MODELS], ["str"] + ["f3"] * len(MODELS), "l" + "c" * len(MODELS),
         "Per-TCR discrimination (AUC$_{0.1}$) for each model.")

    # --- S5: epitope groups
    ep = read("epitope_group_analysis.csv")
    ep["Model"] = ep.Model.map(canon)
    s5 = ep.pivot(index="Epitope_Group", columns="Model", values="AUC01").reindex(columns=MODELS)
    s5 = s5.reset_index().rename(columns={"Epitope_Group": "epitope_group",
                                          **{m: LABEL[m] for m in MODELS}})
    save("tableS5_epitope_group", s5,
         ["Supplementary: AUC0.1 per epitope group x model",
          "source: results/analysis/epitope_group_analysis.csv (stage 5B)",
          "groups are the dataset's own Epitope_Group labels (R-5 vs the rest)"],
         ["Epitope group"] + [LABEL[m] for m in MODELS], ["str"] + ["f3"] * len(MODELS),
         "l" + "c" * len(MODELS), "Discrimination within each epitope group.")

    # --- S6: coverage (what each model scored, per dataset)
    cov = read("coverage_summary.csv")
    cov["model"] = cov.model.map(canon)
    s6 = cov.assign(model=[LABEL[m] for m in cov.model])[
        ["dataset", "model", "rows", "scored", "unscored", "positives", "negatives"]]
    save("tableS6_coverage", s6,
         ["Supplementary: rows scored per dataset x model",
          "source: results/analysis/coverage_summary.csv (stage 5C)",
          "unscored = rows removed for that model by training-overlap de-duplication",
          "every dataset row is accounted for: scored + unscored = rows"],
         ["Dataset", "Model", "Rows", "Scored", "Unscored", "Positives", "Negatives"],
         ["str", "str", "int", "int", "int", "int", "int"], "llrrrrr",
         "Benchmark coverage per dataset and model.")

    pd.Series(specs).to_json(R_TABLES / "_table_specs.json", indent=2)
    return specs


def render():
    banner("stage 6 --render", "source tables -> LaTeX tabulars")
    import json
    specs = json.loads(need(R_TABLES / "_table_specs.json", "table specs",
                            "run --prepare first").read_text())
    for tid, spec in specs.items():
        df = pd.read_csv(need(R_TABLES / f"{tid}_source.csv", f"source table for {tid}"))
        headers, formats, align = spec["headers"], spec["formats"], spec["align"]
        if len(headers) != len(df.columns) or len(formats) != len(df.columns):
            raise SystemExit(f"{tid}: {len(df.columns)} columns but {len(headers)} headers / "
                             f"{len(formats)} formats")
        lines = [f"% GENERATED by stages/stage6_tables.py from {tid}_source.csv - do not edit.",
                 f"% {spec['caption']}",
                 f"\\begin{{tabular}}{{{align}}}", "\\hline",
                 " & ".join(headers) + " \\\\", "\\hline"]
        for _, row in df.iterrows():
            lines.append(" & ".join(fmt(v, f) for v, f in zip(row.values, formats)) + " \\\\")
        lines += ["\\hline", "\\end{tabular}"]
        # trailing % : keeps \input from adding a stray space inside the table environment
        (R_TABLES / f"{tid}.tex").write_text("\n".join(lines) + "%\n")
        print(f"  {tid}.tex  <- {tid}_source.csv ({len(df)} rows)")


def main():
    set_seeds()   # see stages/common.py: the chain is deterministic; this keeps it that way
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    if not a.prepare and not a.render:
        a.prepare = a.render = True
    if a.prepare:
        prepare()
    if a.render:
        render()
    print("\nstage 6 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
