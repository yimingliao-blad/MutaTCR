#!/usr/bin/env python3
"""Stage 7 - figures, in two reviewable steps.

  --prepare   results/analysis/*.csv  ->  results/figures/<fig>_source.csv   (+ .provenance.txt)
              One small source table per figure. This is where any narrowing happens, and each
              provenance file records exactly what was dropped and why.
  --render    results/figures/<fig>_source.csv + templates/<fig>.tex.in
              ->  results/figures/<fig>.tex        (the LaTeX the manuscript includes)
              ->  results/figures/<fig>.pdf/.png   (a standalone image for review)
              plus results/figures/values.tex, the numbers the prose and captions quote.

To change a figure you edit its template, or its source table, and re-render - you never have to go
back to the raw results. Default runs both steps.

  python3 stages/stage7_figures.py
  python3 stages/stage7_figures.py --prepare
  python3 stages/stage7_figures.py --render --no-images
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import (LABEL, MODELS, PROSE_NAME, R_ANALYSIS, R_FIGURES, REPO, banner,  # noqa: E402
                           canon, die, need, write_provenance)

TEMPLATES = REPO / "templates"
MANUSCRIPT = REPO / "manuscript"
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
SPELLED = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
           8: "eight", 9: "nine", 10: "ten"}
FIGURES = ["fig2a", "fig2b", "fig2c", "fig3", "fig4", "fig5", "fig6", "fig7a", "fig7b"]
macros: list[tuple[str, str]] = []


def macro(name, value, fmt="{:.3f}"):
    macros.append((name, value if isinstance(value, str) else fmt.format(value)))


def tex_name(s):
    out = s.replace(".", "").replace("-", "")
    for d, w in WORD.items():
        out = out.replace(str(d), w)
    return out


def read(name):
    return pd.read_csv(need(R_ANALYSIS / name, f"analysis table {name}",
                            "run stages/stage5_analysis.py first"))


def prepare():
    banner("stage 7 --prepare", "analysis tables -> per-figure source tables")
    R_FIGURES.mkdir(parents=True, exist_ok=True)

    def save(fid, df, lines):
        out = R_FIGURES / f"{fid}_source.csv"
        df.to_csv(out, index=False)
        write_provenance(out, lines)
        print(f"  {out.name}: {len(df)} rows x {len(df.columns)} cols")

    # --- Fig 2a: experimental fingerprint heatmap (one point per TCR x mutation)
    heat = read("fig4d_experimental_heatmap.csv")
    save("fig2a", heat, ["Fig 2a: experimental log2 fold-change per TCR x mutation",
                         "source: results/analysis/fig4d_experimental_heatmap.csv (exp2.py)",
                         f"rows: {len(heat)} - every measured TCR x mutation pair; nothing dropped"])

    # --- Fig 2b: position-wise mean binder fraction
    pos = read("experiment2_position_summary.csv")
    f2b = pos[["Mutation_Position", "Mean_Binder_Fraction", "N_Mutations"]].rename(
        columns={"Mutation_Position": "position", "Mean_Binder_Fraction": "mean_binder_fraction",
                 "N_Mutations": "n_mutations"})
    save("fig2b", f2b, ["Fig 2b: mean binder fraction per peptide position",
                        "source: results/analysis/experiment2_position_summary.csv (exp2.py)",
                        f"rows: {len(f2b)} - all nine positions, including P1"])
    for _, r in f2b.iterrows():
        macro(f"valBinderFracP{WORD[int(r.position)]}", r.mean_binder_fraction)

    # --- Fig 2c: R5 vs non-R5 binder fraction
    r5b = read("experiment2_r5_summary.csv")
    f2c = r5b[["R5_Group", "Mean_Binder_Fraction", "N_Mutations"]].rename(
        columns={"R5_Group": "group", "Mean_Binder_Fraction": "mean_binder_fraction",
                 "N_Mutations": "n_mutations"})
    save("fig2c", f2c, ["Fig 2c: mean binder fraction, R5 substitutions vs all other positions",
                        "source: results/analysis/experiment2_r5_summary.csv (exp2.py)",
                        "rows: 2 groups; the non-R5 group excludes P1 as stated in the Methods"])
    for _, r in f2c.iterrows():
        macro(f"valBinderFrac{tex_name(str(r.group).replace('-', ''))}", r.mean_binder_fraction)

    # --- Fig 3: overall Macro AUC0.1 per model (ascending, as the bars are ordered)
    overall = read("experiment1_overall_metrics.csv")
    overall["Model"] = overall.Model.map(canon)
    f3 = (overall.assign(model=overall.Model.map(LABEL))
          .sort_values("Macro_AUC0.1")[["model", "Macro_AUC0.1", "N_Eligible_Mutations"]]
          .rename(columns={"Macro_AUC0.1": "macro_auc01", "N_Eligible_Mutations": "n_mutations"}))
    save("fig3", f3, ["Fig 3: overall Macro AUC0.1 per model",
                      "source: results/analysis/experiment1_overall_metrics.csv (exp1.py)",
                      f"rows: {len(f3)} models, ascending by Macro AUC0.1 (bar order)",
                      "Each model's Macro AUC0.1 averages its AUC-eligible mutations."])
    for _, r in f3.iterrows():
        macro(f"valMacroAuc{tex_name(r.model)}", r.macro_auc01)
    o = overall.iloc[0]
    macro("valNpairs", f"{int(o.N_Pairs):,}")
    macro("valNpositive", f"{int(o.N_Positive_Pairs):,}")
    macro("valNnegative", f"{int(o.N_Negative_Pairs):,}")
    macro("valNmutations", int(o.N_Mutations), "{:d}")
    macro("valNeligible", int(o.N_Eligible_Mutations), "{:d}")
    macro("valNexcluded", int(o.N_Mutations - o.N_Eligible_Mutations), "{:d}")

    # --- Fig 4: distribution of mutation-specific AUC0.1 per model
    msa = read("mutation_specific_auc.csv")
    msa["Model"] = msa.Model.map(canon)
    rows, wide = [], {}
    for i, m in enumerate(MODELS, start=1):
        v = msa[msa.Model == m].AUC01.reset_index(drop=True)
        wide[LABEL[m]] = v
        rows.append({"draw_position": i, "model": LABEL[m], "n_mutations": len(v),
                     "lower_whisker": v.min(), "q1": v.quantile(.25), "median": v.median(),
                     "q3": v.quantile(.75), "upper_whisker": v.max()})
    f4 = pd.DataFrame(rows)
    save("fig4", f4, ["Fig 4: five-number summary of mutation-specific AUC0.1 per model",
                      "source: results/analysis/mutation_specific_auc.csv (exp3.py)",
                      "whiskers = min and max (no outlier trimming); quartiles = linear interpolation",
                      f"each model summarises the same {len(wide[LABEL[MODELS[0]]])} AUC-eligible mutations",
                      "row order = draw_position = the x order in the figure"])
    vals = pd.DataFrame(wide)
    vals.to_csv(R_FIGURES / "fig4_values.csv", index=False)
    write_provenance(R_FIGURES / "fig4_values.csv", [
        "Fig 4 backing data: every mutation-specific AUC0.1 behind the box statistics",
        "source: results/analysis/mutation_specific_auc.csv (exp3.py)",
        f"rows: {len(vals)} mutations x {len(vals.columns)} models - kept so the boxes can be recomputed"])
    print(f"  fig4_values.csv: {len(vals)} rows x {len(vals.columns)} cols (backing data)")

    # --- Fig 5: position x model mean AUC0.1 + best model per position
    grid = msa.groupby(["Model", "Mutation_Position"]).AUC01.mean().reset_index()
    grid["y"] = grid.Model.map({m: i for i, m in enumerate(MODELS, start=1)})
    grid["model"] = grid.Model.map(LABEL)
    f5 = (grid[["Mutation_Position", "y", "AUC01", "model"]]
          .rename(columns={"Mutation_Position": "x", "AUC01": "auc01"}).sort_values(["y", "x"]))
    save("fig5", f5, ["Fig 5: mean mutation-specific AUC0.1 per model x peptide position",
                      "source: results/analysis/mutation_specific_auc.csv (exp3.py)",
                      f"rows: {len(f5)} = {f5.model.nunique()} models x {f5.x.nunique()} positions",
                      "P1 is absent because no P1 substitution has both experimental classes,",
                      "so AUC0.1 is undefined there (a property of the data, not a filter)."])
    best = f5.loc[f5.groupby("x").auc01.idxmax(), ["x", "y", "model", "auc01"]].sort_values("x")
    best.to_csv(R_FIGURES / "fig5_best_source.csv", index=False)
    write_provenance(R_FIGURES / "fig5_best_source.csv", [
        "Fig 5 outline layer: the best model at each position",
        "computed as the argmax of fig5_source.csv within each position"])
    print(f"  fig5_best_source.csv: {len(best)} rows (outline layer)")
    for _, r in f5.iterrows():
        macro(f"valPosAuc{tex_name(r.model)}P{WORD[int(r.x)]}", r.auc01)
    for _, r in best.iterrows():
        macro(f"valBestModelP{WORD[int(r.x)]}", r.model)

    # --- Fig 6: R5 vs non-R5 mean AUC0.1 per model
    r5a = read("r5_vs_nonr5_auc_summary.csv")
    r5a["Model"] = r5a.Model.map(canon)
    piv = r5a.pivot(index="Model", columns="R5_Status", values="Mean_AUC01").reindex(MODELS)
    nmut = r5a.pivot(index="Model", columns="R5_Status", values="N_Mutations").reindex(MODELS)
    f6 = pd.DataFrame({"model": [LABEL[m] for m in piv.index],
                       "r5": piv["R5"].values, "non_r5": piv["non-R5"].values,
                       "n_r5": nmut["R5"].values, "n_non_r5": nmut["non-R5"].values})
    save("fig6", f6, ["Fig 6: mean AUC0.1 for R5 substitutions vs all other positions, per model",
                      "source: results/analysis/r5_vs_nonr5_auc_summary.csv (exp3.py)",
                      f"rows: {len(f6)} models; n_r5 / n_non_r5 give the mutation counts behind each mean"])
    for _, r in f6.iterrows():
        macro(f"valRfiveAuc{tex_name(r.model)}", r.r5)
        macro(f"valNonRfiveAuc{tex_name(r.model)}", r.non_r5)
    macro("valNrfive", int(nmut["R5"].iloc[0]), "{:d}")
    macro("valNnonrfive", int(nmut["non-R5"].iloc[0]), "{:d}")
    macro("valBestModelRfive", LABEL[piv["R5"].idxmax()])

    # --- Fig 7a/7b: severity vs cross-model performance (one point per eligible mutation)
    sev = read("fig_severity_vs_mean_auc.csv")
    save("fig7", sev, ["Fig 7a/7b: experimental severity vs mean AUC0.1 across models",
                       "source: results/analysis/fig_severity_vs_mean_auc.csv (exp3.py)",
                       f"rows: {len(sev)} AUC-eligible mutations; both panels read this one table",
                       "Mean_AUC01 averages the eight models for that mutation."])
    for tag, col in (("Disruption", "Experimental_Disruption"), ("BinderLoss", "Binder_Loss_Fraction")):
        rho, p = spearmanr(sev[col], sev.Mean_AUC01)
        mant, exp = f"{p:.1e}".split("e")
        macro(f"valRho{tag}", abs(rho))
        macro(f"valP{tag}", f"{mant}\\times10^{{{int(exp)}}}")
    macro("valNmodels", int(sev.N_Models.iloc[0]), "{:d}")
    macro("valNtcrs", int(heat.TCR_Group.nunique()), "{:d}")

    # --- numbers the Methods/Results quote that are not on a figure axis
    ovl = read("overlap_summary.csv")
    ovl = ovl[ovl.dataset == "fingerprinting"].copy()
    ovl["model"] = ovl.model.map(canon)
    npep = 172  # peptide species in the panel: reference + 171 substitutions
    macro("valNpeptides", npep, "{:d}")
    over = ovl[ovl.pep_uniq > 0]
    if len(over) != 1:
        die(f"expected exactly one model with peptide overlap, found {len(over)}")
    r = over.iloc[0]
    macro("valOverlapModel", PROSE_NAME[r.model])
    macro("valOverlapPeptidesWord", SPELLED[int(r.pep_uniq)])
    macro("valOverlapPairRowsWord", SPELLED[int(r.pair_rows)])
    for _, r in ovl.set_index("model").iterrows():
        pass
    for key, row in ovl.set_index("model").iterrows():
        tag = tex_name(LABEL[key])
        total = int(row.total_rows)
        macro(f"valOvlPepUniq{tag}", int(row.pep_uniq), "{:d}")
        macro(f"valOvlPepPct{tag}", row.pep_uniq / npep * 100, "{:.1f}")
        macro(f"valOvlPepRows{tag}", int(row.pep_rows), "{:d}")
        macro(f"valOvlPepRowPct{tag}", row.pep_rows / total * 100, "{:.1f}")
        macro(f"valOvlPairRows{tag}", int(row.pair_rows), "{:d}")
        macro(f"valOvlPairPct{tag}", row.pair_rows / total * 100, "{:.1f}")

    out = R_FIGURES / "values.tex"
    with open(out, "w") as f:
        f.write("% GENERATED by stages/stage7_figures.py --prepare - do not edit.\n"
                "% Every number the manuscript states in prose or a caption is defined here.\n")
        for name, val in macros:
            f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    print(f"  values.tex: {len(macros)} macros")


def render(make_images=True):
    banner("stage 7 --render", "source tables + templates -> LaTeX and images")
    pre = need(TEMPLATES / "figure_preamble.tex.in", "shared figure preamble template").read_text()
    for name in re.findall(r"\{\{DATA:([^}]+)\}\}", pre):
        need(R_FIGURES / name, f"source table {name} for the figure preamble", "run --prepare first")
    pre = re.sub(r"\{\{DATA:([^}]+)\}\}", lambda m: m.group(1), pre)
    (R_FIGURES / "figure_preamble.tex").write_text(
        pre.replace("% TEMPLATE for the shared figure preamble - rendered to",
                    "% GENERATED from templates/figure_preamble.tex.in - do not edit; rendered to"))
    print("  figure_preamble.tex  <- templates/figure_preamble.tex.in (colormaps + pgfplots setup)")
    rendered = []
    for fid in FIGURES:
        tpl = need(TEMPLATES / f"{fid}.tex.in", f"template for {fid}").read_text()
        used = re.findall(r"\{\{DATA:([^}]+)\}\}", tpl)
        for name in used:
            need(R_FIGURES / name, f"source table {name} for {fid}", "run --prepare first")
        body = re.sub(r"\{\{DATA:([^}]+)\}\}", lambda m: m.group(1), tpl)
        body = body.replace(f"% TEMPLATE for {fid} - rendered by",
                            f"% GENERATED from templates/{fid}.tex.in by")
        out = R_FIGURES / f"{fid}.tex"
        # plain concatenation: the body is LaTeX and full of % characters
        # trailing % : without it the file's final newline becomes a space in the figure body
        out.write_text(f"% GENERATED - edit templates/{fid}.tex.in and re-run stage 7, "
                       f"not this file.\n" + body.rstrip("\n") + "%\n")
        rendered.append((fid, used))
        print(f"  {fid}.tex  <- templates/{fid}.tex.in + {used or ['(no table)']}")
    if make_images:
        build_images([f for f, _ in rendered])
    return rendered


def build_images(fids):
    """Compile each fragment on its own so reviewers can look at a figure without LaTeX."""
    engine = shutil.which("pdflatex")
    docker = shutil.which("docker")
    if not engine and not docker:
        print("  (no pdflatex and no docker: skipping images; the .tex fragments are still written)")
        return
    wrapper = (r"\documentclass[border=4pt]{standalone}"
               "\n" r"\usepackage{pgfplots}\usepackage{pgfplotstable}"
               "\n" r"\input{figure_preamble.tex}"
               "\n" r"\pgfplotsset{table/search path={.}}"
               "\n" r"\begin{document}" "\n" r"\input{FRAGMENT}" "\n" r"\end{document}" "\n")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in R_FIGURES.glob("*_source.csv"):
            shutil.copy2(f, td / f.name)
        for f in R_FIGURES.glob("fig4_values.csv"):
            shutil.copy2(f, td / f.name)
        shutil.copy2(R_FIGURES / "figure_preamble.tex", td / "figure_preamble.tex")
        for fid in fids:
            shutil.copy2(R_FIGURES / f"{fid}.tex", td / f"{fid}.tex")
            (td / f"{fid}_standalone.tex").write_text(wrapper.replace("FRAGMENT", fid))
        cmd = "for f in *_standalone.tex; do pdflatex -interaction=nonstopmode -halt-on-error $f " \
              "> ${f%.tex}.log 2>&1 || echo FAILED $f; done"
        if engine:
            r = subprocess.run(["sh", "-c", cmd], cwd=td, capture_output=True, text=True)
            out = r.stdout
        else:
            r = subprocess.run(["docker", "run", "--rm", "-u", f"{__import__('os').getuid()}:{__import__('os').getgid()}",
                                "-v", f"{td}:/w", "-w", "/w", "texlive/texlive:latest", "sh", "-c", cmd],
                               capture_output=True, text=True)
            out = r.stdout
        failed = [l for l in out.splitlines() if l.startswith("FAILED")]
        made = 0
        for fid in fids:
            pdf = td / f"{fid}_standalone.pdf"
            if not pdf.exists():
                log = td / f"{fid}_standalone.log"
                tail = log.read_text()[-800:] if log.exists() else "(no log)"
                die(f"figure image for {fid} did not compile:\n{tail}")
            shutil.copy2(pdf, R_FIGURES / f"{fid}.pdf")
            made += 1
            if shutil.which("pdftoppm"):
                subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile",
                                str(R_FIGURES / f"{fid}.pdf"), str(R_FIGURES / fid)], check=True)
        if failed:
            die("some figures failed to compile: " + ", ".join(failed))
        png = len(list(R_FIGURES.glob("*.png")))
        print(f"  images: {made} PDF" + (f" + {png} PNG" if png else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    if not a.prepare and not a.render:
        a.prepare = a.render = True
    if a.prepare:
        prepare()
    if a.render:
        render(make_images=not a.no_images)
    print("\nstage 7 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
