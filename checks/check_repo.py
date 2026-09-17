#!/usr/bin/env python3
"""Is this repository standalone, and does it still reproduce its own results?

  A. Standalone: no tracked file reads a path outside the repo, every file the manuscript and the
     figure fragments include exists, and every input the stages need is committed.
  B. Reproducible: re-running the table and figure stages into a temp directory reproduces the
     committed results/ byte for byte.
  C. Consistent: every \\val macro main.tex uses is defined, no figure body carries typed-in data,
     and the argmax/ordering claims in the text still hold in the data.
  D. Accounted for: the stages' row counts add up (scored + unscored = dataset rows).

Usage: python3 checks/check_repo.py        (exit 1 on any failure)
"""
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stages.common import (B_MERGED, DATASETS, MODELS, R_ANALYSIS, R_FIGURES, R_TABLES,  # noqa: E402
                           canon)

failures = []
# Paths outside the repo are allowed only where they are documented as unavailable inputs.
ALLOWED_OUTSIDE = {
    "config/benchmark_config.yaml",   # model repos/weights, documented in docs/INFERENCE.md
    "docs/INFERENCE.md",
    "docs/PROVENANCE.md",
    "data/PROVENANCE.tsv",            # records where the canonical data came from
    "README.md",
}


def report(ok, name, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else ': ' + detail}")
    if not ok:
        failures.append(name)


def tracked_files():
    r = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return [REPO / p for p in r.stdout.split()]
    out = []
    for dp, dns, fns in os.walk(REPO):
        dns[:] = [d for d in dns if d not in {".git", "build", "__pycache__"}]
        out += [Path(dp) / f for f in fns]
    return out


def check_a():
    # A1: no absolute outside paths in code/tex
    offenders = []
    for f in tracked_files():
        rel = f.relative_to(REPO).as_posix()
        if rel in ALLOWED_OUTSIDE or f.suffix not in {".py", ".sh", ".tex", ".in", ".yaml", ".json"}:
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for m in re.finditer(r"[\"'](/(?:home|Users|mnt|data|opt)/[^\"']+)[\"']", text):
            offenders.append(f"{rel}: {m.group(1)}")
    report(not offenders, "A1. no tracked file hard-codes a path outside the repo",
           f"{len(offenders)} found, e.g. {offenders[:3]}")

    # A2: everything the manuscript includes exists
    tex = (REPO / "manuscript" / "main.tex").read_text()
    missing = []
    for inc in re.findall(r"\\input\{([^}]+)\}", tex):
        p = (REPO / "manuscript" / inc).resolve()
        if not p.exists():
            missing.append(inc)
    for img in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (REPO / "manuscript" / img).exists():
            missing.append(img)
    report(not missing, "A2. every file main.tex includes exists", f"missing: {missing}")

    # A3: every data file a figure fragment plots exists
    missing = []
    for frag in sorted(R_FIGURES.glob("fig*.tex")):
        for t in re.findall(r"\{([A-Za-z0-9_]+\.csv)\}", frag.read_text()):
            if not (R_FIGURES / t).exists():
                missing.append(f"{frag.name} -> {t}")
    for t in re.findall(r"\{([A-Za-z0-9_]+\.csv)\}", (R_FIGURES / "figure_preamble.tex").read_text()):
        if not (R_FIGURES / t).exists():
            missing.append(f"figure_preamble.tex -> {t}")
    report(not missing, "A3. every source table a figure plots exists", f"missing: {missing}")

    # A4: the committed inputs the stages need
    need = [REPO / "data/unified" / f"{ds}_unified.csv" for ds in DATASETS]
    need += [REPO / "data/scores" / f"{ds}_model_scores.csv" for ds in DATASETS]
    need += [REPO / "data/dedup/removed_ids.csv", REPO / "data/overlap/overlap_summary.csv",
             REPO / "data/raw/FingerPrinting/Fingerprinting_TCR_Data_Combined_final.xlsx",
             REPO / "data/raw/IMMREP23/immref23.csv",
             REPO / "data/raw/TetTCR-SeqHD/TCR_antigen_binding_sheet.csv"]
    absent = [str(p.relative_to(REPO)) for p in need if not p.exists()]
    report(not absent, "A4. every committed input the stages need is present", f"absent: {absent}")


def check_b():
    """Re-run stages 6 and 7 into a temp tree and compare with the committed results."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        env = dict(os.environ, PYTHONPATH=str(REPO))
        shadow = td / "results"
        (shadow / "analysis").mkdir(parents=True)
        for f in R_ANALYSIS.glob("*.csv"):
            shutil.copy2(f, shadow / "analysis" / f.name)
        # run the two stages with results/ redirected into the temp tree
        patch = (f'import stages.common as c\n'
                 f'c.RESULTS = c.Path(r"{shadow}")\n'
                 f'c.R_ANALYSIS = c.RESULTS / "analysis"\n'
                 f'c.R_TABLES = c.RESULTS / "tables"\n'
                 f'c.R_FIGURES = c.RESULTS / "figures"\n'
                 f'import runpy, sys\n')
        for stage, out_dir, committed in (("stage6_tables.py", "tables", R_TABLES),
                                          ("stage7_figures.py", "figures", R_FIGURES)):
            script = td / f"run_{stage}"
            script.write_text(patch + f'sys.argv = ["{stage}"' +
                              (', "--render", "--no-images"' if stage.startswith("stage7") else "") +
                              f']\nrunpy.run_path(r"{REPO / "stages" / stage}", run_name="__main__")\n')
            if stage.startswith("stage7"):
                # --prepare then --render, without images (images need LaTeX)
                s1 = td / "run_prep.py"
                s1.write_text(patch + f'sys.argv = ["{stage}", "--prepare"]\n'
                                      f'runpy.run_path(r"{REPO / "stages" / stage}", run_name="__main__")\n')
                r = subprocess.run([sys.executable, str(s1)], capture_output=True, text=True, env=env, cwd=REPO)
                if r.returncode != 0:
                    report(False, f"B. re-run {stage} --prepare", r.stderr.strip()[-300:])
                    return
            r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=REPO)
            if r.returncode != 0:
                report(False, f"B. re-run {stage}", r.stderr.strip()[-300:])
                return
            fresh = shadow / out_dir
            names = {p.name for p in committed.glob("*")} - {p.name for p in committed.glob("*.pdf")} \
                    - {p.name for p in committed.glob("*.png")}
            diffs = [n for n in sorted(names)
                     if (fresh / n).exists() and not filecmp.cmp(committed / n, fresh / n, shallow=False)]
            absent = [n for n in sorted(names) if not (fresh / n).exists()]
            report(not diffs and not absent, f"B. committed results/{out_dir} match a fresh run",
                   f"differ: {diffs}; not regenerated: {absent}")


def check_c():
    tex = (REPO / "manuscript" / "main.tex").read_text()
    values = (R_FIGURES / "values.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\(val[A-Za-z]+)\}", values))
    used = set(re.findall(r"\\(val[A-Za-z]+)", tex))
    report(not (used - defined), "C1. every \\val macro used in main.tex is defined",
           f"undefined: {sorted(used - defined)}")
    body = tex + "".join(p.read_text() for p in R_FIGURES.glob("fig*.tex"))
    typed = re.findall(r"\\addplot(?:\+)?[^;]*?coordinates\s*\{\s*\(", body)
    typed += re.findall(r"boxplot prepared=\{[^}]*?=\s*0?\.\d+", body)
    typed += re.findall(r"\)\s*\[\s*0\.\d+\s*\]", body)
    report(not typed, "C2. no figure carries typed-in data values", f"{len(typed)} literal(s)")

    msa = pd.read_csv(R_ANALYSIS / "mutation_specific_auc.csv")
    msa["Model"] = msa.Model.map(canon)
    grid = msa.groupby(["Model", "Mutation_Position"]).AUC01.mean()
    best = pd.read_csv(R_FIGURES / "fig5_best_source.csv")
    wrong = [int(r.x) for _, r in best.iterrows()
             if grid[grid.index.get_level_values(1) == int(r.x)].idxmax()[0] != canon(r.model)]
    report(not wrong, "C3. the best-model-per-position outlines are the data's argmax",
           f"positions disagreeing: {wrong}")
    r5 = pd.read_csv(R_ANALYSIS / "r5_vs_nonr5_auc_summary.csv")
    r5["Model"] = r5.Model.map(canon)
    piv = r5.pivot(index="Model", columns="R5_Status", values="Mean_AUC01")
    claimed = re.search(r"\\newcommand\{\\valBestModelRfive\}\{([^}]*)\}", values).group(1)
    report(canon(claimed) == piv["R5"].idxmax(), "C4. \\valBestModelRfive is the data's best model on R5",
           f"macro says {claimed}, data says {piv['R5'].idxmax()}")
    report(bool((piv["R5"] < piv["non-R5"]).all()),
           "C5. 'R5 below non-R5 for all eight models' holds",
           f"exceptions: {list(piv.index[piv['R5'] >= piv['non-R5']])}")


def check_d():
    cov = pd.read_csv(R_ANALYSIS / "coverage_summary.csv")
    bad = cov[cov.scored + cov.unscored != cov.rows]
    report(bad.empty, "D1. scored + unscored = dataset rows, for every model",
           f"{len(bad)} row(s) do not add up")
    ok = True
    for ds in DATASETS:
        mg = pd.read_csv(B_MERGED / f"{ds}_all_models.csv") if (B_MERGED / f"{ds}_all_models.csv").exists() else None
        if mg is None:
            continue
        u = pd.read_csv(REPO / "data/unified" / f"{ds}_unified.csv")
        ok &= len(mg) == len(u)
    report(ok, "D2. the merged tables keep every unified row (outer join, nothing dropped)")


def main():
    print(f"checking {REPO}\n")
    check_a()
    check_b()
    check_c()
    check_d()
    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed: {failures}")
        return 1
    print("PASS: the repository is standalone, reproducible and internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
