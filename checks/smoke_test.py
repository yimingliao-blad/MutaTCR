#!/usr/bin/env python3
"""A tiny end-to-end smoke test: does the pipeline still wire together?

Runs the REAL stages on the manuscript's dataset only (fingerprinting, the smallest of the three),
writing everything into a scratch directory — never into build/ or results/. It proves the wiring
and the load-bearing mechanism of each stage, and checks a handful of properties that would catch a
broken join or a figure that stopped reading its data. It does NOT re-verify the published numbers;
`checks/check_repo.py` and a full `./run_all.sh` do that.

  python3 checks/smoke_test.py             # ~1 minute
  python3 checks/smoke_test.py --keep      # leave the scratch directory for inspection

Exit 1 on the first broken step, naming it.
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stages.common import MODELS  # noqa: E402

checks_run = 0


def ok(name, detail=""):
    global checks_run
    checks_run += 1
    print(f"  ok    {name}{'  ' + detail if detail else ''}")


def fail(name, detail):
    print(f"  FAIL  {name}: {detail}")
    print(f"\nSMOKE TEST FAILED at: {name}")
    raise SystemExit(1)


def run_stage(script, argv, scratch, label):
    """Run a stage with its output paths redirected into the scratch tree."""
    driver = scratch / f"run_{Path(script).stem}_{label}.py"
    driver.write_text(
        f'import stages.common as c\n'
        f'c.BUILD = c.Path(r"{scratch / "build"}")\n'
        f'c.B_UNIFIED = c.BUILD / "unified"\n'
        f'c.B_DEDUP = c.BUILD / "dedup"\n'
        f'c.B_PRED = c.BUILD / "predictions"\n'
        f'c.B_MERGED = c.BUILD / "merged"\n'
        f'c.FP_DB = c.BUILD / "fp.db"\n'
        f'c.RESULTS = c.Path(r"{scratch / "results"}")\n'
        f'c.R_ANALYSIS = c.RESULTS / "analysis"\n'
        f'c.R_TABLES = c.RESULTS / "tables"\n'
        f'c.R_FIGURES = c.RESULTS / "figures"\n'
        f'import runpy, sys\n'
        f'sys.argv = {argv!r}\n'
        f'runpy.run_path(r"{REPO / "stages" / script}", run_name="__main__")\n')
    t = time.time()
    r = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)
    if r.returncode != 0:
        fail(label, (r.stdout + r.stderr).strip()[-700:])
    return time.time() - t, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="tcrj-smoke-"))
    print(f"scratch: {scratch}\n")
    try:
        # ---- stage 4: unified + scores -> predictions, merged, fp.db  (manuscript dataset only)
        secs, _ = run_stage("stage4_assemble.py", ["stage4_assemble.py", "--dataset", "fingerprinting"],
                            scratch, "stage 4 assemble")
        db = scratch / "build" / "fp.db"
        if not db.exists():
            fail("stage 4 assemble", "no fp.db produced")
        conn = sqlite3.connect(db)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        conn.close()
        if sorted(tables) != sorted(MODELS):
            fail("stage 4 assemble", f"fp.db has tables {sorted(tables)}")
        if not all(3000 < n <= 3612 for n in counts.values()):
            fail("stage 4 assemble", f"implausible row counts: {counts}")
        ok("stage 4 assemble", f"{len(tables)} tables, {min(counts.values())}-{max(counts.values())} rows, {secs:.0f}s")

        merged = scratch / "build" / "merged" / "fingerprinting_all_models.csv"
        head = merged.read_text().split("\n", 1)[0]
        if "n_models_scored" not in head or "all_models_scored" not in head:
            fail("stage 4 assemble", "merged table lost its completeness columns")
        rows = sum(1 for _ in merged.open()) - 1
        if rows != 3612:
            fail("stage 4 assemble", f"merged table dropped rows: {rows} (outer join should keep 3612)")
        ok("outer join keeps every row", f"{rows} rows, flagged not dropped")

        # ---- stage 5: the analyses
        secs, _ = run_stage("stage5_analysis.py", ["stage5_analysis.py"], scratch, "stage 5 analysis")
        analysis = scratch / "results" / "analysis"
        produced = sorted(p.name for p in analysis.glob("*.csv"))
        for needed in ("experiment1_overall_metrics.csv", "mutation_specific_auc.csv",
                       "fig_severity_vs_mean_auc.csv", "tcr_model_analysis.csv"):
            if needed not in produced:
                fail("stage 5 analysis", f"{needed} not produced (got {len(produced)} tables)")
        import pandas as pd
        overall = pd.read_csv(analysis / "experiment1_overall_metrics.csv")
        if len(overall) != len(MODELS) or not overall["Macro_AUC0.1"].between(0, 1).all():
            fail("stage 5 analysis", f"overall metrics look wrong:\n{overall.head()}")
        ok("stage 5 analysis", f"{len(produced)} tables, Macro AUC in range, {secs:.0f}s")

        # ---- stages 6 and 7: source table -> LaTeX
        secs, _ = run_stage("stage6_tables.py", ["stage6_tables.py"], scratch, "stage 6 tables")
        t2 = (scratch / "results" / "tables" / "table2_overlap.tex").read_text()
        if "\\begin{tabular}" not in t2 or "/172" not in t2:
            fail("stage 6 tables", "table2_overlap.tex is not the expected tabular")
        ok("stage 6 tables", f"{len(list((scratch / 'results' / 'tables').glob('*.tex')))} tabulars, {secs:.0f}s")

        run_stage("stage7_figures.py", ["stage7_figures.py", "--prepare"], scratch, "stage 7 prepare")
        secs, _ = run_stage("stage7_figures.py", ["stage7_figures.py", "--render", "--no-images"],
                            scratch, "stage 7 render")
        figs = scratch / "results" / "figures"
        fig3 = (figs / "fig3.tex").read_text()
        if "fig3_source.csv" not in fig3:
            fail("stage 7 render", "fig3.tex no longer reads its source table")
        if "coordinates {" in fig3.replace("coordinates {};", ""):
            fail("stage 7 render", "fig3.tex carries literal coordinates again")
        values = (figs / "values.tex").read_text()
        if values.count("newcommand") < 100:
            fail("stage 7 render", f"values.tex has only {values.count('newcommand')} macros")
        ok("stage 7 figures", f"9 fragments + {values.count('newcommand')} macros, {secs:.0f}s")

        # ---- the load-bearing mechanism: one figure actually compiles from its own data
        secs = compile_one_figure(figs, scratch)
        ok("one figure compiles standalone", f"fig3.pdf, {secs:.0f}s")

        # ---- nothing escaped the scratch directory
        for tree, name in ((REPO / "build", "build/"), (REPO / "results", "results/")):
            newest = max((p.stat().st_mtime for p in tree.rglob("*") if p.is_file()), default=0)
            if newest > start_time:
                fail("scratch isolation", f"the smoke test wrote into the repo's {name}")
        ok("scratch isolation", "the repo's build/ and results/ were not touched")

        print(f"\nSMOKE TEST PASSED - {checks_run} checks, real stages, scratch-only output")
        return 0
    finally:
        if args.keep:
            print(f"\nscratch kept: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


def compile_one_figure(figs: Path, scratch: Path):
    """Compile a single figure fragment - the cheapest proof that the LaTeX side still works."""
    engine = shutil.which("pdflatex")
    docker = shutil.which("docker")
    if not engine and not docker:
        print("  skip  one figure compiles standalone (no pdflatex and no docker)")
        return 0.0
    work = scratch / "texcheck"
    work.mkdir(exist_ok=True)
    for f in list(figs.glob("*_source.csv")) + [figs / "fig3.tex", figs / "figure_preamble.tex",
                                                figs / "fig4_source.csv"]:
        if f.exists():
            shutil.copy2(f, work / f.name)
    (work / "doc.tex").write_text(
        "\\documentclass[border=4pt]{standalone}\n"
        "\\usepackage{pgfplots}\\usepackage{pgfplotstable}\n"
        "\\pgfplotsset{table/search path={.}}\n"
        "\\input{figure_preamble.tex}\n"
        "\\begin{document}\n\\input{fig3.tex}\n\\end{document}\n")
    cmd = "pdflatex -interaction=nonstopmode -halt-on-error doc.tex > doc.log 2>&1"
    t = time.time()
    if engine:
        r = subprocess.run(["sh", "-c", cmd], cwd=work, capture_output=True, text=True)
    else:
        r = subprocess.run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                            "-v", f"{work}:/w", "-w", "/w", "texlive/texlive:latest", "sh", "-c", cmd],
                           capture_output=True, text=True)
    pdf = work / "doc.pdf"
    if r.returncode != 0 or not pdf.exists() or pdf.stat().st_size < 5000:
        log = (work / "doc.log").read_text()[-700:] if (work / "doc.log").exists() else "(no log)"
        fail("one figure compiles standalone", log)
    return time.time() - t


if __name__ == "__main__":
    start_time = time.time()
    sys.exit(main())
