#!/usr/bin/env python3
"""Is this machine able to rebuild the repository?

Reports the Python packages the stages import, the versions used for the committed results, and
whether a LaTeX toolchain with pgfplots is reachable (needed for the figure images and the PDF).
Exits 1 if anything required is missing, so run_all.sh stops before a stage fails halfway.

Usage: python3 checks/check_dependencies.py [--quiet]
"""
import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# import name, pip name, minimum, version used for the committed results, what needs it
REQUIRED = [
    ("pandas", "pandas", (1, 3), "2.3.3", "every stage"),
    ("numpy", "numpy", (1, 20), "2.0.2", "every stage"),
    ("scipy", "scipy", (1, 7), "1.13.1", "stages 5 and 7 (Spearman)"),
    ("sklearn", "scikit-learn", (1, 0), "1.6.1", "stages 5 (partial AUC)"),
    ("openpyxl", "openpyxl", (3, 0), "3.1.5", "stage 1 (reads the raw .xlsx)"),
    ("yaml", "PyYAML", (5, 4), "6.0.3", "config/benchmark_config.yaml"),
]
OPTIONAL = [
    ("matplotlib", "matplotlib", (3, 4), "3.9.4", "stage 8 supplementary figures"),
    ("PIL", "Pillow", (8, 0), "11.3.0", "stage 8 compound figure"),
    ("cairosvg", "cairosvg", (2, 5), "2.8.2", "stage 8 compound figure panel (a)"),
]
problems = []


def version_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check_group(group, required):
    for mod, pip_name, minimum, used, why in group:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            line = f"MISSING   {pip_name:14s} (need >= {'.'.join(map(str, minimum))}) - {why}"
            print(line)
            if required:
                problems.append(f"{pip_name} not installed")
            continue
        have = getattr(m, "__version__", "?")
        ok = have == "?" or version_tuple(have) >= minimum
        print(f"{'ok' if ok else 'TOO OLD':9s} {pip_name:14s} {have:10s} "
              f"(need >= {'.'.join(map(str, minimum))}, committed results used {used}) - {why}")
        if not ok and required:
            problems.append(f"{pip_name} {have} < {'.'.join(map(str, minimum))}")


def check_latex():
    engine = shutil.which("pdflatex")
    docker = shutil.which("docker")
    if engine:
        print(f"ok        pdflatex       {engine}")
        r = subprocess.run(["kpsewhich", "pgfplots.sty"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            print(f"ok        pgfplots       {r.stdout.strip()}")
        else:
            print("MISSING   pgfplots.sty   - install texlive-pictures / pgfplots >= 1.18")
            problems.append("pgfplots not found for the local pdflatex")
        return
    if docker:
        print("ok        docker         no local pdflatex; build_manuscript.sh will use "
              "the texlive/texlive image")
        return
    print("MISSING   LaTeX          - no pdflatex and no docker: figure images and the PDF "
          "cannot be built (the .tex sources still are)")
    problems.append("no LaTeX toolchain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.parse_args()
    print(f"python    {sys.version.split()[0]}  ({sys.executable})")
    if version_tuple(sys.version.split()[0]) < (3, 9):
        problems.append(f"python {sys.version.split()[0]} < 3.9")
    print("\n-- required (stages 1-7, the manuscript's numbers)")
    check_group(REQUIRED, required=True)
    print("\n-- optional (stage 8 supplementary figures)")
    check_group(OPTIONAL, required=False)
    print("\n-- LaTeX (figure images and the manuscript PDF)")
    check_latex()
    print()
    if problems:
        print("MISSING DEPENDENCIES:")
        for p in problems:
            print(f"  - {p}")
        print("\n  pip install -r requirements.txt      (or: conda env create -f environment.yml)")
        return 1
    print("PASS: everything needed to rebuild this repository is available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
