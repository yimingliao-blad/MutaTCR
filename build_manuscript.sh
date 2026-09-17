#!/usr/bin/env bash
# Compile manuscript/main.tex. Uses a local pdflatex when there is one, otherwise the
# texlive/texlive Docker image. The PDF lands at manuscript/main.pdf.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The manuscript is kept outside this repository; point at it with MUTATCR_MANUSCRIPT.
MANUSCRIPT="${MUTATCR_MANUSCRIPT:-$ROOT/manuscript}"
if [ ! -f "$MANUSCRIPT/main.tex" ]; then
    echo "No manuscript at $MANUSCRIPT"
    echo "The manuscript is maintained separately. Set MUTATCR_MANUSCRIPT=/path/to/manuscript,"
    echo "where main.tex includes ../results/figures/*.tex relative to this repository."
    exit 1
fi

# The figures plot their source tables at compile time, and those are regenerated rather than
# committed (see .gitignore). Say so plainly instead of letting pgfplots fail file by file.
missing=0
for f in "$ROOT"/results/figures/fig*_source.csv; do [ -e "$f" ] && { missing=0; break; } || missing=1; done
if [ "$missing" = 1 ] || [ ! -e "$ROOT/results/figures/values.tex" ]; then
    echo "The figures' source tables are not built yet."
    echo "Run:  ./run_all.sh --no-pdf     (or: stages 1-7 individually)"
    exit 1
fi

cd "$MANUSCRIPT"

run() {  # three passes plus bibtex, so references and the bibliography settle
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    bibtex main > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1
}

if command -v pdflatex > /dev/null; then
    run
elif command -v docker > /dev/null; then
    docker run --rm -u "$(id -u):$(id -g)" -v "$ROOT:/repo" -v "$MANUSCRIPT:/manuscript" -w /manuscript \
        texlive/texlive:latest sh -c '
        pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
        bibtex main > /dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1'
else
    echo "no pdflatex and no docker: cannot compile (the .tex sources and figure images are still built)"
    exit 1
fi

if grep -qE '^! ' build.log; then
    echo "LaTeX reported errors:"; grep -E '^! ' build.log | head; exit 1
fi
grep -E 'Output written' build.log
