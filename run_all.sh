#!/usr/bin/env bash
# Rebuild everything in this repository, in order, from the raw data to the compiled manuscript.
# Every stage is independently runnable; this is just the order they belong in.
#
#   ./run_all.sh              stages 1-7 + the manuscript PDF (needs a LaTeX engine or Docker)
#   ./run_all.sh --no-pdf     stop after the figures and tables
#   ./run_all.sh --with-supplementary   also build the exploratory figure suite (stage 8, slow)
#
# Stage 3 (model inference) is not run: it needs ~10 GB of published model weights and eight conda
# environments. Its outputs ship with the repo as data/scores/*.csv - see docs/INFERENCE.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${TCRJ_PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "python not found: $PY (set TCRJ_PYTHON)"; exit 1; }
"$PY" checks/check_dependencies.py    # stops here if something needed is missing

WITH_PDF=1; WITH_SUPP=0
for a in "$@"; do
    case "$a" in
        --no-pdf) WITH_PDF=0 ;;
        --with-supplementary) WITH_SUPP=1 ;;
        *) echo "unknown option: $a"; exit 2 ;;
    esac
done

"$PY" stages/stage1_preprocess.py          # raw            -> build/unified (compared with data/unified)
"$PY" stages/stage3_inference.py --check   # report the committed model scores
"$PY" stages/stage2_dedup.py               # unified + removed ids -> build/dedup
"$PY" stages/stage4_assemble.py            # unified + scores      -> predictions, merged, fp.db
"$PY" stages/stage5_analysis.py            # fp.db + merged        -> results/analysis
"$PY" stages/stage6_tables.py              # analysis -> results/tables (source csv + .tex)
"$PY" stages/stage7_figures.py             # analysis -> results/figures (source csv + .tex + image)
"$PY" checks/check_repo.py                 # standalone + reproducibility checks

if [ "$WITH_SUPP" = 1 ]; then
    "$PY" stages/stage8_supplementary_figures.py
fi

if [ "$WITH_PDF" = 1 ]; then
    echo
    echo "=== compiling the manuscript"
    ./build_manuscript.sh
fi

echo
echo "ALL STAGES PASSED"
