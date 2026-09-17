"""Shared paths, constants and fail-loud helpers for the staged pipeline.

Data policy: every stage keeps as much information as it has. Rows are never silently dropped —
missing model scores stay as empty cells with a `n_models_scored` count beside them, and any
narrowing happens in the final figure/table stage, where it is counted and written into the
artifact's own provenance file.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# committed inputs
DATA = REPO / "data"
RAW = DATA / "raw"
UNIFIED = DATA / "unified"
SCORES = DATA / "scores"
DEDUP_IDS = DATA / "dedup" / "removed_ids.csv"
OVERLAP = DATA / "overlap" / "overlap_summary.csv"

# generated, reproducible, not committed
BUILD = REPO / "build"
B_UNIFIED = BUILD / "unified"
B_DEDUP = BUILD / "dedup"
B_PRED = BUILD / "predictions"
B_MERGED = BUILD / "merged"
FP_DB = BUILD / "fp.db"

# committed outputs
RESULTS = REPO / "results"
R_ANALYSIS = RESULTS / "analysis"
R_TABLES = RESULTS / "tables"
R_FIGURES = RESULTS / "figures"
MANUSCRIPT = REPO / "manuscript"
M_GENERATED = MANUSCRIPT / "generated"

# The model order used throughout the manuscript figures (best overall first).
MODELS = ["ERGO", "EPACT", "PanPep", "TITAN", "NetTCR", "NetTCR22", "ERGO2", "SCEPTR"]
# As written in the manuscript.
LABEL = {"ERGO": "ERGO", "EPACT": "EPACT", "PanPep": "PanPep", "TITAN": "TITAN",
         "NetTCR": "NetTCR", "NetTCR22": "NetTCR2.2", "ERGO2": "ERGO2", "SCEPTR": "SCEPTR"}
PROSE_NAME = {"ERGO": "ERGO", "ERGO2": "ERGO-II", "NetTCR": "NetTCR-2.0", "NetTCR22": "NetTCR-2.2",
              "TITAN": "TITAN", "PanPep": "PanPep", "SCEPTR": "SCEPTR", "EPACT": "EPACT"}
# Every spelling of a model seen in an input file or a label -> one canonical key.
CANON = {**{m: m for m in MODELS}, "NetTCR2.2": "NetTCR22", "NetTCR-2.2": "NetTCR22",
         "NetTCR2.0": "NetTCR", "NetTCR-2.0": "NetTCR", "ERGO-II": "ERGO2", "NETTCR22": "NetTCR22"}
DATASETS = ["fingerprinting", "immrep23", "tettcr"]
# The manuscript's benchmark.
MAIN_DATASET = "fingerprinting"
REFERENCE_PEPTIDE = "YLQPRTFLL"


# ---------------------------------------------------------------------------
# Reproducibility
#
# Stages 1, 2, 4, 5, 6 and 7 contain no randomness: they are deterministic arithmetic and table
# joins, so the published numbers do not depend on a seed (docs/REPRODUCIBILITY.md shows the
# evidence). set_seeds() is still called at the top of every stage so that anything added later -
# a sampled control, a bootstrap, a jittered scatter - is repeatable by default rather than by luck.
SEED = 42


def set_seeds(seed: int = SEED) -> int:
    """Seed every generator this pipeline could reach. Returns the seed for logging."""
    import random as _random
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:  # only present in the model-inference environments
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
            # bitwise-identical GPU results also need deterministic kernels
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def canon(name: str) -> str:
    if name not in CANON:
        die(f"unknown model label {name!r} - add it to CANON in stages/common.py")
    return CANON[name]


def die(msg: str, code: int = 1):
    """Stop loudly. No stage continues past a problem it cannot explain."""
    print(f"\nERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def need(path: Path, what: str, how: str = "") -> Path:
    if not Path(path).exists():
        die(f"missing {what}: {path}" + (f"\n  {how}" if how else ""))
    return Path(path)


def banner(stage: str, text: str):
    print(f"\n=== {stage}: {text}")


def write_provenance(target: Path, lines: list[str]):
    """Sidecar describing how an artifact was made and what (if anything) it drops."""
    p = Path(target).with_suffix(Path(target).suffix + ".provenance.txt")
    p.write_text("\n".join(lines) + "\n")
    return p
