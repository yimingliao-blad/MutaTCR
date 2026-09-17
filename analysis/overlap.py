#!/usr/bin/env python3
"""
Count test/training overlap for the 3 TCRP source datasets against 8 models.

Replicates the extraction logic of
  src/data_processing/deduplicate.py :: TrainingDataDeduplicator
exactly (same training files, same column candidates, same .upper() handling),
then reports overlap under two criteria:

  (a) peptide only        -- test Peptide appears anywhere in training
  (b) CDR3b + peptide     -- the exact pair appears in training  [= shipped exact_match]

Reports both rows affected and unique keys hit.
Criterion (b) is cross-checked against the stored *_deduplicated.csv files.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

# Paths resolve inside this repository. Recomputing the overlap needs each model's TRAINING data
# (~10 GB of published model repos), which is not part of the repo: set TCRJ_MODELS to a checkout,
# or read the committed result at data/overlap/overlap_summary.csv. See docs/INFERENCE.md.
import os

REPO = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("TCRJ_MODELS", REPO / "models"))
UNIFIED = REPO / "data" / "unified"
DEDUP = Path(os.environ.get("TCRJ_DEDUP", REPO / "build" / "dedup"))

DATASETS = ["fingerprinting", "immrep23", "tettcr"]
MODEL_ORDER = ["ERGO", "ERGO2", "NetTCR", "NetTCR22", "TITAN", "EPACT", "PanPep", "SCEPTR"]

# deduplicate.py lines 83-92
TRAINING_FILES = {
    "ERGO":     MODELS / "ERGO" / "data" / "VDJDB_complete.tsv",
    "ERGO2":    MODELS / "ERGO2" / "Samples" / "mcpas_train_samples.pickle",
    "NetTCR":   MODELS / "NetTCR" / "data" / "train_ab_95_alphabeta.csv",
    "NetTCR22": MODELS / "NetTCR22" / "data" / "nettcr_2_2_full_dataset.csv",
    "TITAN":    MODELS / "TITAN" / "datasets" / "full_data+covid.csv",
    "EPACT":    MODELS / "EPACT" / "sample" / "VDJdb-GLCTLVAML.csv",
    "PanPep":   MODELS / "PanPep" / "Data" / "majority_training_dataset.csv",
    # SCEPTR: deduplicate.py lines 163-173 hardcode empty sets
}

# deduplicate.py lines 133-134
CDR3B_COLS = ["CDR3b", "cdr3b", "CDR3.beta", "CDR3.beta.aa", "TRB_cdr3",
              "CDR3", "beta", "tcrb", "B3", "binding_TCR"]
PEPTIDE_COLS = ["Peptide", "peptide", "Epitope", "epitope",
                "Epitope.peptide", "antigen.epitope"]


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_training_file(path):
    if path.suffix == ".pickle":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    raise ValueError(f"unknown format: {path.suffix}")


def extract_training_sets(model, path):
    """Return (peptide_set, tuple_set, n_train_rows, cdr3b_col, peptide_col)."""
    df = load_training_file(path)
    cdr3b_col = find_column(df, CDR3B_COLS)
    peptide_col = find_column(df, PEPTIDE_COLS)

    peptides, tuples_ = set(), set()
    for _, row in df.iterrows():
        cdr3b = str(row.get(cdr3b_col, "")).upper() if cdr3b_col else ""
        pep = str(row.get(peptide_col, "")).upper() if peptide_col else ""
        if pep and pep != "NAN":
            peptides.add(pep)
        if cdr3b and cdr3b != "NAN" and pep and pep != "NAN":
            tuples_.add((cdr3b, pep))
    return peptides, tuples_, len(df), cdr3b_col, peptide_col


def main():
    print("=" * 100)
    print("LOADING TRAINING DATA")
    print("=" * 100)
    train = {}
    for model in MODEL_ORDER:
        if model not in TRAINING_FILES:
            train[model] = (set(), set(), 0, None, None)
            print(f"{model:9s}  HARDCODED EMPTY (deduplicate.py:171-172)")
            continue
        path = TRAINING_FILES[model]
        peps, tups, n, ccol, pcol = extract_training_sets(model, path)
        train[model] = (peps, tups, n, ccol, pcol)
        print(f"{model:9s}  rows={n:>7,}  uniq_peptide={len(peps):>6,}  "
              f"uniq_(cdr3b,pep)={len(tups):>7,}  cols=({ccol}, {pcol})")

    rows_out = []
    for ds in DATASETS:
        df = pd.read_csv(UNIFIED / f"{ds}_unified.csv", low_memory=False)
        n_total = len(df)
        pep = df["Peptide"].astype(str).str.upper()
        cdr = df["CDR3b"].astype(str).str.upper()
        pairs = list(zip(cdr, pep))

        print()
        print("=" * 100)
        print(f"{ds.upper()}   rows={n_total:,}   "
              f"uniq_peptide={pep.nunique():,}   uniq_(cdr3b,pep)={len(set(pairs)):,}")
        print("=" * 100)

        for model in MODEL_ORDER:
            tr_peps, tr_tups, _, _, _ = train[model]

            # (a) peptide only
            m_pep = pep.isin(tr_peps)
            a_rows = int(m_pep.sum())
            a_uniq = pep[m_pep].nunique()

            # (b) cdr3b + peptide  (= shipped exact_match rule)
            m_pair = pd.Series([p in tr_tups for p in pairs], index=df.index)
            b_rows = int(m_pair.sum())
            b_uniq = len({p for p, hit in zip(pairs, m_pair) if hit})

            # cross-check (b) against the stored dedup output
            stored = DEDUP / model / f"{ds}_deduplicated.csv"
            check = "n/a"
            if stored.exists():
                sdf = pd.read_csv(stored, low_memory=False)
                if ds == "tettcr":
                    # negatives are subsampled after dedup -> compare ID sets, not counts
                    kept = set(sdf["ID"])
                    survivors_that_should_be_gone = sum(
                        1 for i, hit in zip(df["ID"], m_pair) if hit and i in kept
                    )
                    check = "OK" if survivors_that_should_be_gone == 0 else \
                            f"{survivors_that_should_be_gone} leaked"
                else:
                    expected = n_total - b_rows
                    check = "OK" if len(sdf) == expected else f"got {len(sdf)}, exp {expected}"

            rows_out.append(dict(
                dataset=ds, model=model, total_rows=n_total,
                pep_rows=a_rows, pep_pct=100 * a_rows / n_total, pep_uniq=a_uniq,
                pair_rows=b_rows, pair_pct=100 * b_rows / n_total, pair_uniq=b_uniq,
                verify=check,
            ))
            print(f"  {model:9s} | peptide-only {a_rows:>6,} ({100*a_rows/n_total:5.1f}%) "
                  f"uniq {a_uniq:>4} | cdr3b+pep {b_rows:>6,} ({100*b_rows/n_total:5.1f}%) "
                  f"uniq {b_uniq:>5,} | dedup check: {check}")

    out = pd.DataFrame(rows_out)
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("overlap_summary.csv")
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
