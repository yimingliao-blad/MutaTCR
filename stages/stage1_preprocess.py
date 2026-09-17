#!/usr/bin/env python3
"""Stage 1 - raw data -> unified benchmark tables.

Reads only the files under data/raw/ and writes one unified CSV per dataset. The committed copies in
data/unified/ were produced by exactly this step; by default we write to build/unified/ and compare,
so a re-run proves reproducibility instead of overwriting the shipped data.

  python3 stages/stage1_preprocess.py                 # rebuild into build/unified + compare
  python3 stages/stage1_preprocess.py --write-canonical   # also refresh data/unified/
"""
import argparse
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from stages.common import (B_UNIFIED, DATASETS, RAW, REPO, UNIFIED, banner, die, need, set_seeds)  # noqa: E402


def main():
    set_seeds()   # see stages/common.py: the chain is deterministic; this keeps it that way
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-canonical", action="store_true",
                    help="copy the rebuilt tables over data/unified/ (only when you mean to)")
    args = ap.parse_args()

    need(RAW / "FingerPrinting" / "Fingerprinting_TCR_Data_Combined_final.xlsx", "FingerPrinting raw workbook")
    need(RAW / "IMMREP23" / "immref23.csv", "IMMREP23 raw table")
    need(RAW / "TetTCR-SeqHD" / "TCR_antigen_binding_sheet.csv", "TetTCR binding sheet")

    B_UNIFIED.mkdir(parents=True, exist_ok=True)
    banner("stage 1", f"raw -> {B_UNIFIED.relative_to(REPO)}")

    from src.data_processing.preprocess_raw import RawDataPreprocessor
    from src.utils.paths import PathManager

    paths = PathManager(base_path=REPO)
    paths.RAW_DATA_DIR = RAW
    paths.TETTCR_RAW_DIR = RAW / "TetTCR-SeqHD"
    paths.IMMREP23_RAW_DIR = RAW / "IMMREP23"
    paths.FINGERPRINTING_RAW_DIR = RAW / "FingerPrinting"
    paths.UNIFIED_DATA_DIR = B_UNIFIED
    # In the original tree this one file sat outside the project; here it lives with the raw data.
    paths.get_fingerprinting_log2fc_file = lambda: RAW / "FingerPrinting" / "finger_print.xlsx"

    pre = RawDataPreprocessor()
    pre.paths = paths  # the preprocessor builds its own PathManager; point it at this repo
    pre.process_all()

    ok = True
    for ds in DATASETS:
        new = B_UNIFIED / f"{ds}_unified.csv"
        ref = UNIFIED / f"{ds}_unified.csv"
        if not new.exists():
            die(f"stage 1 did not produce {new}")
        a = pd.read_csv(new, dtype=str, keep_default_na=False)
        b = pd.read_csv(ref, dtype=str, keep_default_na=False)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            print(f"  {ds}: DIFFERS in shape/columns  rebuilt={a.shape} committed={b.shape}")
            ok = False
            continue
        diff_cols = [c for c in a.columns if not (a[c].values == b[c].values).all()]
        print(f"  {ds}: {len(a)} rows, "
              + ("identical to the committed table" if not diff_cols
                 else f"differs in columns {diff_cols}"))
        ok &= not diff_cols
        if args.write_canonical:
            shutil.copy2(new, ref)
    print("\nstage 1 " + ("reproduced the committed unified tables" if ok
                          else "produced tables that DIFFER from the committed ones (see above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
