#!/usr/bin/env python3
"""Stage 3 - model inference (the one stage this repo cannot run on its own).

The eight predictors need ~10 GB of published weights and eight conda environments, so their
outputs are committed instead: data/scores/<dataset>_model_scores.csv holds one probability column
per model, keyed by row ID, with an empty cell where de-duplication removed that row for that model.
Everything downstream is rebuilt from those score tables, so the whole manuscript regenerates
offline. This stage exists to say so out loud, and to run inference when the models are available.

  python3 stages/stage3_inference.py --check     # report what is committed (default)
  python3 stages/stage3_inference.py --run       # run the models (needs weights + environments)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import DATASETS, MODELS, SCORES, banner, die, need  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually run the models")
    ap.add_argument("--check", action="store_true", help="report the committed scores (default)")
    args = ap.parse_args()

    if args.run:
        from src.utils.config import get_config
        cfg = get_config()
        conda = cfg.get_conda_path()
        models_dir = cfg.get_models_dir()
        die("--run needs the published model weights and their conda environments, which are not\n"
            "  part of this repo (~10 GB).\n"
            f"  expected models at: {models_dir}\n"
            f"  expected conda at:  {conda}\n"
            "  See docs/INFERENCE.md. The committed data/scores/*.csv are the outputs of this stage.")

    banner("stage 3", "committed model scores (inference itself needs the model weights)")
    for ds in DATASETS:
        f = need(SCORES / f"{ds}_model_scores.csv", f"score table for {ds}")
        s = pd.read_csv(f)
        cols = [f"{m}_Prob" for m in MODELS]
        if [c for c in cols if c not in s.columns]:
            die(f"{f}: missing columns {[c for c in cols if c not in s.columns]}")
        miss = {m: int(s[f"{m}_Prob"].isna().sum()) for m in MODELS}
        print(f"  {ds}: {len(s)} rows x {len(MODELS)} models; "
              f"rows without a score: { {k: v for k, v in miss.items() if v} or 'none'}")
    print("\nstage 3 ok (nothing to run; scores are committed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
