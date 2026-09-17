#!/usr/bin/env python3
"""Stage 2 - training/benchmark de-duplication.

Each model was screened against its own training data and the overlapping benchmark rows were
removed before scoring. Recomputing that needs every model's training set (~10 GB, not in this
repo), so the *result* is committed as data/dedup/removed_ids.csv (dataset, model, removed_id) and
this stage materialises the per-model tables from unified + that list. Nothing is lost: the removed
rows stay in the unified tables, and each materialised file records what it dropped.

  python3 stages/stage2_dedup.py              # materialise build/dedup/ from the committed list
  python3 stages/stage2_dedup.py --recompute  # re-screen against the models' training data
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import (B_DEDUP, DATASETS, DEDUP_IDS, MODELS, UNIFIED, banner, die,  # noqa: E402
                           need, write_provenance, set_seeds)


def main():
    set_seeds()   # see stages/common.py: the chain is deterministic; this keeps it that way
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="re-screen against each model's training data (requires the model repos)")
    args = ap.parse_args()

    if args.recompute:
        from src.utils.config import get_config
        cfg = get_config()
        missing = [m for m in MODELS
                   if (p := cfg.get_model_training_data_path(m)) is None or not Path(p).exists()]
        die("--recompute needs each model's training data, which is not part of this repo.\n"
            f"  missing for: {', '.join(missing)}\n"
            "  Point config/benchmark_config.yaml at a checkout of the model repos, or drop\n"
            "  --recompute to materialise from the committed data/dedup/removed_ids.csv.")

    need(DEDUP_IDS, "removed-id list")
    removed = pd.read_csv(DEDUP_IDS, dtype=str)
    banner("stage 2", f"materialise de-duplicated tables -> build/dedup ({len(removed)} removed rows recorded)")

    for ds in DATASETS:
        u = pd.read_csv(need(UNIFIED / f"{ds}_unified.csv", f"unified {ds}"), dtype=str, keep_default_na=False)
        for m in MODELS:
            drop = set(removed[(removed.dataset == ds) & (removed.model == m)].removed_id)
            unknown = drop - set(u.ID)
            if unknown:
                die(f"{ds}/{m}: {len(unknown)} removed IDs are not in the unified table")
            out_dir = B_DEDUP / m
            out_dir.mkdir(parents=True, exist_ok=True)
            kept = u[~u.ID.isin(drop)]
            out = out_dir / f"{ds}_deduplicated.csv"
            kept.to_csv(out, index=False)
            write_provenance(out, [
                f"stage 2: {ds} de-duplicated for {m}",
                f"source: data/unified/{ds}_unified.csv ({len(u)} rows)",
                f"removed: {len(drop)} rows whose (CDR3b, Peptide) appear in this model's training data",
                f"kept: {len(kept)} rows",
                "removed IDs: " + (", ".join(sorted(drop)) if drop else "(none)"),
            ])
        n = len(removed[(removed.dataset == ds)])
        print(f"  {ds}: {len(u)} unified rows -> 8 model tables, {n} model-specific removals in total")
    print("\nstage 2 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
