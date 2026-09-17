#!/usr/bin/env python3
"""Stage 8 - supplementary visualisations (the exploratory figure suite).

These are the matplotlib figures behind the epitope- and TCR-level analyses: per-TCR heatmaps,
per-model prediction heatmaps, correlation panels, the log2FC scatter, and the multi-panel compound
figure used in the earlier preprint. They read the merged tables from stage 4 and write PNG/SVG into
build/supplementary/, which is not committed - the manuscript does not depend on them, and they are
a few hundred MB when generated in full.

  python3 stages/stage8_supplementary_figures.py --list
  python3 stages/stage8_supplementary_figures.py --only compound_figure
  python3 stages/stage8_supplementary_figures.py            # every figure below
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stages.common import B_MERGED, BUILD, REPO, banner, die, need, set_seeds  # noqa: E402

OUT = BUILD / "supplementary"
ASSETS = REPO / "assets"

# name -> (module, callable) in src/visualization
FIGURES = {
    "compound_figure": ("src.visualization.compound_figure", "generate_compound_figure"),
    "auc_boxplots": ("src.visualization.auc_boxplot", "generate_all_boxplots"),
    "fingerprinting_heatmaps": ("src.visualization.fingerprinting_heatmap", "generate_all_heatmaps"),
    "fingerprinting_dual_heatmaps": ("src.visualization.fingerprinting_dual_heatmap", "generate_all_dual_heatmaps"),
    "tcr_model_heatmaps": ("src.visualization.tcr_model_heatmaps", "generate_tcr_model_heatmaps"),
    "tcr_correlation": ("src.visualization.fingerprinting_tcr_correlation", "generate_tcr_correlation_visualizations"),
    "model_correlation": ("src.visualization.fingerprinting_correlation", "generate_fingerprinting_correlation_plots"),
    "position_correlation": ("src.visualization.fingerprinting_position_correlation", "create_position_correlation_plot"),
    "auc_bars": ("src.visualization.fingerprinting_auc_bars", "generate_fingerprinting_auc_bars"),
    "log2fc_scatter": ("src.visualization.log2fc_scatter", "generate_log2fc_plots"),
    "cutoff_analysis": ("src.visualization.fingerprinting_cutoff_analysis", "generate_cutoff_analysis"),
}


def setup_paths():
    """Point the v4 visualisation modules at this repo's build tree."""
    from src.utils import paths as paths_mod
    pm = paths_mod.PathManager(base_path=REPO, mode="exact_match")
    pm.MERGED_DIR = B_MERGED
    pm.OUTPUT_DIR = BUILD
    pm.MODE_OUTPUT_DIR = BUILD
    pm.PREDICTIONS_DIR = BUILD / "predictions"
    pm.AGGREGATED_DIR = BUILD / "aggregated"
    pm.VISUALIZATIONS_DIR = OUT            # the attribute the visualisation modules actually read
    pm.HEATMAPS_DIR = OUT / "fingerprinting_heatmaps"
    OUT.mkdir(parents=True, exist_ok=True)
    # panel (a) of the compound figure and the pipeline overview are drawings, not generated
    for name in ("Refined Flow.svg", "new_process.png"):
        src = ASSETS / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    pm.get_visualization_file = lambda name: OUT / name
    pm.get_merged_file = lambda dataset: B_MERGED / f"{dataset}_all_models.csv"
    paths_mod._path_manager = pm          # the module-level singleton get_paths() returns
    return pm


def main():
    set_seeds()   # see stages/common.py: the chain is deterministic; this keeps it that way
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=sorted(FIGURES), help="run just these")
    ap.add_argument("--list", action="store_true", help="list the available figures and exit")
    a = ap.parse_args()
    if a.list:
        for k, (mod, fn) in sorted(FIGURES.items()):
            print(f"  {k:32s} {mod}.{fn}")
        return 0

    need(B_MERGED / "fingerprinting_all_models.csv", "merged table",
         "run stages/stage4_assemble.py first")
    setup_paths()
    banner("stage 8", f"supplementary visualisations -> {OUT.relative_to(REPO)}")
    import importlib
    wanted = a.only or list(FIGURES)
    failed = []
    for name in wanted:
        mod_name, fn_name = FIGURES[name]
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            fn("exact_match")
            print(f"  {name}: ok")
        except Exception as e:  # report and keep going; this stage is supplementary
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name}: FAILED - {type(e).__name__}: {e}")
    made = len(list(OUT.rglob("*.png"))) + len(list(OUT.rglob("*.svg")))
    print(f"\n{made} image(s) in {OUT.relative_to(REPO)}")
    if failed:
        print("\nfailed figures (these are supplementary; the manuscript does not use them):")
        for n, why in failed:
            print(f"  {n}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
