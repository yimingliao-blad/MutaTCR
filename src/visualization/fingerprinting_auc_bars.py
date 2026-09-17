"""
Fingerprinting AUC Bar Chart Visualizations for TCRP Benchmark V4.

Generates bar charts for fingerprinting dataset:
1. Overall AUC
2. Partial AUC (0.1)
3. Mean per-peptide Partial AUC (0.1)

All charts exclude wild type and unstable peptides (Antigen_Status == 'valid' only).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import argparse
import logging
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name, get_model_colors

logger = logging.getLogger(__name__)

MODELS = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']


def load_fingerprinting_predictions(paths) -> Optional[pd.DataFrame]:
    """Load fingerprinting predictions from all 8 models."""
    base_df = None
    output_dir = paths.MODE_OUTPUT_DIR

    for model in MODELS:
        pred_file = output_dir / model / "predictions" / "fingerprinting_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)

            if base_df is None:
                cols_to_keep = ['ID', 'Peptide', 'CDR3b', 'TCR_Group', 'Label', 'log2foldchange']
                cols_available = [c for c in cols_to_keep if c in df.columns]
                base_df = df[cols_available].copy()
                base_df[f'{model}_Prob'] = df['Prediction_Prob'].values
            else:
                model_df = df[['ID', 'Prediction_Prob']].rename(columns={'Prediction_Prob': f'{model}_Prob'})
                base_df = base_df.merge(model_df, on='ID', how='inner')
        else:
            logger.warning(f"Prediction file not found: {pred_file}")

    if base_df is not None:
        # Load Antigen_Status from unified data
        unified_file = paths.UNIFIED_DATA_DIR / "fingerprinting_unified.csv"
        if unified_file.exists():
            unified_df = pd.read_csv(unified_file)
            if 'Antigen_Status' in unified_df.columns:
                # Create mapping from Peptide to Antigen_Status
                status_map = unified_df[['Peptide', 'Antigen_Status']].drop_duplicates().set_index('Peptide')['Antigen_Status'].to_dict()
                base_df['Antigen_Status'] = base_df['Peptide'].map(status_map)

        # Filter to only 'valid' Antigen_Status entries
        if 'Antigen_Status' in base_df.columns:
            original_len = len(base_df)
            base_df = base_df[base_df['Antigen_Status'] == 'valid']
            logger.info(f"Filtered from {original_len} to {len(base_df)} valid entries")

    return base_df


def calculate_overall_auc(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall ROC AUC for each model."""
    results = {}
    y_true = df['Label'].values

    for model in MODELS:
        prob_col = f'{model}_Prob'
        if prob_col in df.columns:
            y_score = df[prob_col].values
            mask = ~np.isnan(y_score)
            if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
                try:
                    auc = roc_auc_score(y_true[mask], y_score[mask])
                    results[model] = auc
                except Exception as e:
                    logger.warning(f"Error calculating AUC for {model}: {e}")
                    results[model] = 0.5
            else:
                results[model] = 0.5
    return results


def calculate_partial_auc(df: pd.DataFrame, max_fpr: float = 0.1) -> Dict[str, float]:
    """Calculate partial ROC AUC (at max_fpr) for each model."""
    results = {}
    y_true = df['Label'].values

    for model in MODELS:
        prob_col = f'{model}_Prob'
        if prob_col in df.columns:
            y_score = df[prob_col].values
            mask = ~np.isnan(y_score)
            if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
                try:
                    pauc = roc_auc_score(y_true[mask], y_score[mask], max_fpr=max_fpr)
                    results[model] = pauc
                except Exception as e:
                    logger.warning(f"Error calculating pAUC for {model}: {e}")
                    results[model] = 0.5
            else:
                results[model] = 0.5
    return results


def calculate_mean_per_peptide_pauc(df: pd.DataFrame, max_fpr: float = 0.1, min_samples: int = 5) -> Dict[str, float]:
    """Calculate mean per-peptide partial AUC for each model."""
    results = {}

    for model in MODELS:
        prob_col = f'{model}_Prob'
        if prob_col not in df.columns:
            continue

        peptide_paucs = []
        for peptide in df['Peptide'].unique():
            peptide_df = df[df['Peptide'] == peptide]
            y_true = peptide_df['Label'].values
            y_score = peptide_df[prob_col].values

            # Remove NaN
            mask = ~np.isnan(y_score)
            y_true_clean = y_true[mask]
            y_score_clean = y_score[mask]

            # Need sufficient samples with both classes
            n_pos = (y_true_clean == 1).sum()
            n_neg = (y_true_clean == 0).sum()

            if len(y_true_clean) < min_samples or n_pos == 0 or n_neg == 0:
                continue

            try:
                pauc = roc_auc_score(y_true_clean, y_score_clean, max_fpr=max_fpr)
                peptide_paucs.append(pauc)
            except:
                pass

        if peptide_paucs:
            results[model] = np.mean(peptide_paucs)
        else:
            results[model] = 0.5

    return results


def create_auc_bar_chart(
    auc_values: Dict[str, float],
    ylabel: str,
    output_path: Path,
    baseline: float = 0.5
):
    """
    Create a bar chart for AUC metrics.

    Args:
        auc_values: Dictionary mapping model names to AUC values
        ylabel: Y-axis label
        output_path: Output file path
        baseline: Random baseline value (default 0.5)
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    setup_plot_style()

    # Prepare data in MODEL_ORDER
    models = []
    values = []
    display_names = []

    for model in MODEL_ORDER:
        if model in auc_values:
            models.append(model)
            values.append(auc_values[model])
            display_names.append(format_model_name(model))

    if not models:
        logger.warning(f"No data for bar chart")
        return

    # Create figure - narrower since only one bar per group
    fig, ax = plt.subplots(figsize=(8, 6))

    # All bars in red
    bar_color = '#d62728'  # Red

    # Create bars with appropriate width (similar to boxplot box width)
    x = np.arange(len(models))
    bar_width = 0.5
    bars = ax.bar(x, values, width=bar_width, color=bar_color, edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add random baseline
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Fingerprinting Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend (similar to combined_seen_pauc_boxplot)
    legend_elements = [
        Patch(facecolor=bar_color, alpha=0.8, edgecolor='black', label='FingerPrinting'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Random Baseline')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_fingerprinting_auc_bars(mode: str = "exact_match"):
    """
    Generate fingerprinting AUC bar charts.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)

    logger.info(f"Generating fingerprinting AUC bar charts for {mode}...")

    # Load fingerprinting predictions
    df = load_fingerprinting_predictions(paths)
    if df is None:
        logger.warning("Could not load fingerprinting predictions")
        return

    logger.info(f"Loaded {len(df)} valid fingerprinting samples")
    logger.info(f"Unique peptides: {df['Peptide'].nunique()}")
    logger.info(f"Positive samples: {(df['Label'] == 1).sum()}, Negative samples: {(df['Label'] == 0).sum()}")

    # 1. Overall AUC bar chart
    logger.info("Calculating overall AUC...")
    overall_auc = calculate_overall_auc(df)
    logger.info(f"Overall AUC values: {overall_auc}")

    output_path = paths.get_visualization_file('fingerprinting_overall_auc_bars.png')
    create_auc_bar_chart(
        overall_auc,
        'ROC AUC',
        output_path
    )

    # 2. Partial AUC (0.1) bar chart
    logger.info("Calculating partial AUC (0.1)...")
    partial_auc = calculate_partial_auc(df, max_fpr=0.1)
    logger.info(f"Partial AUC values: {partial_auc}")

    output_path = paths.get_visualization_file('fingerprinting_partial_auc_bars.png')
    create_auc_bar_chart(
        partial_auc,
        'Partial AUC$_{0.1}$',
        output_path
    )

    # 3. Mean per-peptide Partial AUC bar chart
    logger.info("Calculating mean per-peptide partial AUC...")
    mean_per_peptide_pauc = calculate_mean_per_peptide_pauc(df, max_fpr=0.1)
    logger.info(f"Mean per-peptide pAUC values: {mean_per_peptide_pauc}")

    output_path = paths.get_visualization_file('fingerprinting_mean_peptide_pauc_bars.png')
    create_auc_bar_chart(
        mean_per_peptide_pauc,
        'Mean Per-Peptide pAUC$_{0.1}$',
        output_path
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting AUC bar charts")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_auc_bars", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_fingerprinting_auc_bars(mode)


if __name__ == "__main__":
    main()
