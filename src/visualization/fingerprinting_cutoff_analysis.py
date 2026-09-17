"""
Fingerprinting Log2 Cutoff Sensitivity Analysis.

Generates visualizations showing how model AUC and pAUC change across
different log2foldchange cutoff thresholds for defining positive/negative labels.

Output:
- fingerprinting_cutoff_auc.png: 8 subplots showing AUC vs cutoff
- fingerprinting_cutoff_pauc.png: 8 subplots showing pAUC vs cutoff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name

logger = logging.getLogger(__name__)

# Cutoff values for log2foldchange threshold
CUTOFF_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Model configuration
MODEL_ORDER_LOCAL = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']
MODEL_DISPLAY = {
    'ERGO': 'ERGO', 'ERGO2': 'ERGO2', 'NetTCR': 'NetTCR2',
    'NetTCR22': 'NetTCR-2.2', 'TITAN': 'TITAN', 'EPACT': 'EPACT',
    'PanPep': 'PanPep', 'SCEPTR': 'SCEPTR'
}


def calculate_metrics_at_cutoff(
    df: pd.DataFrame,
    prob_col: str,
    cutoff: float,
    max_fpr: float = 0.1
) -> pd.DataFrame:
    """
    Calculate AUC and pAUC for each TCR at a given cutoff threshold.

    Args:
        df: DataFrame with predictions and log2foldchange
        prob_col: Prediction probability column name
        cutoff: Log2foldchange threshold for positive/negative labels
        max_fpr: Maximum FPR for partial AUC calculation

    Returns:
        DataFrame with columns: TCR_Group, AUC, pAUC
    """
    results = []

    for tcr_group in df['TCR_Group'].unique():
        tcr_df = df[df['TCR_Group'] == tcr_group].copy()

        if len(tcr_df) < 2:
            continue

        # Reassign labels based on cutoff
        y_true = (tcr_df['log2foldchange'] > cutoff).astype(int).values
        y_score = tcr_df[prob_col].values

        # Remove NaN predictions
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]

        # Need both classes and sufficient samples
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_score)
            pauc = roc_auc_score(y_true, y_score, max_fpr=max_fpr)

            results.append({
                'TCR_Group': tcr_group,
                'AUC': auc,
                'pAUC': pauc
            })
        except Exception as e:
            logger.debug(f"Error calculating metrics for {tcr_group} at cutoff {cutoff}: {e}")
            continue

    return pd.DataFrame(results)


def create_cutoff_analysis_plot(mode: str, metric: str = 'auc'):
    """
    Create 8-subplot figure with cutoffs on x-axis, metric on y-axis.

    Args:
        mode: Deduplication mode ("exact_match" or "fuzzy_match")
        metric: 'auc' or 'pauc'
    """
    paths = get_paths(mode=mode)

    # Load merged fingerprinting data
    merged_file = paths.MERGED_DIR / 'fingerprinting_all_models.csv'
    if not merged_file.exists():
        logger.warning(f"Merged file not found: {merged_file}")
        return

    df = pd.read_csv(merged_file)
    logger.info(f"Loaded {len(df)} samples from fingerprinting data")

    # Filter to only 'valid' Antigen_Status entries
    if 'Antigen_Status' in df.columns:
        df = df[df['Antigen_Status'] == 'valid']
        logger.info(f"Filtered to {len(df)} valid entries")

    # Verify required columns
    if 'log2foldchange' not in df.columns or 'TCR_Group' not in df.columns:
        logger.error("Required columns not found in data")
        return

    # Setup plot
    setup_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Color palette for box plots
    box_color = '#3498db'  # Blue

    # Process each model
    for model_idx, model in enumerate(MODEL_ORDER_LOCAL):
        ax = axes[model_idx]
        prob_col = f'{model}_Prob'
        display_name = MODEL_DISPLAY.get(model, model)

        if prob_col not in df.columns:
            ax.text(0.5, 0.5, f'{display_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # Calculate metrics for each cutoff
        all_data = []
        positions = []
        means = []

        for cutoff_idx, cutoff in enumerate(CUTOFF_VALUES):
            metrics_df = calculate_metrics_at_cutoff(df, prob_col, cutoff)

            if len(metrics_df) > 0:
                metric_col = 'AUC' if metric.lower() == 'auc' else 'pAUC'
                values = metrics_df[metric_col].values
                all_data.append(values)
                positions.append(cutoff_idx)
                means.append(np.mean(values))
            else:
                all_data.append([])
                positions.append(cutoff_idx)
                means.append(np.nan)

        # Create box plots
        valid_data = [d for d in all_data if len(d) > 0]
        valid_positions = [p for i, p in enumerate(positions) if len(all_data[i]) > 0]

        if len(valid_data) > 0:
            bp = ax.boxplot(valid_data, positions=valid_positions, widths=0.6,
                           patch_artist=True, showfliers=False)

            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
                patch.set_alpha(0.6)
            for whisker in bp['whiskers']:
                whisker.set_color('gray')
            for cap in bp['caps']:
                cap.set_color('gray')
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            # Add scatter points with jitter
            np.random.seed(42)
            for i, (data, pos) in enumerate(zip(valid_data, valid_positions)):
                if len(data) > 0:
                    jitter = np.random.uniform(-0.15, 0.15, len(data))
                    ax.scatter(pos + jitter, data, c=box_color, alpha=0.4, s=15,
                              edgecolor='white', linewidth=0.3, zorder=3)

            # Add mean markers (diamonds)
            valid_means = [means[p] for p in valid_positions]
            ax.scatter(valid_positions, valid_means, marker='D', color='white',
                      edgecolor='black', s=40, zorder=5, linewidths=1)

        # Random baseline
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Formatting
        ax.set_xticks(range(len(CUTOFF_VALUES)))
        ax.set_xticklabels([str(c) for c in CUTOFF_VALUES], fontsize=8, rotation=45)
        ax.set_xlabel('Log2FC Cutoff', fontsize=9)

        metric_label = 'AUC' if metric.lower() == 'auc' else 'pAUC$_{0.1}$'
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.set_ylim(0.3, 1.05)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Main title
    metric_name = 'AUC' if metric.lower() == 'auc' else 'Partial AUC (FPR ≤ 0.1)'
    fig.suptitle(f'FingerPrinting: {metric_name} vs Log2 Fold Change Cutoff',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    metric_suffix = 'auc' if metric.lower() == 'auc' else 'pauc'
    output_path = paths.get_visualization_file(f'fingerprinting_cutoff_{metric_suffix}.png')
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_cutoff_analysis(mode: str):
    """
    Generate both AUC and pAUC cutoff analysis plots.

    Args:
        mode: Deduplication mode
    """
    logger.info(f"Generating cutoff analysis plots for {mode}...")

    # Generate AUC plot
    create_cutoff_analysis_plot(mode, metric='auc')

    # Generate pAUC plot
    create_cutoff_analysis_plot(mode, metric='pauc')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting cutoff analysis")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match"],
                        default="exact_match", help="Deduplication mode")
    parser.add_argument("--metric", choices=["auc", "pauc", "both"],
                        default="both", help="Metric to plot")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_cutoff_analysis", level="INFO")

    if args.metric == "both":
        generate_cutoff_analysis(args.mode)
    else:
        create_cutoff_analysis_plot(args.mode, args.metric)


if __name__ == "__main__":
    main()
