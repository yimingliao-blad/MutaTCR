"""
Fingerprinting pAUC Boxplot Visualization.

Generates per-TCR partial AUC (0.1) box plots for fingerprinting data,
separated by R5-APL (position 5 substitution) and Non-5-Pos (other positions).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import argparse
import logging
from sklearn.metrics import roc_auc_score
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name

logger = logging.getLogger(__name__)


def calculate_per_tcr_pauc(df: pd.DataFrame, prob_col: str, max_fpr: float = 0.1) -> pd.DataFrame:
    """
    Calculate per-TCR partial AUC from fingerprinting data.

    Args:
        df: DataFrame with predictions and labels
        prob_col: Prediction probability column name
        max_fpr: Maximum FPR for partial AUC

    Returns:
        DataFrame with TCR_Group, Epitope_Group, and pAUC columns
    """
    results = []

    for tcr_group in df['TCR_Group'].unique():
        for epitope_group in df['Epitope_Group'].unique():
            subset = df[(df['TCR_Group'] == tcr_group) & (df['Epitope_Group'] == epitope_group)]

            if len(subset) < 2:
                continue

            y_true = subset['Label'].values
            y_score = subset[prob_col].values

            # Remove NaN
            mask = ~np.isnan(y_score)
            y_true = y_true[mask]
            y_score = y_score[mask]

            if len(np.unique(y_true)) < 2 or len(y_true) < 2:
                continue

            try:
                pauc = roc_auc_score(y_true, y_score, max_fpr=max_fpr)
                results.append({
                    'TCR_Group': tcr_group,
                    'Epitope_Group': epitope_group,
                    'pAUC': pauc
                })
            except:
                pass

    return pd.DataFrame(results)


def create_fingerprinting_pauc_boxplot(mode: str):
    """
    Create fingerprinting pAUC boxplot separated by R5-APL and Non-5-Pos groups.

    Args:
        mode: Deduplication mode ("exact_match" or "fuzzy_match")
    """
    paths = get_paths(mode=mode)
    config = get_config()

    # Model configuration
    MODEL_ORDER_LOCAL = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']
    MODEL_DISPLAY = {
        'ERGO': 'ERGO', 'ERGO2': 'ERGO2', 'NetTCR': 'NetTCR2',
        'NetTCR22': 'NetTCR-2.2', 'TITAN': 'TITAN', 'EPACT': 'EPACT',
        'PanPep': 'PanPep', 'SCEPTR': 'SCEPTR'
    }

    # Display names from config
    GROUP_DISPLAY = {
        'R-5': config.get_epitope_group_display_name('fingerprinting', 'R-5'),     # R5-APL
        'non-R-5': config.get_epitope_group_display_name('fingerprinting', 'non-R-5')  # Non-5-Pos
    }

    # Colors from config
    COLORS = {
        GROUP_DISPLAY['R-5']: config.get_group_color('fingerprinting', 'R5-APL'),
        GROUP_DISPLAY['non-R-5']: config.get_group_color('fingerprinting', 'Non-5-Pos')
    }

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

    # Calculate per-TCR pAUC for each model and group
    plot_data = []

    for model in MODEL_ORDER_LOCAL:
        prob_col = f'{model}_Prob'
        display_name = MODEL_DISPLAY.get(model, model)

        if prob_col not in df.columns:
            continue

        pauc_df = calculate_per_tcr_pauc(df, prob_col)

        for _, row in pauc_df.iterrows():
            group_display = GROUP_DISPLAY.get(row['Epitope_Group'], row['Epitope_Group'])
            plot_data.append({
                'Model': display_name,
                'Group': group_display,
                'pAUC': row['pAUC'],
                'TCR_Group': row['TCR_Group']
            })

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        logger.warning("No data for fingerprinting pAUC boxplot")
        return

    # Create the boxplot
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    model_order = [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER_LOCAL]
    groups = [GROUP_DISPLAY['R-5'], GROUP_DISPLAY['non-R-5']]

    # Create positions for grouped boxplots
    box_width = 0.35
    group_positions = np.arange(len(model_order))

    # Plot boxplots for each group
    for i, group in enumerate(groups):
        subset = plot_df[plot_df['Group'] == group]

        data_by_model = []
        pos = []
        for j, model in enumerate(model_order):
            model_data = subset[subset['Model'] == model]['pAUC'].values
            if len(model_data) > 0:
                data_by_model.append(model_data)
                pos.append(j + (i - 0.5) * box_width)

        if data_by_model:
            bp = ax.boxplot(data_by_model, positions=pos, widths=box_width * 0.8,
                           patch_artist=True, showfliers=False)

            for patch in bp['boxes']:
                patch.set_facecolor(COLORS[group])
                patch.set_alpha(0.7)
            for whisker in bp['whiskers']:
                whisker.set_color('gray')
            for cap in bp['caps']:
                cap.set_color('gray')
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

    # Plot individual TCR pAUC values as scatter points
    np.random.seed(42)
    for i, group in enumerate(groups):
        subset = plot_df[plot_df['Group'] == group]
        color = COLORS[group]

        for j, model in enumerate(model_order):
            model_data = subset[subset['Model'] == model]['pAUC'].values
            if len(model_data) > 0:
                x_pos = j + (i - 0.5) * box_width
                jitter = np.random.uniform(-box_width * 0.25, box_width * 0.25, len(model_data))
                ax.scatter(x_pos + jitter, model_data, c=color, alpha=0.6, s=30,
                          edgecolor='white', linewidth=0.5, zorder=4)

    # Add macro average markers (diamonds)
    for i, group in enumerate(groups):
        subset = plot_df[plot_df['Group'] == group]
        for j, model in enumerate(model_order):
            model_data = subset[subset['Model'] == model]['pAUC']
            if len(model_data) > 0:
                mean_val = model_data.mean()
                x_pos = j + (i - 0.5) * box_width
                ax.scatter(x_pos, mean_val, marker='D', color='white', edgecolor='black',
                          s=60, zorder=5, linewidths=1.5)

    # Add top 3 TCR labels for each bar (horizontally separated)
    top_n = 3
    x_offsets = [-0.06, 0, 0.06]  # Horizontal offsets for 3 labels

    for i, group in enumerate(groups):
        subset = plot_df[plot_df['Group'] == group]

        for j, model in enumerate(model_order):
            model_subset = subset[subset['Model'] == model]
            if len(model_subset) > 0:
                x_pos = j + (i - 0.5) * box_width
                top_tcrs = model_subset.nlargest(min(top_n, len(model_subset)), 'pAUC')

                for k, (_, row) in enumerate(top_tcrs.iterrows()):
                    if k < len(x_offsets):
                        tcr_label = row['TCR_Group']
                        # Truncate long TCR names
                        if len(tcr_label) > 10:
                            tcr_label = tcr_label[:10]
                        ax.annotate(tcr_label,
                                   (x_pos + x_offsets[k], row['pAUC'] + 0.02),
                                   fontsize=5, ha='center', va='bottom', rotation=90,
                                   alpha=0.8, zorder=6)

    # Add random baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Formatting
    ax.set_xticks(group_positions)
    ax.set_xticklabels(model_order, fontsize=11)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Partial AUC$_{0.1}$', fontsize=12)
    ax.set_title('FingerPrinting: Per-TCR pAUC by Peptide Position Group', fontsize=14, fontweight='bold')
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(-0.5, len(model_order) - 0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS[GROUP_DISPLAY['R-5']], alpha=0.7, label=GROUP_DISPLAY['R-5']),
        Patch(facecolor=COLORS[GROUP_DISPLAY['non-R-5']], alpha=0.7, label=GROUP_DISPLAY['non-R-5']),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='black',
               markersize=8, label='Macro pAUC$_{0.1}$'),
        Line2D([0], [0], color='gray', linestyle='--', label='Random Baseline')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Add stats annotation
    stats_text = ""
    for group in groups:
        subset = plot_df[plot_df['Group'] == group]
        n_tcrs = subset['TCR_Group'].nunique()
        mean_pauc = subset['pAUC'].mean()
        stats_text += f"{group}: {n_tcrs} TCRs, mean pAUC={mean_pauc:.3f}\n"

    ax.text(0.02, 0.02, stats_text.strip(), transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = paths.get_visualization_file('fingerprinting_pauc_boxplot.png')
    save_figure(fig, output_path)
    logger.info(f"Saved fingerprinting pAUC boxplot: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting pAUC boxplot")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match"],
                        default="exact_match", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_pauc_boxplot", level="INFO")

    create_fingerprinting_pauc_boxplot(args.mode)


if __name__ == "__main__":
    main()
