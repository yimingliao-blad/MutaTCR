"""
log2foldchange Scatter Plot Visualization for TCRP Benchmark V2.

Generates scatter plots of log2foldchange vs predicted probability for FingerPrinting data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import argparse
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER, get_all_model_keys
from src.visualization.utils import (
    setup_plot_style, save_figure, get_model_colors, get_epitope_group_colors,
    add_correlation_annotation, add_regression_line, format_model_name
)

logger = logging.getLogger(__name__)


def load_merged_data(paths, dataset: str) -> Optional[pd.DataFrame]:
    """Load merged predictions + unified data."""
    merged_file = paths.get_merged_file(dataset)
    if merged_file.exists():
        return pd.read_csv(merged_file)
    return None


def create_log2fc_scatter_8models(df: pd.DataFrame, output_path: Path):
    """
    Create 8-model log2foldchange scatter plot grid.

    Args:
        df: Merged DataFrame with log2foldchange and predictions
        output_path: Output file path
    """
    setup_plot_style()
    config = get_config()

    if len(df) == 0:
        logger.warning("No data for log2fc scatter")
        return

    # Create 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = get_epitope_group_colors()

    for i, model_key in enumerate(MODEL_ORDER):
        ax = axes[i]

        # Find the correct column name
        prob_col = None
        for col in df.columns:
            if '_Prob' in col:
                # Match model key (handle variations like NetTCR-2.2 -> NetTCR22)
                model_normalized = model_key.lower().replace('-', '').replace('.', '')
                col_normalized = col.lower().replace('-', '').replace('.', '')
                if model_normalized in col_normalized:
                    prob_col = col
                    break

        if prob_col is None:
            ax.text(0.5, 0.5, f'No data for {model_key}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(format_model_name(model_key))
            continue

        # Plot by epitope group with high-contrast colors
        if 'Epitope_Group' in df.columns:
            # Define explicit high-contrast colors for fingerprinting groups
            group_colors = {
                'R-5': '#d62728',      # Bright red - index-4 mutations
                'non-R-5': '#1f77b4',  # Strong blue - other peptides
            }
            groups = df['Epitope_Group'].unique()
            group_corrs = {}  # Store correlations per group
            for group in sorted(groups, reverse=True):  # Plot R-5 on top
                group_df = df[df['Epitope_Group'] == group]
                color = group_colors.get(group, colors.get(group, '#333333'))
                ax.scatter(
                    group_df[prob_col],
                    group_df['log2foldchange'],
                    c=color,
                    alpha=0.6,
                    s=25,
                    label=group,
                    edgecolor='white',
                    linewidth=0.3
                )
                # Calculate per-group correlation
                gx = group_df[prob_col].values
                gy = group_df['log2foldchange'].values
                gmask = ~(np.isnan(gx) | np.isnan(gy))
                if gmask.sum() > 3:
                    from scipy.stats import spearmanr
                    r, _ = spearmanr(gx[gmask], gy[gmask])
                    group_corrs[group] = r
        else:
            ax.scatter(
                df[prob_col],
                df['log2foldchange'],
                c='#1f77b4',  # Strong blue
                alpha=0.6,
                s=25,
                edgecolor='white',
                linewidth=0.3
            )
            group_corrs = {}

        # Add zero line (threshold)
        ax.axhline(y=0, color='red', linestyle=':', linewidth=1, alpha=0.5)

        # Add overall correlation
        x = df[prob_col].values
        y = df['log2foldchange'].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 3:
            from scipy.stats import spearmanr
            r_all, _ = spearmanr(x[mask], y[mask])
            add_regression_line(ax, x[mask], y[mask], color='black')

            # Build correlation text with per-group values
            corr_text = f'ρ={r_all:.3f}'
            if group_corrs:
                for grp, r_val in sorted(group_corrs.items()):
                    corr_text += f'\n{grp}: ρ={r_val:.3f}'
            ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Formatting
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('log2 Fold Change')
        ax.set_title(format_model_name(model_key))
        ax.set_xlim(0, 1)

    # Add shared legend if we have epitope groups
    if 'Epitope_Group' in df.columns:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=len(labels),
                       bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('FingerPrinting: log2 Fold Change vs Predicted Probability', y=1.05)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_log2fc_scatter_by_threshold(df: pd.DataFrame, output_path: Path, threshold: float = 1.0):
    """
    Create 8-model log2foldchange scatter plot grid grouped by threshold.

    Points are colored by whether log2foldchange >= threshold (positive/red)
    or log2foldchange < threshold (negative/blue).

    Args:
        df: Merged DataFrame with log2foldchange and predictions
        output_path: Output file path
        threshold: log2foldchange threshold (default: 1.0)
    """
    setup_plot_style()

    if len(df) == 0:
        logger.warning("No data for log2fc scatter by threshold")
        return

    # Create 4x2 grid (4 rows, 2 columns)
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    axes = axes.flatten()

    # Define colors: positive (>=threshold) = red, negative (<threshold) = blue
    pos_color = '#d62728'  # Red
    neg_color = '#1f77b4'  # Blue

    # Add threshold-based grouping
    df = df.copy()
    df['Binding_Group'] = df['log2foldchange'].apply(
        lambda x: 'Positive' if x >= threshold else 'Negative'
    )

    for i, model_key in enumerate(MODEL_ORDER):
        ax = axes[i]

        # Find the correct column name
        prob_col = None
        for col in df.columns:
            if '_Prob' in col:
                model_normalized = model_key.lower().replace('-', '').replace('.', '')
                col_normalized = col.lower().replace('-', '').replace('.', '')
                if model_normalized in col_normalized:
                    prob_col = col
                    break

        if prob_col is None:
            ax.text(0.5, 0.5, f'No data for {model_key}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(format_model_name(model_key))
            continue

        # Plot negative first (so positive points are on top)
        group_corrs = {}
        for group, color in [('Negative', neg_color), ('Positive', pos_color)]:
            group_df = df[df['Binding_Group'] == group]
            if len(group_df) > 0:
                ax.scatter(
                    group_df[prob_col],
                    group_df['log2foldchange'],
                    c=color,
                    alpha=0.6,
                    s=25,
                    label=f'{group} (n={len(group_df)})',
                    edgecolor='white',
                    linewidth=0.3
                )
                # Calculate per-group correlation
                gx = group_df[prob_col].values
                gy = group_df['log2foldchange'].values
                gmask = ~(np.isnan(gx) | np.isnan(gy))
                if gmask.sum() > 3:
                    from scipy.stats import spearmanr
                    r, _ = spearmanr(gx[gmask], gy[gmask])
                    group_corrs[group] = r

        # Add threshold line at log2foldchange = threshold
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold (log2FC={threshold})')

        # Add overall correlation
        x = df[prob_col].values
        y = df['log2foldchange'].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 3:
            from scipy.stats import spearmanr
            r_all, _ = spearmanr(x[mask], y[mask])
            add_regression_line(ax, x[mask], y[mask], color='gray')

            # Build correlation text with per-group values
            corr_text = f'ρ={r_all:.3f}'
            if group_corrs:
                for grp in ['Positive', 'Negative']:
                    if grp in group_corrs:
                        corr_text += f'\n{grp}: ρ={group_corrs[grp]:.3f}'
            ax.text(0.98, 0.02, corr_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Formatting
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('log2 Fold Change')
        ax.set_title(format_model_name(model_key))
        ax.set_xlim(0, 1)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('FingerPrinting: log2 Fold Change vs Predicted Probability', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_log2fc_plots(mode: str):
    """
    Generate log2fc scatter plots for a mode.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)

    logger.info(f"Generating log2fc scatter plots for {mode}...")

    # Load FingerPrinting merged data
    df = load_merged_data(paths, 'fingerprinting')
    if df is None:
        logger.warning("FingerPrinting merged data not found")
        return

    # Filter to only 'valid' Antigen_Status entries
    if 'Antigen_Status' in df.columns:
        df = df[df['Antigen_Status'] == 'valid']
        logger.info(f"Filtered to {len(df)} valid entries")

    if 'log2foldchange' not in df.columns:
        logger.warning("log2foldchange column not found in data")
        return

    # 8-model grid
    output_path = paths.get_visualization_file('log2fc_scatter_8models.png')
    create_log2fc_scatter_8models(df, output_path)

    # 8-model grid grouped by log2FC threshold
    output_path_threshold = paths.get_visualization_file('log2fc_scatter_by_threshold.png')
    create_log2fc_scatter_by_threshold(df, output_path_threshold, threshold=1.0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate log2fc scatter plots")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("log2fc_scatter", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_log2fc_plots(mode)


if __name__ == "__main__":
    main()
