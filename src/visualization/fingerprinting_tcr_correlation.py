"""
Fingerprinting Per-TCR Correlation Heatmap Visualization.

Generates a grid of 8 model subplots, each showing 21 TCR correlations
(Spearman correlation between prediction probability and log2foldchange).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Optional
import argparse
import logging
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name

logger = logging.getLogger(__name__)

# TCR groups in specific order (matching other fingerprinting heatmaps)
TCR_GROUPS = [
    'SVAR.1', 'SVAR.2', 'SVAR.3', 'SVAR.5', 'SVAR.6', 'SVAR.7',
    'SVAR.13', 'SVAR.14', 'SVAR.16', 'SVAR.21', 'SVAR.26', 'SVAR.56',
    'SVAR.57', 'SVAR.59', 'SVAR.73', 'SVAR.74', 'SVAR.75', 'SVAR.76',
    'SVAR.82', 'SVAR.85', 'SVAR.90'
]


def calculate_tcr_correlations(df: pd.DataFrame, prob_col: str) -> dict:
    """
    Calculate Spearman correlation for each TCR group.

    Args:
        df: DataFrame with predictions and log2foldchange
        prob_col: Prediction probability column name

    Returns:
        Dictionary mapping TCR group to correlation value
    """
    correlations = {}
    tcr_groups = sorted(df['TCR_Group'].unique())

    for tcr_group in tcr_groups:
        tcr_df = df[df['TCR_Group'] == tcr_group]
        pred_vals = tcr_df[prob_col].values
        log2fc_vals = tcr_df['log2foldchange'].values

        # Remove NaN values
        mask = ~(np.isnan(pred_vals) | np.isnan(log2fc_vals))

        if mask.sum() > 3:
            try:
                r, _ = spearmanr(pred_vals[mask], log2fc_vals[mask])
                correlations[tcr_group] = r if not np.isnan(r) else 0.0
            except:
                correlations[tcr_group] = 0.0
        else:
            correlations[tcr_group] = 0.0

    return correlations


def create_tcr_correlation_grid(df: pd.DataFrame, output_path: Path):
    """
    Create 8-model grid showing per-TCR correlation heatmaps.
    Each model's subplot shows TCR correlations in a 5x5 grid layout.

    Args:
        df: Merged DataFrame with predictions and log2foldchange
        output_path: Output file path
    """
    setup_plot_style()

    models = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']

    # Custom colormap: blue -> white -> red
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('custom_rwb', colors, N=256)

    # Create figure with 4x2 grid for 8 models (4 rows, 2 columns)
    fig, axes = plt.subplots(4, 2, figsize=(8, 14))
    axes = axes.flatten()

    im = None
    for i, model in enumerate(models):
        ax = axes[i]
        prob_col = f'{model}_Prob'

        if prob_col not in df.columns:
            ax.set_visible(False)
            continue

        # Calculate correlations for each TCR
        correlations = calculate_tcr_correlations(df, prob_col)

        # Create 5x5 matrix for TCR correlations
        # First 20 TCRs in rows 0-4, cols 0-3; SVAR.90 in position [4,4]
        corr_matrix = np.full((5, 5), np.nan)
        tcr_list = TCR_GROUPS[:-1]  # First 20 TCRs

        for j, tcr in enumerate(tcr_list):
            row = j // 4
            col = j % 4
            corr_matrix[row, col] = correlations.get(tcr, 0.0)

        # SVAR.90 in bottom right corner
        corr_matrix[4, 4] = correlations.get('SVAR.90', 0.0)

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

        # Add text annotations with TCR names and correlation values
        for j, tcr in enumerate(tcr_list):
            row = j // 4
            col = j % 4
            val = correlations.get(tcr, 0.0)
            color = 'white' if abs(val) > 0.5 else 'black'
            # Show TCR name (shortened) and correlation
            tcr_short = tcr.replace('SVAR.', '')
            ax.text(col, row, f'{tcr_short}\n{val:.2f}', ha='center', va='center',
                   fontsize=6, color=color, fontweight='bold')

        # SVAR.90
        val_90 = correlations.get('SVAR.90', 0.0)
        color_90 = 'white' if abs(val_90) > 0.5 else 'black'
        ax.text(4, 4, f'90\n{val_90:.2f}', ha='center', va='center',
               fontsize=6, color=color_90, fontweight='bold')

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Title with overall correlation
        all_corrs = [correlations.get(tcr, 0.0) for tcr in TCR_GROUPS]
        overall_corr = np.mean(all_corrs)
        display_name = format_model_name(model)
        ax.set_title(f'{display_name} (mean ρ={overall_corr:.3f})', fontsize=10, fontweight='bold')

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.84])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Spearman ρ', fontsize=10)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Main title
    fig.suptitle('Per-TCR Spearman Correlation: Prediction vs Log2FC',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.90, 0.96])
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_tcr_correlation_combined_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create a single combined heatmap with TCRs as rows and models as columns.

    Args:
        df: Merged DataFrame with predictions and log2foldchange
        output_path: Output file path
    """
    setup_plot_style()

    models = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']
    model_display = ['ERGO', 'ERGO2', 'NetTCR2', 'NetTCR-2.2', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']
    tcr_groups = sorted(df['TCR_Group'].unique())

    # Custom colormap
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('custom_rwb', colors, N=256)

    # Build correlation matrix: TCRs x Models
    corr_matrix = np.zeros((len(tcr_groups), len(models)))

    for j, model in enumerate(models):
        prob_col = f'{model}_Prob'
        if prob_col not in df.columns:
            continue

        correlations = calculate_tcr_correlations(df, prob_col)

        for i, tcr in enumerate(tcr_groups):
            corr_matrix[i, j] = correlations.get(tcr, 0.0)

    # Calculate summary statistics
    mean_per_model = np.mean(corr_matrix, axis=0)
    mean_per_tcr = np.mean(corr_matrix, axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap with annotations
    sns.heatmap(corr_matrix, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                xticklabels=model_display, yticklabels=tcr_groups,
                linewidths=0.5, cbar_kws={'label': 'Spearman Correlation (ρ)'})

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('TCR Group', fontsize=12)
    ax.set_title('Per-TCR Spearman Correlation: Prediction vs Log2FC\n(All Models)',
                 fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add summary text
    summary_text = "Mean ρ per model:\n"
    for model, mean_val in zip(model_display, mean_per_model):
        summary_text += f"  {model}: {mean_val:.3f}\n"

    ax.text(1.18, 0.5, summary_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_tcr_correlation_visualizations(mode: str):
    """Generate all TCR correlation visualizations."""
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

    # Create output directory
    output_dir = paths.VISUALIZATIONS_DIR / 'fingerprinting_heatmaps'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 8-model grid
    logger.info("Generating 8-model TCR correlation grid...")
    output_path = output_dir / 'fingerprinting_tcr_correlation_8models.png'
    create_tcr_correlation_grid(df, output_path)

    # Generate combined heatmap
    logger.info("Generating combined TCR correlation heatmap...")
    output_path = output_dir / 'fingerprinting_tcr_correlation_combined.png'
    create_tcr_correlation_combined_heatmap(df, output_path)

    logger.info(f"All TCR correlation visualizations saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting per-TCR correlation heatmaps")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match"],
                        default="exact_match", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_tcr_correlation", level="INFO")

    generate_tcr_correlation_visualizations(args.mode)


if __name__ == "__main__":
    main()
