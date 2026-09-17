"""
Fingerprinting Position-based Spearman Correlation Analysis.

Analyzes the correlation between model predictions and log2foldchange values,
grouped by mutation position (1-9) relative to the reference peptide YLQPRTFLL.

Output:
- fingerprinting_position_correlation.png: 8 subplots with 3x3 position grids
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name

logger = logging.getLogger(__name__)

# Reference peptide
REFERENCE_PEPTIDE = "YLQPRTFLL"

# Model configuration
MODEL_ORDER_LOCAL = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']
MODEL_DISPLAY = {
    'ERGO': 'ERGO', 'ERGO2': 'ERGO2', 'NetTCR': 'NetTCR2',
    'NetTCR22': 'NetTCR-2.2', 'TITAN': 'TITAN', 'EPACT': 'EPACT',
    'PanPep': 'PanPep', 'SCEPTR': 'SCEPTR'
}


def get_mutation_position(peptide: str, reference: str = REFERENCE_PEPTIDE) -> int:
    """
    Return mutation position (1-9) or 0 for reference peptide.

    Args:
        peptide: Peptide sequence to analyze
        reference: Reference peptide sequence

    Returns:
        Position of mutation (1-9) or 0 if reference/multiple mutations
    """
    if peptide == reference:
        return 0

    if len(peptide) != len(reference):
        return 0

    diff_positions = []
    for i in range(len(peptide)):
        if peptide[i] != reference[i]:
            diff_positions.append(i + 1)  # 1-indexed

    # Only return position for single mutations
    if len(diff_positions) == 1:
        return diff_positions[0]

    return 0


def calculate_position_correlation(
    df: pd.DataFrame,
    prob_col: str,
    position: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Spearman correlation for a specific mutation position.

    Args:
        df: DataFrame with predictions and log2foldchange
        prob_col: Prediction probability column name
        position: Mutation position (1-9)

    Returns:
        Tuple of (correlation, p_value) or (None, None) if insufficient data
    """
    # Add position column if not present
    if 'Position' not in df.columns:
        df = df.copy()
        df['Position'] = df['Peptide'].apply(get_mutation_position)

    # Filter for this position
    pos_df = df[df['Position'] == position]

    if len(pos_df) < 4:
        return None, None

    predictions = pos_df[prob_col].values
    log2fc = pos_df['log2foldchange'].values

    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(log2fc))
    predictions = predictions[mask]
    log2fc = log2fc[mask]

    if len(predictions) < 4:
        return None, None

    try:
        r, p = spearmanr(predictions, log2fc)
        return r, p
    except Exception as e:
        logger.debug(f"Error calculating correlation for position {position}: {e}")
        return None, None


def create_position_correlation_plot(mode: str):
    """
    Create 8-subplot figure with 3x3 position grids showing Spearman correlation.

    Args:
        mode: Deduplication mode ("exact_match" or "fuzzy_match")
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

    # Add position column
    df['Position'] = df['Peptide'].apply(get_mutation_position)

    # Log position distribution
    pos_counts = df['Position'].value_counts().sort_index()
    logger.info(f"Position distribution:\n{pos_counts}")

    # Setup plot
    setup_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    # Colormap (blue-white-red)
    cmap = plt.cm.RdBu_r

    # Calculate all correlations first to determine color range
    all_correlations = []
    model_correlations = {}

    for model in MODEL_ORDER_LOCAL:
        prob_col = f'{model}_Prob'
        if prob_col not in df.columns:
            continue

        model_correlations[model] = {}
        for pos in range(1, 10):
            r, p = calculate_position_correlation(df, prob_col, pos)
            model_correlations[model][pos] = r
            if r is not None:
                all_correlations.append(r)

    # Determine color normalization (symmetric around 0)
    if len(all_correlations) > 0:
        max_abs = max(abs(min(all_correlations)), abs(max(all_correlations)))
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = -1, 1

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Process each model
    for model_idx, model in enumerate(MODEL_ORDER_LOCAL):
        ax = axes[model_idx]
        display_name = MODEL_DISPLAY.get(model, model)

        if model not in model_correlations:
            ax.text(0.5, 0.5, f'{display_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.axis('off')
            continue

        # Create 3x3 correlation matrix
        corr_matrix = np.full((3, 3), np.nan)

        for pos in range(1, 10):
            row = (pos - 1) // 3
            col = (pos - 1) % 3
            r = model_correlations[model].get(pos)
            if r is not None:
                corr_matrix[row, col] = r

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

        # Add text annotations
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j + 1
                val = corr_matrix[i, j]

                if not np.isnan(val):
                    # Choose text color based on background
                    text_color = 'white' if abs(val) > 0.5 * max_abs else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=9, color=text_color, fontweight='bold')

                # Add position label at bottom of cell
                ref_aa = REFERENCE_PEPTIDE[pos - 1]
                ax.text(j, i + 0.35, f'P{pos}({ref_aa})', ha='center', va='center',
                       fontsize=6, color='gray')

        # Highlight position 5 (the original R-5 focus)
        # Draw a thicker border around position 5 cell
        pos5_row, pos5_col = 1, 1  # Position 5 is at (1,1) in 0-indexed 3x3 grid
        rect = patches.Rectangle((pos5_col - 0.5, pos5_row - 0.5), 1, 1,
                                 linewidth=2, edgecolor='gold', facecolor='none')
        ax.add_patch(rect)

        # Formatting
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(display_name, fontsize=11, fontweight='bold')

        # Add border around subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Spearman ρ', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Main title
    fig.suptitle('FingerPrinting: Spearman Correlation (Predictions vs Log2FC) by Mutation Position',
                 fontsize=12, fontweight='bold', y=1.02)

    # Add legend note
    fig.text(0.02, 0.02,
            f'Reference: {REFERENCE_PEPTIDE} | Gold border: Position 5 (R)\n'
            f'P1-P9: Mutation positions | Values: Spearman ρ',
            fontsize=8, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 0.9, 1])

    # Save
    output_path = paths.get_visualization_file('fingerprinting_position_correlation.png')
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")

    # Print summary statistics
    logger.info("\n=== Correlation Summary ===")
    for model in MODEL_ORDER_LOCAL:
        if model in model_correlations:
            display_name = MODEL_DISPLAY.get(model, model)
            correlations = [r for r in model_correlations[model].values() if r is not None]
            if correlations:
                mean_r = np.mean(correlations)
                logger.info(f"{display_name}: mean ρ = {mean_r:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting position correlation analysis")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match"],
                        default="exact_match", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_position_correlation", level="INFO")

    create_position_correlation_plot(args.mode)


if __name__ == "__main__":
    main()
