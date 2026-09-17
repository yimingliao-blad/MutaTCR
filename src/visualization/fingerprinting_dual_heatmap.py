"""
Fingerprinting Dual-Panel Heatmap Visualization.

Generates per-model heatmaps showing both log2foldchange (left) and
predicted probability (right) across all peptide variants.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name

logger = logging.getLogger(__name__)

# Reference peptide and amino acid order
REFERENCE_PEPTIDE = "YLQPRTFLL"
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'V', 'Y']

# TCR groups in specific order (matching fingerprinting_heatmap.py)
TCR_GROUPS = [
    'SVAR.1', 'SVAR.2', 'SVAR.3', 'SVAR.5', 'SVAR.6', 'SVAR.7',
    'SVAR.13', 'SVAR.14', 'SVAR.16', 'SVAR.21', 'SVAR.26', 'SVAR.56',
    'SVAR.57', 'SVAR.59', 'SVAR.73', 'SVAR.74', 'SVAR.75', 'SVAR.76',
    'SVAR.82', 'SVAR.85', 'SVAR.90'
]


def get_position_and_aa_from_peptide(peptide: str) -> Tuple[int, str]:
    """Extract position and substituted amino acid from peptide variant."""
    if len(peptide) != len(REFERENCE_PEPTIDE):
        return (0, '')

    diffs = []
    for i, (p, r) in enumerate(zip(peptide.upper(), REFERENCE_PEPTIDE)):
        if p != r:
            diffs.append((i + 1, p))

    if len(diffs) == 1:
        return diffs[0]
    elif len(diffs) == 0:
        return (0, peptide[0])
    return (0, '')


def prepare_heatmap_data(df: pd.DataFrame, tcr_group: str, value_col: str):
    """
    Prepare heatmap matrix for a single TCR group.

    Returns:
        Tuple of (value_matrix, status_matrix) where status_matrix indicates
        'valid', 'unstable', or 'wild type' for each cell
    """
    tcr_df = df[df['TCR_Group'] == tcr_group].copy()

    if len(tcr_df) == 0:
        return np.full((20, 9), np.nan), np.full((20, 9), '', dtype=object)

    matrix = np.full((20, 9), np.nan)
    status_matrix = np.full((20, 9), '', dtype=object)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    for _, row in tcr_df.iterrows():
        peptide = row['Peptide']
        value = row.get(value_col, np.nan)
        status = row.get('Antigen_Status', 'valid')

        pos, aa = get_position_and_aa_from_peptide(peptide)

        if pos > 0 and aa in aa_to_idx:
            row_idx = aa_to_idx[aa]
            col_idx = pos - 1
            matrix[row_idx, col_idx] = value
            status_matrix[row_idx, col_idx] = status

    return matrix, status_matrix


def get_reference_aa_positions() -> List[Tuple[int, int]]:
    """Get the (row, col) positions of reference amino acids."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    positions = []
    for col_idx, aa in enumerate(REFERENCE_PEPTIDE):
        if aa in aa_to_idx:
            positions.append((aa_to_idx[aa], col_idx))
    return positions


def create_dual_panel_heatmap(
    df: pd.DataFrame,
    model_key: str,
    output_path: Path
):
    """
    Create dual-panel heatmap for a model showing log2fc and probability.

    Args:
        df: Merged DataFrame with predictions and log2foldchange
        model_key: Model key (e.g., 'ERGO', 'NetTCR22')
        output_path: Output file path
    """
    setup_plot_style()

    prob_col = f'{model_key}_Prob'
    if prob_col not in df.columns:
        logger.warning(f"Column {prob_col} not found")
        return

    # Get TCR groups
    tcr_groups = sorted(df['TCR_Group'].unique())

    # Custom colormap: blue -> white -> red
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('custom_rwb', colors, N=256)

    # Get reference positions for black borders
    ref_positions = get_reference_aa_positions()

    # Create figure with subplots for each TCR
    n_tcrs = len(tcr_groups)
    fig, axes = plt.subplots(n_tcrs, 2, figsize=(10, 3 * n_tcrs))

    if n_tcrs == 1:
        axes = axes.reshape(1, 2)

    for i, tcr_group in enumerate(tcr_groups):
        # Left: log2foldchange
        ax_log2fc = axes[i, 0]
        log2fc_matrix = prepare_heatmap_data(df, tcr_group, 'log2foldchange')

        # Clip log2fc to 0-4 range
        log2fc_matrix_clipped = np.clip(log2fc_matrix, 0, 4)

        im_log2fc = ax_log2fc.imshow(log2fc_matrix_clipped, cmap=cmap, vmin=0, vmax=4, aspect='auto')

        ax_log2fc.set_xticks(range(9))
        ax_log2fc.set_xticklabels(range(1, 10), fontsize=8)
        ax_log2fc.set_yticks(range(20))
        ax_log2fc.set_yticklabels(AMINO_ACIDS, fontsize=7)
        ax_log2fc.set_title(f'{tcr_group}: Log2 FC pMHC Binding', fontsize=10, fontweight='bold')

        # Mark reference amino acids
        for row_idx, col_idx in ref_positions:
            rect = patches.Rectangle(
                (col_idx - 0.5, row_idx - 0.5), 1, 1,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax_log2fc.add_patch(rect)

        # Right: prediction probability
        ax_prob = axes[i, 1]
        prob_matrix = prepare_heatmap_data(df, tcr_group, prob_col)

        im_prob = ax_prob.imshow(prob_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        ax_prob.set_xticks(range(9))
        ax_prob.set_xticklabels(range(1, 10), fontsize=8)
        ax_prob.set_yticks(range(20))
        ax_prob.set_yticklabels(AMINO_ACIDS, fontsize=7)
        ax_prob.set_title(f'{tcr_group}: Pred Prob. pMHC Binding', fontsize=10, fontweight='bold')

        # Mark reference amino acids
        for row_idx, col_idx in ref_positions:
            rect = patches.Rectangle(
                (col_idx - 0.5, row_idx - 0.5), 1, 1,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax_prob.add_patch(rect)

    # Add colorbars
    # Log2FC colorbar (left side)
    cbar_ax1 = fig.add_axes([0.08, 0.02, 0.35, 0.015])
    cbar1 = fig.colorbar(im_log2fc, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Log2 FC pMHC Binding', fontsize=9)
    cbar1.set_ticks([0, 1, 2, 3, 4])

    # Probability colorbar (right side)
    cbar_ax2 = fig.add_axes([0.57, 0.02, 0.35, 0.015])
    cbar2 = fig.colorbar(im_prob, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Pred Prob. pMHC Binding', fontsize=9)
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Main title
    display_name = format_model_name(model_key)
    fig.suptitle(f'{display_name}: Log2FC vs Prediction Probability',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_single_tcr_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    status_matrix: np.ndarray,
    tcr_group: str,
    cmap,
    vmin: float,
    vmax: float,
    ref_positions: List[Tuple[int, int]],
    show_ylabel: bool = False,
    show_xlabel: bool = False,
    norm=None
):
    """
    Create a single heatmap on the given axes with square cells.

    Args:
        ax: Matplotlib axes
        matrix: 20x9 heatmap matrix
        status_matrix: 20x9 matrix with Antigen_Status values
        tcr_group: TCR group name for title
        cmap: Colormap
        vmin, vmax: Color scale limits
        ref_positions: Reference amino acid positions for black borders
        show_ylabel: Whether to show y-axis labels
        show_xlabel: Whether to show x-axis labels
        norm: Optional normalization (e.g., TwoSlopeNorm for centering)
    """
    # Create a copy of matrix for display, masking invalid entries
    display_matrix = matrix.copy()

    # Plot the heatmap
    if norm is not None:
        im = ax.imshow(display_matrix, cmap=cmap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(display_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Overlay grey rectangles for unstable and wild type entries
    for row_idx in range(20):
        for col_idx in range(9):
            status = status_matrix[row_idx, col_idx]
            if status in ['unstable', 'wild type']:
                rect = patches.Rectangle(
                    (col_idx - 0.5, row_idx - 0.5), 1, 1,
                    linewidth=0, facecolor='#808080', alpha=1.0
                )
                ax.add_patch(rect)

    # Set ticks
    ax.set_xticks(range(9))
    ax.set_yticks(range(20))

    if show_xlabel:
        ax.set_xticklabels(range(1, 10), fontsize=5)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_yticklabels(AMINO_ACIDS, fontsize=5)
    else:
        ax.set_yticklabels([])

    ax.set_title(tcr_group, fontsize=7, fontweight='bold', pad=2)

    # Mark reference amino acids with black border
    for row_idx, col_idx in ref_positions:
        rect = patches.Rectangle(
            (col_idx - 0.5, row_idx - 0.5), 1, 1,
            linewidth=1, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)

    return im


def create_single_model_dual_heatmap(
    df: pd.DataFrame,
    model_key: str,
    output_dir: Path
):
    """
    Create a dual-panel figure for one model with 5x5 grid layout.

    Layout: Two 5x5 grids side by side (log2fc left, prob right)
    - First 20 TCRs in positions [row 0-4, col 0-3]
    - SVAR.90 in position [row 4, col 4] (bottom right corner)

    Args:
        df: Merged DataFrame
        model_key: Model key
        output_dir: Output directory
    """
    setup_plot_style()

    prob_col = f'{model_key}_Prob'
    if prob_col not in df.columns:
        logger.warning(f"Column {prob_col} not found")
        return

    # Custom colormap for log2FC centered at 1 (matching finger_print.jpg)
    # Scale: -5 (blue) -> 1 (white) -> 5+ (red)
    # We need to create a colormap where the center (white) is at value 1
    # For a scale of -5 to 5, value 1 is at position (1 - (-5)) / (5 - (-5)) = 6/10 = 0.6
    from matplotlib.colors import TwoSlopeNorm
    colors_log2fc = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                     '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap_log2fc = LinearSegmentedColormap.from_list('custom_rwb', colors_log2fc, N=256)

    # Colormap for probability (0-1)
    colors_prob = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                   '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap_prob = LinearSegmentedColormap.from_list('custom_rwb_prob', colors_prob, N=256)

    # Get reference positions
    ref_positions = get_reference_aa_positions()

    # Use predefined TCR order
    tcr_list = TCR_GROUPS[:-1]  # First 20 TCRs (exclude SVAR.90)

    # Create figure with two 5x5 grids side by side
    fig = plt.figure(figsize=(22, 14))

    # Create two main gridspecs for left (log2fc) and right (prob) panels
    from matplotlib.gridspec import GridSpec

    # Main layout: 1 row, 2 columns
    outer_gs = GridSpec(1, 2, figure=fig, wspace=0.12, left=0.04, right=0.96, top=0.92, bottom=0.06)

    # Left panel: Log2FC (5x5 grid)
    gs_left = outer_gs[0].subgridspec(5, 5, hspace=0.20, wspace=0.08)

    # Right panel: Probability (5x5 grid)
    gs_right = outer_gs[1].subgridspec(5, 5, hspace=0.20, wspace=0.08)

    im_log2fc = None
    im_prob = None

    # Log2FC scale: -1 to 5, centered at 1 (matching finger_print.jpg)
    # -1 and less = most blue, 1 = white, 5+ = most red
    log2fc_vmin = -1
    log2fc_vmax = 5
    log2fc_vcenter = 1
    log2fc_norm = TwoSlopeNorm(vmin=log2fc_vmin, vcenter=log2fc_vcenter, vmax=log2fc_vmax)

    # Plot first 20 TCRs in 5x4 grid (rows 0-4, cols 0-3)
    for i, tcr_group in enumerate(tcr_list):
        row = i // 4
        col = i % 4

        # Left panel: log2fc
        ax_log2fc = fig.add_subplot(gs_left[row, col])
        log2fc_matrix, log2fc_status = prepare_heatmap_data(df, tcr_group, 'log2foldchange')
        log2fc_matrix_clipped = np.clip(log2fc_matrix, log2fc_vmin, log2fc_vmax)

        show_ylabel = (col == 0)  # Only first column shows y labels
        show_xlabel = (row == 4)  # Only last row shows x labels

        im_log2fc = create_single_tcr_heatmap(
            ax_log2fc, log2fc_matrix_clipped, log2fc_status, tcr_group, cmap_log2fc,
            log2fc_vmin, log2fc_vmax, ref_positions, show_ylabel=show_ylabel, show_xlabel=show_xlabel,
            norm=log2fc_norm
        )

        # Right panel: probability
        ax_prob = fig.add_subplot(gs_right[row, col])
        prob_matrix, prob_status = prepare_heatmap_data(df, tcr_group, prob_col)

        im_prob = create_single_tcr_heatmap(
            ax_prob, prob_matrix, prob_status, tcr_group, cmap_prob, 0, 1,
            ref_positions, show_ylabel=show_ylabel, show_xlabel=show_xlabel
        )

    # Add SVAR.90 in bottom right corner (row 4, col 4)
    # Left panel
    ax_svar90_log2fc = fig.add_subplot(gs_left[4, 4])
    log2fc_matrix_svar90, log2fc_status_svar90 = prepare_heatmap_data(df, 'SVAR.90', 'log2foldchange')
    log2fc_matrix_svar90_clipped = np.clip(log2fc_matrix_svar90, log2fc_vmin, log2fc_vmax)
    im_log2fc = create_single_tcr_heatmap(
        ax_svar90_log2fc, log2fc_matrix_svar90_clipped, log2fc_status_svar90, 'SVAR.90',
        cmap_log2fc, log2fc_vmin, log2fc_vmax, ref_positions, show_ylabel=False, show_xlabel=True,
        norm=log2fc_norm
    )

    # Right panel
    ax_svar90_prob = fig.add_subplot(gs_right[4, 4])
    prob_matrix_svar90, prob_status_svar90 = prepare_heatmap_data(df, 'SVAR.90', prob_col)
    im_prob = create_single_tcr_heatmap(
        ax_svar90_prob, prob_matrix_svar90, prob_status_svar90, 'SVAR.90',
        cmap_prob, 0, 1, ref_positions, show_ylabel=False, show_xlabel=True
    )

    # Add panel titles
    fig.text(0.25, 0.95, 'Log2 FC pMHC Binding', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.75, 0.95, 'Pred Prob. pMHC Binding', ha='center', fontsize=12, fontweight='bold')

    # Add colorbars in the empty space (column 4, rows 0-3)
    # Log2FC colorbar with scale centered at 1: -5 (blue) -> 1 (white) -> 5 (red)
    cbar_ax1 = fig.add_axes([0.42, 0.50, 0.012, 0.35])
    # Create a ScalarMappable with the same norm for the colorbar
    import matplotlib.cm as cm
    sm_log2fc = cm.ScalarMappable(cmap=cmap_log2fc, norm=log2fc_norm)
    sm_log2fc.set_array([])
    cbar1 = fig.colorbar(sm_log2fc, cax=cbar_ax1, orientation='vertical')
    cbar1.set_label('Log2 FC', fontsize=9)
    cbar1.set_ticks([-1, 0, 1, 2, 3, 4, 5])

    # Probability colorbar
    cbar_ax2 = fig.add_axes([0.95, 0.50, 0.012, 0.35])
    cbar2 = fig.colorbar(im_prob, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('Pred Prob.', fontsize=9)
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Add legend for grey cells (unstable/wild type)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', edgecolor='black', label='Unstable/Wild Type')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=1,
               fontsize=10, bbox_to_anchor=(0.5, 0.01))

    # Main title
    display_name = format_model_name(model_key)
    fig.suptitle(f'{display_name}: Log2FC vs Prediction Probability',
                 fontsize=14, fontweight='bold', y=0.99)

    output_path = output_dir / f'{model_key}_dual_heatmap.png'
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_all_dual_heatmaps(mode: str):
    """Generate dual-panel heatmaps for all models."""
    paths = get_paths(mode=mode)

    # Load merged fingerprinting data
    merged_file = paths.MERGED_DIR / 'fingerprinting_all_models.csv'
    if not merged_file.exists():
        logger.warning(f"Merged file not found: {merged_file}")
        return

    df = pd.read_csv(merged_file)
    logger.info(f"Loaded {len(df)} samples from fingerprinting data")

    # Don't filter - show all entries, but mark unstable/wild type as grey
    if 'Antigen_Status' in df.columns:
        status_counts = df['Antigen_Status'].value_counts()
        logger.info(f"Antigen_Status distribution: {status_counts.to_dict()}")

    # Create output directory
    output_dir = paths.VISUALIZATIONS_DIR / 'fingerprinting_dual_heatmaps'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each model
    models = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']

    for model in models:
        logger.info(f"Generating dual heatmap for {model}...")
        create_single_model_dual_heatmap(df, model, output_dir)

    logger.info(f"All dual heatmaps saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting dual-panel heatmaps")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match"],
                        default="exact_match", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_dual_heatmap", level="INFO")

    generate_all_dual_heatmaps(args.mode)


if __name__ == "__main__":
    main()
