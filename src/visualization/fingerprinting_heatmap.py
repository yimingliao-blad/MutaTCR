"""
Fingerprinting TCR Heatmap Visualization for TCRP Benchmark V2.

Generates per-TCR heatmaps showing predicted binding probabilities
across all 172 peptide variants (single amino acid substitutions).

Reference: finger_print.jpg shows the expected visualization style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse
import logging
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure, format_model_name

logger = logging.getLogger(__name__)

# Reference peptide and amino acid order
REFERENCE_PEPTIDE = "YLQPRTFLL"
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'V', 'Y']

# TCR groups from the fingerprinting experiment
TCR_GROUPS = [
    'SVAR.1', 'SVAR.2', 'SVAR.3', 'SVAR.5', 'SVAR.6', 'SVAR.7',
    'SVAR.13', 'SVAR.14', 'SVAR.16', 'SVAR.21', 'SVAR.26', 'SVAR.56',
    'SVAR.57', 'SVAR.59', 'SVAR.73', 'SVAR.74', 'SVAR.75', 'SVAR.76',
    'SVAR.82', 'SVAR.85', 'SVAR.90'
]


def get_position_and_aa_from_peptide(peptide: str) -> Tuple[int, str]:
    """
    Extract position and substituted amino acid from peptide variant.

    Args:
        peptide: Peptide sequence (e.g., 'ALQPRTFLL')

    Returns:
        Tuple of (position_1indexed, amino_acid) or (0, '') if not a single substitution
    """
    if len(peptide) != len(REFERENCE_PEPTIDE):
        return (0, '')

    diffs = []
    for i, (p, r) in enumerate(zip(peptide.upper(), REFERENCE_PEPTIDE)):
        if p != r:
            diffs.append((i + 1, p))  # 1-indexed position

    if len(diffs) == 1:
        return diffs[0]
    elif len(diffs) == 0:
        # Wild type - return position 0 to indicate original
        return (0, peptide[0])
    return (0, '')


def load_merged_data(paths, dataset: str) -> Optional[pd.DataFrame]:
    """Load merged predictions + unified data."""
    merged_file = paths.get_merged_file(dataset)
    if merged_file.exists():
        return pd.read_csv(merged_file)
    return None


def prepare_heatmap_data(
    df: pd.DataFrame,
    tcr_group: str,
    prob_col: str
):
    """
    Prepare heatmap matrix for a single TCR group.

    Args:
        df: Merged DataFrame
        tcr_group: TCR group name (e.g., 'SVAR.1')
        prob_col: Prediction probability column name

    Returns:
        Tuple of (value_matrix, status_matrix) - 20x9 numpy arrays
    """
    # Filter for this TCR group
    tcr_df = df[df['TCR_Group'] == tcr_group].copy()

    if len(tcr_df) == 0:
        return np.full((20, 9), np.nan), np.full((20, 9), '', dtype=object)

    # Initialize matrices
    matrix = np.full((20, 9), np.nan)
    status_matrix = np.full((20, 9), '', dtype=object)

    # Map amino acids to row indices
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    # Fill matrix
    for _, row in tcr_df.iterrows():
        peptide = row['Peptide']
        prob = row.get(prob_col, np.nan)
        status = row.get('Antigen_Status', 'valid')

        pos, aa = get_position_and_aa_from_peptide(peptide)

        if pos > 0 and aa in aa_to_idx:
            row_idx = aa_to_idx[aa]
            col_idx = pos - 1  # 0-indexed
            matrix[row_idx, col_idx] = prob
            status_matrix[row_idx, col_idx] = status

    return matrix, status_matrix


def get_reference_aa_positions() -> List[Tuple[int, int]]:
    """
    Get the (row, col) positions of reference amino acids in the heatmap.

    Returns:
        List of (row_idx, col_idx) for reference amino acids
    """
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    positions = []

    for col_idx, aa in enumerate(REFERENCE_PEPTIDE):
        if aa in aa_to_idx:
            positions.append((aa_to_idx[aa], col_idx))

    return positions


def calculate_tcr_correlation(
    df: pd.DataFrame,
    tcr_group: str,
    prob_col: str
) -> Optional[float]:
    """
    Calculate correlation between predictions and log2foldchange for a TCR.

    Args:
        df: DataFrame with predictions and log2foldchange
        tcr_group: TCR group name
        prob_col: Prediction probability column name

    Returns:
        Spearman correlation coefficient or None if not enough data
    """
    tcr_df = df[df['TCR_Group'] == tcr_group]

    if len(tcr_df) < 4:
        return None

    if prob_col not in tcr_df.columns or 'log2foldchange' not in tcr_df.columns:
        return None

    pred_vals = tcr_df[prob_col].values
    log2fc_vals = tcr_df['log2foldchange'].values

    mask = ~(np.isnan(pred_vals) | np.isnan(log2fc_vals))

    if mask.sum() > 3:
        try:
            r, _ = spearmanr(pred_vals[mask], log2fc_vals[mask])
            return r
        except:
            return None

    return None


def create_single_tcr_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    status_matrix: np.ndarray,
    tcr_group: str,
    cmap,
    vmin: float = 0,
    vmax: float = 1,
    show_colorbar: bool = False,
    correlation: Optional[float] = None
):
    """
    Create heatmap for a single TCR group.

    Args:
        ax: Matplotlib axes
        matrix: 20x9 heatmap matrix
        status_matrix: 20x9 matrix with Antigen_Status values
        tcr_group: TCR group name for title
        cmap: Colormap
        vmin, vmax: Color scale limits
        show_colorbar: Whether to show colorbar
        correlation: Correlation value to display in title
    """
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

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

    # Set ticks with smaller fonts for compact layout
    ax.set_xticks(range(9))
    ax.set_xticklabels(range(1, 10), fontsize=5)
    ax.set_yticks(range(20))
    ax.set_yticklabels(AMINO_ACIDS, fontsize=5)

    # Title with correlation
    if correlation is not None:
        title = f'{tcr_group} (ρ={correlation:.2f})'
    else:
        title = tcr_group
    ax.set_title(title, fontsize=7, fontweight='bold', pad=2)

    # Mark reference amino acids with black border
    ref_positions = get_reference_aa_positions()
    for row_idx, col_idx in ref_positions:
        rect = patches.Rectangle(
            (col_idx - 0.5, row_idx - 0.5), 1, 1,
            linewidth=1, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)

    return im


def create_tcr_info_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    tcr_group: str = 'SVAR.90'
):
    """
    Create TCR information panel (like SVAR-90 in the reference image).

    Args:
        ax: Matplotlib axes
        df: DataFrame with TCR info
        tcr_group: TCR group to display info for
    """
    ax.axis('off')

    # Get TCR info from data
    tcr_df = df[df['TCR_Group'] == tcr_group]

    if len(tcr_df) > 0:
        row = tcr_df.iloc[0]
        info_text = f"""
TRAV:    {row.get('Va', 'N/A')}
TRAJ:    {row.get('Ja', 'N/A')}
CDR3α:   {row.get('CDR3a', 'N/A')}
TRBV:    {row.get('Vb', 'N/A')}
TRBJ:    {row.get('Jb', 'N/A')}
CDR3β:   {row.get('CDR3b', 'N/A')}
"""
    else:
        info_text = "TCR info not available"

    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='center',
            fontfamily='monospace')
    ax.set_title(tcr_group, fontsize=9, fontweight='bold')


def create_model_heatmap_figure(
    df: pd.DataFrame,
    model_key: str,
    output_path: Path
):
    """
    Create complete heatmap figure for one model.

    Args:
        df: Merged DataFrame with predictions
        model_key: Model key (e.g., 'ERGO')
        output_path: Output file path
    """
    setup_plot_style()
    config = get_config()

    # Find prediction column for this model
    prob_col = None
    for col in df.columns:
        if '_Prob' in col:
            model_normalized = model_key.lower().replace('-', '').replace('.', '')
            col_normalized = col.lower().replace('-', '').replace('.', '')
            if model_normalized in col_normalized:
                prob_col = col
                break

    if prob_col is None:
        logger.warning(f"No prediction column found for {model_key}")
        return

    # Create figure with 5x5 grid (20 heatmaps + 1 info panel + 4 empty)
    # Smaller figure size for more compact cells
    fig = plt.figure(figsize=(10, 12))

    # Create custom colormap (blue -> white -> red)
    cmap = plt.cm.RdBu_r

    # Layout: 5 rows x 5 columns
    # First 20 TCRs (SVAR.1 to SVAR.85) in a 5x4 grid
    # Last position for SVAR.90 info panel

    tcr_list = TCR_GROUPS[:-1]  # Exclude SVAR.90 for main heatmaps

    # Create subplots with narrower column spacing
    gs = fig.add_gridspec(5, 5, hspace=0.30, wspace=0.12)

    # Plot first 20 TCRs
    for i, tcr_group in enumerate(tcr_list):
        row = i // 4
        col = i % 4

        ax = fig.add_subplot(gs[row, col])
        matrix, status_matrix = prepare_heatmap_data(df, tcr_group, prob_col)
        # Calculate correlation with log2foldchange for this TCR
        corr = calculate_tcr_correlation(df, tcr_group, prob_col)
        im = create_single_tcr_heatmap(ax, matrix, status_matrix, tcr_group, cmap, correlation=corr)

    # Add SVAR.90 with both heatmap and info
    # Position: row 4, col 4
    ax_svar90 = fig.add_subplot(gs[4, 4])
    matrix_svar90, status_matrix_svar90 = prepare_heatmap_data(df, 'SVAR.90', prob_col)
    corr_svar90 = calculate_tcr_correlation(df, 'SVAR.90', prob_col)
    im = create_single_tcr_heatmap(ax_svar90, matrix_svar90, status_matrix_svar90, 'SVAR.90', cmap, correlation=corr_svar90)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Pred Prob', fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # Add legend for grey cells
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', edgecolor='black', label='Unstable/Wild Type')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=8,
               bbox_to_anchor=(0.90, 0.05))

    # Main title
    fig.suptitle(f'{format_model_name(model_key)} - Fingerprinting Heatmaps',
                 fontsize=11, fontweight='bold', y=0.98)

    # Save
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_all_heatmaps(mode: str):
    """
    Generate fingerprinting heatmaps for all models.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)

    logger.info(f"Generating fingerprinting heatmaps for {mode}...")

    # Ensure output directory exists
    paths.HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Load fingerprinting merged data
    df = load_merged_data(paths, 'fingerprinting')
    if df is None:
        logger.warning("FingerPrinting merged data not found")
        return

    # Don't filter - show all entries, but mark unstable/wild type as grey
    if 'Antigen_Status' in df.columns:
        status_counts = df['Antigen_Status'].value_counts()
        logger.info(f"  Antigen_Status distribution: {status_counts.to_dict()}")

    if 'TCR_Group' not in df.columns:
        logger.warning("TCR_Group column not found in data")
        return

    # Generate heatmap for each model
    for model_key in MODEL_ORDER:
        logger.info(f"  Generating heatmap for {model_key}...")
        output_path = paths.get_heatmap_file(model_key)
        create_model_heatmap_figure(df, model_key, output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fingerprinting heatmaps")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Generate for specific model only")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_heatmap", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_all_heatmaps(mode)


if __name__ == "__main__":
    main()
