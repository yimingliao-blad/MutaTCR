"""
TCR Model Heatmaps for TCRP Benchmark V4.

Generates 3x3 grid heatmaps for each TCR showing:
- 8 subplots for each model (ERGO, ERGO2, NetTCR, NetTCR22, TITAN, EPACT, PanPep, SCEPTR)
- 1 subplot for original log2foldchange values
- Correlation coefficient in each model's title

Each heatmap: 20 amino acids (rows) x 9 positions (columns)
Color scheme: Blue (low) -> White (mid) -> Red (high)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional
import logging
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, setup_logging
from src.visualization.utils import setup_plot_style, save_figure

logger = logging.getLogger(__name__)

# Model order for consistent display
MODELS = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']

# Reference peptide and amino acid order
REFERENCE_PEPTIDE = "YLQPRTFLL"
# Amino acids in order from top to bottom
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def get_position_and_aa(peptide: str):
    """Extract position and substituted amino acid from peptide variant."""
    if len(peptide) != len(REFERENCE_PEPTIDE):
        return (0, '')

    diffs = []
    for i, (p, r) in enumerate(zip(peptide.upper(), REFERENCE_PEPTIDE)):
        if p != r:
            diffs.append((i + 1, p))  # 1-indexed position

    if len(diffs) == 1:
        return diffs[0]
    elif len(diffs) == 0:
        return (0, 'WT')
    return (0, '')


def prepare_matrix(df: pd.DataFrame, tcr_group: str, value_col: str):
    """
    Prepare 20x9 heatmap matrix for a single TCR group.

    Args:
        df: DataFrame with peptide data
        tcr_group: TCR group name
        value_col: Column to use for values (model prob or log2foldchange)

    Returns:
        Tuple of (value_matrix, status_matrix) - 20x9 numpy arrays
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

        pos, aa = get_position_and_aa(peptide)

        if pos > 0 and aa in aa_to_idx:
            row_idx = aa_to_idx[aa]
            col_idx = pos - 1
            matrix[row_idx, col_idx] = value
            status_matrix[row_idx, col_idx] = status

    return matrix, status_matrix


def get_reference_positions():
    """Get (row, col) positions of reference amino acids."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    positions = []
    for col_idx, aa in enumerate(REFERENCE_PEPTIDE):
        if aa in aa_to_idx:
            positions.append((aa_to_idx[aa], col_idx))
    return positions


def load_merged_fingerprinting(paths) -> Optional[pd.DataFrame]:
    """Load merged fingerprinting data with all model predictions."""
    merged_file = paths.MERGED_DIR / "fingerprinting_all_models.csv"

    if not merged_file.exists():
        # Try to create merged file from individual predictions
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
                    model_df = df[['ID', 'Prediction_Prob']].rename(
                        columns={'Prediction_Prob': f'{model}_Prob'})
                    base_df = base_df.merge(model_df, on='ID', how='inner')

        if base_df is not None:
            # Save merged file
            paths.MERGED_DIR.mkdir(parents=True, exist_ok=True)
            base_df.to_csv(merged_file, index=False)
            # Don't filter - we'll show unstable/wild type as grey
            return base_df
        return None

    df = pd.read_csv(merged_file)
    # Don't filter - we'll show unstable/wild type as grey
    return df


def create_tcr_heatmap_figure(df: pd.DataFrame, tcr_group: str, output_path: Path):
    """
    Create a 3x3 grid figure for one TCR showing 8 models + log2foldchange.
    """
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.cm as cm

    setup_plot_style()

    # Create custom colormap (blue -> white -> red)
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('custom_rwb', colors, N=256)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 14))
    axes = axes.flatten()

    # Get reference positions for black borders
    ref_positions = get_reference_positions()

    # Get TCR-specific data for correlation calculation (only valid entries)
    tcr_df = df[df['TCR_Group'] == tcr_group].copy()
    tcr_df_valid = tcr_df[tcr_df['Antigen_Status'] == 'valid'] if 'Antigen_Status' in tcr_df.columns else tcr_df

    # Plot 8 models
    for i, model in enumerate(MODELS):
        ax = axes[i]
        prob_col = f'{model}_Prob'

        if prob_col not in df.columns:
            ax.set_visible(False)
            continue

        matrix, status_matrix = prepare_matrix(df, tcr_group, prob_col)

        # Calculate correlation with log2foldchange for this TCR (using valid entries only)
        corr_str = ""
        if prob_col in tcr_df_valid.columns and 'log2foldchange' in tcr_df_valid.columns:
            pred_vals = tcr_df_valid[prob_col].values
            log2fc_vals = tcr_df_valid['log2foldchange'].values
            mask = ~(np.isnan(pred_vals) | np.isnan(log2fc_vals))
            if mask.sum() > 3:
                try:
                    r, _ = spearmanr(pred_vals[mask], log2fc_vals[mask])
                    corr_str = f" (ρ={r:.2f})"
                except:
                    pass

        # Plot heatmap - predictions are 0-1
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        # Overlay grey for unstable/wild type
        for row_idx in range(20):
            for col_idx in range(9):
                status = status_matrix[row_idx, col_idx]
                if status in ['unstable', 'wild type']:
                    rect = patches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5), 1, 1,
                        linewidth=0, facecolor='#808080', alpha=1.0
                    )
                    ax.add_patch(rect)

        # Set labels
        ax.set_xticks(range(9))
        ax.set_xticklabels(range(1, 10), fontsize=8)
        ax.set_yticks(range(20))
        ax.set_yticklabels(AMINO_ACIDS, fontsize=7)

        # Title with correlation
        ax.set_title(f'{model}{corr_str}', fontsize=11, fontweight='bold')

        # Mark reference amino acids with black border
        for row_idx, col_idx in ref_positions:
            rect = patches.Rectangle(
                (col_idx - 0.5, row_idx - 0.5), 1, 1,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax.add_patch(rect)

    # Plot log2foldchange in the 9th subplot
    ax = axes[8]
    matrix, status_matrix = prepare_matrix(df, tcr_group, 'log2foldchange')

    # For log2foldchange: -1 (most blue) -> 1 (white) -> 5 (most red)
    log2fc_vmin = -1
    log2fc_vmax = 5
    log2fc_vcenter = 1
    log2fc_norm = TwoSlopeNorm(vmin=log2fc_vmin, vcenter=log2fc_vcenter, vmax=log2fc_vmax)

    # Clip values for visualization
    matrix_clipped = np.clip(matrix, log2fc_vmin, log2fc_vmax)

    im_log2 = ax.imshow(matrix_clipped, cmap=cmap, norm=log2fc_norm, aspect='auto')

    # Overlay grey for unstable/wild type
    for row_idx in range(20):
        for col_idx in range(9):
            status = status_matrix[row_idx, col_idx]
            if status in ['unstable', 'wild type']:
                rect = patches.Rectangle(
                    (col_idx - 0.5, row_idx - 0.5), 1, 1,
                    linewidth=0, facecolor='#808080', alpha=1.0
                )
                ax.add_patch(rect)

    ax.set_xticks(range(9))
    ax.set_xticklabels(range(1, 10), fontsize=8)
    ax.set_yticks(range(20))
    ax.set_yticklabels(AMINO_ACIDS, fontsize=7)
    ax.set_title('log2 Fold Change', fontsize=11, fontweight='bold')

    # Mark reference amino acids
    for row_idx, col_idx in ref_positions:
        rect = patches.Rectangle(
            (col_idx - 0.5, row_idx - 0.5), 1, 1,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar1 = fig.colorbar(im, cax=cbar_ax1)
    cbar1.set_label('Prediction\nProbability', fontsize=9)
    cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Log2FC colorbar with TwoSlopeNorm
    cbar_ax2 = fig.add_axes([0.92, 0.12, 0.02, 0.3])
    sm_log2fc = cm.ScalarMappable(cmap=cmap, norm=log2fc_norm)
    sm_log2fc.set_array([])
    cbar2 = fig.colorbar(sm_log2fc, cax=cbar_ax2)
    cbar2.set_label('Log2 FC\npMHC Binding', fontsize=9)
    cbar2.set_ticks([-1, 0, 1, 2, 3, 4, 5])

    # Add legend for grey cells
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', edgecolor='black', label='Unstable/Wild Type')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
               bbox_to_anchor=(0.88, 0.02))

    # Main title
    fig.suptitle(f'{tcr_group}: Model Predictions vs Original log2FC',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_tcr_model_heatmaps(mode: str = "exact_match"):
    """
    Generate 3x3 grid heatmaps for each TCR.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)

    logger.info(f"Generating TCR model heatmaps for {mode}...")

    # Ensure output directory exists
    vis_dir = paths.VISUALIZATIONS_DIR / "tcr_model_heatmaps"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load merged fingerprinting data
    df = load_merged_fingerprinting(paths)
    if df is None:
        logger.warning("Could not load fingerprinting predictions")
        return

    if 'TCR_Group' not in df.columns:
        logger.warning("TCR_Group column not found in data")
        return

    # Get unique TCR groups
    tcr_groups = sorted(df['TCR_Group'].unique())
    logger.info(f"Found {len(tcr_groups)} TCR groups")

    # Create heatmap for each TCR
    for tcr in tcr_groups:
        output_path = vis_dir / f"{tcr}_8models_heatmap.png"
        create_tcr_heatmap_figure(df, tcr, output_path)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate TCR model heatmaps")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("tcr_model_heatmaps", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_tcr_model_heatmaps(mode)


if __name__ == "__main__":
    main()
