"""
AUC Box Plot Visualization for TCRP Benchmark V2.

Generates per-peptide partial AUC (0.1) box plots for model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import argparse
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import (
    setup_plot_style, create_figure, save_figure,
    get_model_colors, format_model_name
)

logger = logging.getLogger(__name__)


def load_per_peptide_auc(paths, dataset: str) -> Optional[pd.DataFrame]:
    """Load per-peptide AUC data for a dataset."""
    auc_file = paths.AGGREGATED_DIR / f"{dataset}_per_peptide_auc.csv"
    if auc_file.exists():
        return pd.read_csv(auc_file)
    return None


def create_auc_boxplot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    epitope_group: Optional[str] = None
):
    """
    Create AUC box plot for model comparison.

    Args:
        df: DataFrame with per-peptide AUC values
        title: Plot title
        output_path: Output file path
        epitope_group: Optional filter for epitope group
    """
    setup_plot_style()
    config = get_config()

    # Get model columns
    model_cols = [col for col in df.columns if col.endswith('_pAUC')]
    models = [col.replace('_pAUC', '') for col in model_cols]

    # Prepare data for plotting
    plot_data = []
    for model in models:
        col = f'{model}_pAUC'
        if col in df.columns:
            values = df[col].dropna().values
            display_name = format_model_name(model)
            for v in values:
                plot_data.append({
                    'Model': display_name,
                    'Partial AUC (0.1)': v
                })

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        logger.warning(f"No data for boxplot: {title}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    model_order = [format_model_name(m) for m in MODEL_ORDER if format_model_name(m) in plot_df['Model'].unique()]

    colors = get_model_colors()
    palette = {format_model_name(k): v for k, v in colors.items()}

    sns.boxplot(
        data=plot_df,
        x='Model',
        y='Partial AUC (0.1)',
        order=model_order,
        palette=palette,
        ax=ax,
        width=0.6,
        fliersize=3
    )

    # Add macro average markers (diamonds)
    means = plot_df.groupby('Model')['Partial AUC (0.1)'].mean()
    for i, model in enumerate(model_order):
        if model in means.index:
            ax.scatter(i, means[model], marker='D', color='black', s=50, zorder=5)

    # Add baseline line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')

    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('Partial AUC (FPR ≤ 0.1)')
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right')

    # Add legend for diamond marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=8, label='Macro Average'),
        Line2D([0], [0], color='gray', linestyle='--', label='Random Baseline')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def calculate_per_peptide_pauc(df, prob_col, max_fpr=0.1):
    """Calculate per-peptide partial AUC from merged data."""
    from sklearn.metrics import roc_auc_score

    results = []
    for peptide in df['Peptide'].unique():
        peptide_df = df[df['Peptide'] == peptide]
        y_true = peptide_df['Label'].values
        y_score = peptide_df[prob_col].values

        # Remove NaN
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]

        if len(np.unique(y_true)) < 2 or len(y_true) < 2:
            continue

        try:
            pauc = roc_auc_score(y_true, y_score, max_fpr=max_fpr)
            results.append({'Peptide': peptide, 'pAUC': pauc})
        except:
            pass

    return pd.DataFrame(results)


def create_combined_seen_boxplot(mode: str):
    """
    Create combined TetTCR-SeqHD and IMMREP23 performance boxplot.

    Shows TetTCR-SeqHD (Viral), TetTCR-SeqHD (Self), IMMREP23 (Seen), and IMMREP23 (Unseen)
    grouped by model with per-peptide partial AUC values.
    """
    from sklearn.metrics import roc_auc_score
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    paths = get_paths(mode=mode)
    config = get_config()

    # Model configuration
    MODEL_ORDER_LOCAL = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']
    MODEL_DISPLAY = {
        'ERGO': 'ERGO', 'ERGO2': 'ERGO2', 'NetTCR': 'NetTCR2',
        'NetTCR22': 'NetTCR-2.2', 'TITAN': 'TITAN', 'EPACT': 'EPACT',
        'PanPep': 'PanPep', 'SCEPTR': 'SCEPTR'
    }

    # Colors from display_names config - matching signal_fraction_scatter colors
    COLORS = {
        'TetTCR-SeqHD (Viral)': config.get_group_color('tettcr', 'Viral'),   # Blue #0000a2
        'TetTCR-SeqHD (Self)': config.get_group_color('tettcr', 'Self'),     # Yellow #e9c716
        'IMMREP23 (Seen)': config.get_group_color('immrep23', 'Seen'),       # Green #2ca02c
        'IMMREP23 (Unseen)': config.get_group_color('immrep23', 'Unseen')    # Purple #9467bd
    }

    # Load merged data
    tettcr_file = paths.MERGED_DIR / 'tettcr_all_models.csv'
    immrep23_file = paths.MERGED_DIR / 'immrep23_all_models.csv'

    if not tettcr_file.exists() or not immrep23_file.exists():
        logger.warning("Merged data files not found for combined boxplot")
        return

    tettcr_df = pd.read_csv(tettcr_file)
    immrep23_df = pd.read_csv(immrep23_file)

    # Filter for groups
    tettcr_viral = tettcr_df[tettcr_df['Epitope_Group'] == 'Viral'].copy()
    tettcr_self = tettcr_df[tettcr_df['Epitope_Group'] == 'self'].copy()  # lowercase 'self'
    immrep23_seen = immrep23_df[immrep23_df['Epitope_Group'] == 'seen'].copy()
    immrep23_unseen = immrep23_df[immrep23_df['Epitope_Group'] == 'unseen'].copy()

    # Calculate per-peptide pAUC for each model and dataset
    plot_data = []

    for model in MODEL_ORDER_LOCAL:
        prob_col = f'{model}_Prob'
        display_name = MODEL_DISPLAY.get(model, model)

        # TetTCR Viral
        if prob_col in tettcr_viral.columns:
            pauc_df = calculate_per_peptide_pauc(tettcr_viral, prob_col)
            for _, row in pauc_df.iterrows():
                plot_data.append({
                    'Model': display_name,
                    'Dataset': 'TetTCR-SeqHD (Viral)',
                    'pAUC': row['pAUC'],
                    'Peptide': row['Peptide']
                })

        # TetTCR Self
        if prob_col in tettcr_self.columns:
            pauc_df = calculate_per_peptide_pauc(tettcr_self, prob_col)
            for _, row in pauc_df.iterrows():
                plot_data.append({
                    'Model': display_name,
                    'Dataset': 'TetTCR-SeqHD (Self)',
                    'pAUC': row['pAUC'],
                    'Peptide': row['Peptide']
                })

        # IMMREP23 Seen
        if prob_col in immrep23_seen.columns:
            pauc_df = calculate_per_peptide_pauc(immrep23_seen, prob_col)
            for _, row in pauc_df.iterrows():
                plot_data.append({
                    'Model': display_name,
                    'Dataset': 'IMMREP23 (Seen)',
                    'pAUC': row['pAUC'],
                    'Peptide': row['Peptide']
                })

        # IMMREP23 Unseen
        if prob_col in immrep23_unseen.columns:
            pauc_df = calculate_per_peptide_pauc(immrep23_unseen, prob_col)
            for _, row in pauc_df.iterrows():
                plot_data.append({
                    'Model': display_name,
                    'Dataset': 'IMMREP23 (Unseen)',
                    'pAUC': row['pAUC'],
                    'Peptide': row['Peptide']
                })

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        logger.warning("No data for combined boxplot")
        return

    # Create the combined boxplot
    fig, ax = plt.subplots(figsize=(16, 7))

    model_order = [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER_LOCAL]
    datasets = ['TetTCR-SeqHD (Viral)', 'TetTCR-SeqHD (Self)', 'IMMREP23 (Seen)', 'IMMREP23 (Unseen)']

    # Create positions for grouped boxplots
    box_width = 0.2
    group_positions = np.arange(len(model_order))

    # Plot boxplots for each dataset
    for i, dataset in enumerate(datasets):
        subset = plot_df[plot_df['Dataset'] == dataset]

        data_by_model = []
        pos = []
        for j, model in enumerate(model_order):
            model_data = subset[subset['Model'] == model]['pAUC'].values
            if len(model_data) > 0:
                data_by_model.append(model_data)
                pos.append(j + (i - 1.5) * box_width)

        if data_by_model:
            bp = ax.boxplot(data_by_model, positions=pos, widths=box_width * 0.8,
                           patch_artist=True, showfliers=False)  # Hide default fliers

            for patch in bp['boxes']:
                patch.set_facecolor(COLORS[dataset])
                patch.set_alpha(0.7)
            for whisker in bp['whiskers']:
                whisker.set_color('gray')
            for cap in bp['caps']:
                cap.set_color('gray')
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

    # Plot individual peptide pAUC values as scatter points (jittered)
    np.random.seed(42)  # For reproducible jitter
    for i, dataset in enumerate(datasets):
        subset = plot_df[plot_df['Dataset'] == dataset]
        color = COLORS[dataset]

        for j, model in enumerate(model_order):
            model_data = subset[subset['Model'] == model]['pAUC'].values
            if len(model_data) > 0:
                x_pos = j + (i - 1.5) * box_width
                # Add jitter for visibility
                jitter = np.random.uniform(-box_width * 0.25, box_width * 0.25, len(model_data))
                ax.scatter(x_pos + jitter, model_data, c=color, alpha=0.6, s=25,
                          edgecolor='white', linewidth=0.5, zorder=4)

    # Add macro average markers (diamonds)
    for i, dataset in enumerate(datasets):
        subset = plot_df[plot_df['Dataset'] == dataset]
        for j, model in enumerate(model_order):
            model_subset = subset[subset['Model'] == model]
            model_data = model_subset['pAUC']
            if len(model_data) > 0:
                mean_val = model_data.mean()
                x_pos = j + (i - 1.5) * box_width
                ax.scatter(x_pos, mean_val, marker='D', color='white', edgecolor='black',
                          s=50, zorder=5, linewidths=1.5)

    # Add peptide labels for each group
    # Top 5 for Viral, Top 2 for IMMREP23 Seen, Top 1 for Self and Unseen
    label_config = {
        'TetTCR-SeqHD (Viral)': {'top_n': 5, 'x_offsets': [-0.08, -0.04, 0, 0.04, 0.08]},
        'TetTCR-SeqHD (Self)': {'top_n': 1, 'x_offsets': [0]},
        'IMMREP23 (Seen)': {'top_n': 2, 'x_offsets': [-0.04, 0.04]},
        'IMMREP23 (Unseen)': {'top_n': 1, 'x_offsets': [0]},
    }

    for dataset, config in label_config.items():
        dataset_idx = datasets.index(dataset)
        dataset_subset = plot_df[plot_df['Dataset'] == dataset]

        for j, model in enumerate(model_order):
            model_subset = dataset_subset[dataset_subset['Model'] == model]
            if len(model_subset) > 0:
                x_pos = j + (dataset_idx - 1.5) * box_width
                top_n = model_subset.nlargest(config['top_n'], 'pAUC')

                for k, (_, row) in enumerate(top_n.iterrows()):
                    peptide_label = row['Peptide'][:8] if len(row['Peptide']) > 8 else row['Peptide']
                    ax.annotate(peptide_label, (x_pos + config['x_offsets'][k], row['pAUC'] + 0.02),
                               fontsize=5, ha='center', va='bottom', rotation=90,
                               alpha=0.8)

    # Add random baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Formatting
    ax.set_xticks(group_positions)
    ax.set_xticklabels(model_order, fontsize=22)
    ax.set_xlabel('')
    ax.set_ylabel('Partial AUC$_{0.1}$', fontsize=24)
    # No title per updated figure requirements
    ax.set_ylim(0.35, 1.05)
    ax.set_xlim(-0.5, len(model_order) - 0.5)

    # Legend and stats box removed per updated figure requirements

    plt.tight_layout()

    output_path = paths.get_visualization_file('combined_seen_pauc_boxplot.png')
    save_figure(fig, output_path)
    logger.info(f"Saved combined boxplot: {output_path}")


def generate_all_boxplots(mode: str):
    """
    Generate all AUC boxplots for a mode.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)
    config = get_config()

    logger.info(f"Generating AUC boxplots for {mode}...")

    # TetTCR-SeqHD Viral
    df_tettcr = load_per_peptide_auc(paths, 'tettcr')
    if df_tettcr is not None:
        # Filter for Viral epitope group if column exists
        # Note: The per-peptide AUC may not have epitope group column,
        # need to merge with unified data for filtering
        output_path = paths.get_visualization_file('auc_boxplot_tettcr_viral.png')
        create_auc_boxplot(
            df_tettcr,
            'TetTCR-SeqHD (Viral Epitopes) - Partial AUC',
            output_path
        )

        output_path = paths.get_visualization_file('auc_boxplot_tettcr_self.png')
        create_auc_boxplot(
            df_tettcr,
            'TetTCR-SeqHD (Self Epitopes) - Partial AUC',
            output_path
        )

    # IMMREP23 Seen
    df_immrep = load_per_peptide_auc(paths, 'immrep23')
    if df_immrep is not None:
        output_path = paths.get_visualization_file('auc_boxplot_immrep23_seen.png')
        create_auc_boxplot(
            df_immrep,
            'IMMREP23 (Seen Peptides) - Partial AUC',
            output_path
        )

    # Combined seen epitope boxplot (TetTCR Viral + Self + IMMREP23 Seen)
    create_combined_seen_boxplot(mode)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate AUC boxplots")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("auc_boxplot", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_all_boxplots(mode)


if __name__ == "__main__":
    main()
