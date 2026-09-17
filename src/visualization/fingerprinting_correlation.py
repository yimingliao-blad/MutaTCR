"""
Fingerprinting Correlation Visualizations for TCRP Benchmark V4.

Generates:
- TCR × Model correlation heatmap (prediction vs log2FC)
- Summary heatmap (mean per TCR)
- Correlation matrix heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import logging
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import setup_plot_style, save_figure

logger = logging.getLogger(__name__)

# Model order for consistent display
MODELS = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']


def load_fingerprinting_predictions(paths) -> Optional[pd.DataFrame]:
    """Load fingerprinting predictions from all 8 models."""
    base_df = None
    output_dir = paths.MODE_OUTPUT_DIR

    for model in MODELS:
        pred_file = output_dir / model / "predictions" / "fingerprinting_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            logger.info(f"{model}: {len(df)} samples")

            if base_df is None:
                cols_to_keep = ['ID', 'Peptide', 'CDR3b', 'TCR_Group', 'Label', 'log2foldchange']
                cols_available = [c for c in cols_to_keep if c in df.columns]
                base_df = df[cols_available].copy()
                base_df[model] = df['Prediction_Prob'].values
            else:
                model_df = df[['ID', 'Prediction_Prob']].rename(columns={'Prediction_Prob': model})
                base_df = base_df.merge(model_df, on='ID', how='inner')
        else:
            logger.warning(f"Prediction file not found: {pred_file}")

    if base_df is not None:
        logger.info(f"Final merged data: {len(base_df)} samples")

        # Filter to only 'valid' Antigen_Status entries
        if 'Antigen_Status' in base_df.columns:
            base_df = base_df[base_df['Antigen_Status'] == 'valid']
            logger.info(f"Filtered to {len(base_df)} valid entries")

    return base_df


def create_tcr_model_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create a heatmap showing correlation between predictions and log2FC for each TCR × Model.

    Features:
    - TCRs on y-axis, Models on x-axis
    - ALL TCRs row showing overall correlation
    - Mean |ρ| row showing absolute mean correlation per model
    - Mean |ρ| column showing absolute mean correlation per TCR
    """
    setup_plot_style()

    tcr_groups = sorted(df['TCR_Group'].unique())

    # Calculate correlation matrix: TCRs (rows) × Models (columns)
    corr_matrix = np.zeros((len(tcr_groups), len(MODELS)))

    for i, tcr in enumerate(tcr_groups):
        tcr_df = df[df['TCR_Group'] == tcr]
        log2fc_vals = tcr_df['log2foldchange'].values

        for j, model in enumerate(MODELS):
            pred_vals = tcr_df[model].values
            mask = ~(np.isnan(pred_vals) | np.isnan(log2fc_vals))

            if mask.sum() > 3:
                try:
                    r, _ = spearmanr(pred_vals[mask], log2fc_vals[mask])
                    corr_matrix[i, j] = r if not np.isnan(r) else 0.0
                except:
                    corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = 0.0

    # Calculate overall correlation (all TCRs combined) for each model
    overall_corrs = []
    log2fc_all = df['log2foldchange'].values
    for model in MODELS:
        pred_all = df[model].values
        mask = ~(np.isnan(pred_all) | np.isnan(log2fc_all))
        if mask.sum() > 3:
            try:
                r, _ = spearmanr(pred_all[mask], log2fc_all[mask])
                overall_corrs.append(r if not np.isnan(r) else 0.0)
            except:
                overall_corrs.append(0.0)
        else:
            overall_corrs.append(0.0)

    # Calculate mean absolute correlation per model (across all TCRs)
    abs_mean_corrs = np.abs(corr_matrix).mean(axis=0)

    # Calculate mean absolute correlation per TCR (across all models)
    abs_mean_corr_per_tcr = np.abs(corr_matrix).mean(axis=1)

    # Add overall row and absolute mean row to matrix
    corr_matrix_with_overall = np.vstack([corr_matrix, [overall_corrs], [abs_mean_corrs]])

    # Calculate absolute mean for the summary rows too
    overall_abs_mean = np.mean(np.abs(overall_corrs))
    abs_mean_abs_mean = np.mean(abs_mean_corrs)

    # Add mean |r| column (absolute mean across all models)
    mean_col = np.concatenate([abs_mean_corr_per_tcr, [overall_abs_mean], [abs_mean_abs_mean]])
    corr_matrix_with_mean = np.column_stack([corr_matrix_with_overall, mean_col])

    row_labels = tcr_groups + ['ALL TCRs', 'Mean |ρ|']
    col_labels = MODELS + ['Mean |ρ|']

    # Create DataFrame for heatmap
    corr_df = pd.DataFrame(corr_matrix_with_mean, index=row_labels, columns=col_labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr_df, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                xticklabels=col_labels, yticklabels=row_labels,
                linewidths=0.5, cbar_kws={'label': 'Spearman Correlation (ρ)'})

    # Add horizontal line to separate summary rows
    ax.axhline(y=len(tcr_groups), color='black', linewidth=2)

    # Add vertical line to separate Mean column
    ax.axvline(x=len(MODELS), color='black', linewidth=2)

    ax.set_title('Fingerprinting: Prediction vs log2FC Correlation\n(per TCR × Model)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('TCR', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Highlight the summary row labels
    ytick_labels = ax.get_yticklabels()
    ytick_labels[-1].set_fontweight('bold')
    ytick_labels[-2].set_fontweight('bold')
    ax.set_yticklabels(ytick_labels)

    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_summary_heatmap(df: pd.DataFrame, output_path: Path):
    """Create a combined heatmap showing mean predictions per TCR across all models."""
    setup_plot_style()

    # Calculate mean prediction per TCR for each model
    tcr_means = df.groupby('TCR_Group')[MODELS + ['log2foldchange', 'Label']].mean()
    tcr_means = tcr_means.sort_index()

    # Calculate overall correlation between each model and log2foldchange
    model_correlations = {}
    log2fc_values = df['log2foldchange'].values
    for model in MODELS:
        pred_values = df[model].values
        mask = ~(np.isnan(pred_values) | np.isnan(log2fc_values))
        if mask.sum() > 3:
            r, _ = spearmanr(pred_values[mask], log2fc_values[mask])
            model_correlations[model] = r
        else:
            model_correlations[model] = np.nan

    # Create x-axis labels with correlation values
    model_labels = [f'{m}\n(ρ={model_correlations[m]:.2f})' if not np.isnan(model_correlations[m]) else m
                    for m in MODELS]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': [8, 1.5], 'wspace': 0.05})

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Model predictions heatmap
    model_data = tcr_means[MODELS].values
    sns.heatmap(model_data, ax=ax1, cmap=cmap, center=0.5, vmin=0, vmax=1,
                xticklabels=model_labels, yticklabels=tcr_means.index,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                cbar_kws={'label': 'Mean Prediction Probability', 'shrink': 0.7})
    ax1.set_title('Mean Predictions per TCR (with overall log2FC correlation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('TCR Group')
    ax1.tick_params(axis='x', rotation=45)

    # Log2foldchange heatmap
    log2fc_data = tcr_means[['log2foldchange']].values
    sns.heatmap(log2fc_data, ax=ax2, cmap=cmap, center=0,
                xticklabels=['Mean\nlog2FC'], yticklabels=False,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                cbar_kws={'label': 'Mean log2 Fold Change', 'shrink': 0.7})
    ax2.set_title('Original', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle('Fingerprinting: All Models vs Original log2FC (Mean per TCR)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_correlation_matrix_heatmap(df: pd.DataFrame, output_path: Path):
    """Create a heatmap showing correlation between models and log2foldchange."""
    setup_plot_style()

    # Calculate correlations (Spearman)
    corr_cols = MODELS + ['log2foldchange', 'Label']
    corr_matrix = df[corr_cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr_matrix, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt='.3f', annot_kws={'size': 9},
                xticklabels=corr_cols, yticklabels=corr_cols,
                square=True, linewidths=0.5)

    ax.set_title('Fingerprinting: Correlation Matrix\n(Model Predictions vs log2FC and Labels)',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_fingerprinting_correlation_plots(mode: str = "exact_match"):
    """
    Generate all fingerprinting correlation visualizations.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)

    logger.info(f"Generating fingerprinting correlation plots for {mode}...")

    # Ensure output directory exists
    vis_dir = paths.VISUALIZATIONS_DIR / "fingerprinting_heatmaps"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load fingerprinting predictions from all models
    df = load_fingerprinting_predictions(paths)
    if df is None:
        logger.warning("Could not load fingerprinting predictions")
        return

    if 'TCR_Group' not in df.columns:
        logger.warning("TCR_Group column not found in data")
        return

    if 'log2foldchange' not in df.columns:
        logger.warning("log2foldchange column not found in data")
        return

    # 1. TCR × Model correlation heatmap
    logger.info("  Creating TCR × Model correlation heatmap...")
    create_tcr_model_correlation_heatmap(df, vis_dir / "fingerprinting_tcr_model_correlation.png")

    # 2. Summary heatmap (mean per TCR)
    logger.info("  Creating summary heatmap...")
    create_summary_heatmap(df, vis_dir / "fingerprinting_summary_heatmap.png")

    # 3. Correlation matrix heatmap
    logger.info("  Creating correlation matrix heatmap...")
    create_correlation_matrix_heatmap(df, vis_dir / "fingerprinting_correlation_heatmap.png")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate fingerprinting correlation plots")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("fingerprinting_correlation", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_fingerprinting_correlation_plots(mode)


if __name__ == "__main__":
    main()
