"""
UMI Scatter Plot Visualization for TCRP Benchmark V2.

Generates scatter plots of UMI count vs predicted probability.
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


def create_umi_scatter_combined(df: pd.DataFrame, output_path: Path):
    """
    Create combined UMI scatter plot (mean across all models).

    Args:
        df: Merged DataFrame with UMI and predictions
        output_path: Output file path
    """
    setup_plot_style()
    config = get_config()

    # Use all predicted records, excluding T1D group
    df_plot = df[df['Epitope_Group'] != 'T1D'].copy() if 'Epitope_Group' in df.columns else df.copy()

    if len(df_plot) == 0:
        logger.warning("No data for UMI scatter")
        return

    # Calculate mean prediction across all models
    model_cols = [f'{m}_Prob' for m in get_all_model_keys() if f'{m}_Prob' in df.columns]
    if len(model_cols) == 0:
        logger.warning("No model prediction columns found")
        return

    df_plot['Mean_Prob'] = df_plot[model_cols].mean(axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define high-contrast colors for TetTCR epitope groups
    group_colors = {
        'Viral': '#1f77b4',  # Strong blue
        'Other': '#d62728',  # Bright red
        'self': '#ff7f0e',   # Bright orange
        'T1D': '#17becf',    # Cyan
    }

    # Plot by epitope group
    groups = df_plot['Epitope_Group'].unique()
    for group in sorted(groups):  # Consistent order
        group_df = df_plot[df_plot['Epitope_Group'] == group]
        color = group_colors.get(group, '#333333')
        ax.scatter(
            group_df['Mean_Prob'],
            group_df['UMI'],
            c=color,
            alpha=0.6,
            s=30,
            label=f'{group} (n={len(group_df)})',
            edgecolor='white',
            linewidth=0.3
        )

    # Add correlation annotation
    add_correlation_annotation(ax, df_plot['Mean_Prob'].values, df_plot['UMI'].values)

    # Add regression line
    add_regression_line(ax, df_plot['Mean_Prob'].values, df_plot['UMI'].values, color='black')

    # Formatting
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('UMI Count')
    ax.set_title('TetTCR-SeqHD: UMI vs Prediction (Combined Models)')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left')

    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_umi_scatter_8models(df: pd.DataFrame, output_path: Path):
    """
    Create 8-model UMI scatter plot grid.

    Args:
        df: Merged DataFrame with UMI and predictions
        output_path: Output file path
    """
    setup_plot_style()
    config = get_config()

    # Use all predicted records, excluding T1D group
    df_plot = df[df['Epitope_Group'] != 'T1D'].copy() if 'Epitope_Group' in df.columns else df.copy()

    if len(df_plot) == 0:
        logger.warning("No data for UMI scatter")
        return

    # Create 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Define high-contrast colors for TetTCR epitope groups
    group_colors = {
        'Viral': '#1f77b4',  # Strong blue
        'Other': '#d62728',  # Bright red
        'self': '#ff7f0e',   # Bright orange
        'T1D': '#17becf',    # Cyan
    }

    for i, model_key in enumerate(MODEL_ORDER):
        ax = axes[i]
        prob_col = f'{model_key.replace("-", "").replace(".", "")}_Prob'

        # Try different column name formats
        if prob_col not in df_plot.columns:
            prob_col = f'{model_key}_Prob'
        if prob_col not in df_plot.columns:
            # Try to find matching column
            for col in df_plot.columns:
                if model_key.lower().replace('-', '').replace('.', '') in col.lower():
                    prob_col = col
                    break

        if prob_col not in df_plot.columns:
            ax.text(0.5, 0.5, f'No data for {model_key}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(format_model_name(model_key))
            continue

        # Plot by epitope group with high-contrast colors
        groups = df_plot['Epitope_Group'].unique()
        for group in sorted(groups):  # Consistent order
            group_df = df_plot[df_plot['Epitope_Group'] == group]
            color = group_colors.get(group, '#333333')
            ax.scatter(
                group_df[prob_col],
                group_df['UMI'],
                c=color,
                alpha=0.6,
                s=20,
                label=group,
                edgecolor='white',
                linewidth=0.3
            )

        # Add correlation
        x = df_plot[prob_col].values
        y = df_plot['UMI'].values
        add_correlation_annotation(ax, x, y, loc='upper left')

        # Formatting
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('UMI Count')
        ax.set_title(format_model_name(model_key))
        ax.set_xlim(0, 1)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def create_umi_scatter_2models(df: pd.DataFrame, output_path: Path):
    """
    Create 2-model UMI scatter plot (NetTCR-2.2 and EPACT).

    Args:
        df: Merged DataFrame with UMI and predictions
        output_path: Output file path
    """
    setup_plot_style()

    # Use all predicted records, excluding T1D group
    df_plot = df[df['Epitope_Group'] != 'T1D'].copy() if 'Epitope_Group' in df.columns else df.copy()

    if len(df_plot) == 0:
        logger.warning("No data for UMI scatter")
        return

    # Create 1x2 grid for NetTCR-2.2 and EPACT
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Define high-contrast colors for TetTCR epitope groups
    group_colors = {
        'Viral': '#1f77b4',  # Strong blue
        'Other': '#d62728',  # Bright red
        'self': '#ff7f0e',   # Bright orange
    }

    # Models to plot
    models_to_plot = [('NetTCR-2.2', 'NetTCR22'), ('EPACT', 'EPACT')]

    for i, (display_name, model_key) in enumerate(models_to_plot):
        ax = axes[i]

        # Find the correct column name
        prob_col = None
        for col in df_plot.columns:
            if '_Prob' in col and model_key.lower().replace('-', '').replace('.', '') in col.lower().replace('-', '').replace('.', ''):
                prob_col = col
                break

        if prob_col is None:
            ax.text(0.5, 0.5, f'No data for {display_name}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_name)
            continue

        # Plot by epitope group with high-contrast colors
        groups = df_plot['Epitope_Group'].unique()
        for group in sorted(groups):
            group_df = df_plot[df_plot['Epitope_Group'] == group]
            color = group_colors.get(group, '#333333')
            ax.scatter(
                group_df[prob_col],
                group_df['UMI'],
                c=color,
                alpha=0.6,
                s=40,
                label=f'{group} (n={len(group_df)})',
                edgecolor='white',
                linewidth=0.3
            )

        # Add correlation and regression line
        x = df_plot[prob_col].values
        y = df_plot['UMI'].values
        add_correlation_annotation(ax, x, y, loc='upper left')
        add_regression_line(ax, x, y, color='black')

        # Formatting
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('UMI Count', fontsize=12)
        ax.set_title(display_name, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('TetTCR-SeqHD: UMI vs Prediction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")


def generate_all_umi_plots(mode: str):
    """
    Generate all UMI scatter plots for a mode.

    Args:
        mode: "exact_match" or "fuzzy_match"
    """
    paths = get_paths(mode=mode)

    logger.info(f"Generating UMI scatter plots for {mode}...")

    # Load TetTCR-SeqHD merged data
    df = load_merged_data(paths, 'tettcr')
    if df is None:
        logger.warning("TetTCR merged data not found")
        return

    if 'UMI' not in df.columns:
        logger.warning("UMI column not found in data")
        return

    # Combined plot
    output_path = paths.get_visualization_file('umi_scatter_combined.png')
    create_umi_scatter_combined(df, output_path)

    # 8-model grid
    output_path = paths.get_visualization_file('umi_scatter_8models.png')
    create_umi_scatter_8models(df, output_path)

    # 2-model plot (NetTCR-2.2 and EPACT)
    output_path = paths.get_visualization_file('umi_scatter_2models.png')
    create_umi_scatter_2models(df, output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate UMI scatter plots")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match", "both"],
                        default="both", help="Deduplication mode")
    args = parser.parse_args()

    logger = setup_logging("umi_scatter", level="INFO")

    if args.mode == "both":
        modes = ["exact_match", "fuzzy_match"]
    else:
        modes = [args.mode]

    for mode in modes:
        generate_all_umi_plots(mode)


if __name__ == "__main__":
    main()
