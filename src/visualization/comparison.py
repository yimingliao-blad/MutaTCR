"""
Cross-Mode Comparison Visualization for TCRP Benchmark V2.

Generates comparison plots between exact_match and fuzzy_match modes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict
import argparse
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, MODEL_ORDER
from src.visualization.utils import (
    setup_plot_style, save_figure, get_model_colors, format_model_name
)

logger = logging.getLogger(__name__)


def load_metrics_summary(mode: str) -> Optional[pd.DataFrame]:
    """Load metrics summary for a mode."""
    paths = get_paths(mode=mode)
    metrics_file = paths.get_metrics_summary_file()
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return None


def create_mode_comparison_plot(output_dir: Path):
    """
    Create side-by-side comparison of exact vs fuzzy match results.

    Args:
        output_dir: Output directory for comparison plots
    """
    setup_plot_style()

    # Load metrics for both modes
    exact_df = load_metrics_summary('exact_match')
    fuzzy_df = load_metrics_summary('fuzzy_match')

    if exact_df is None and fuzzy_df is None:
        logger.warning("No metrics data found for either mode")
        return

    # Prepare comparison data
    comparison_data = []

    datasets = ['tettcr', 'immrep23', 'fingerprinting']

    for dataset in datasets:
        for model in MODEL_ORDER:
            display_name = format_model_name(model)

            # Get exact match value
            if exact_df is not None:
                exact_row = exact_df[(exact_df['Dataset'] == dataset) &
                                     (exact_df['Model'] == display_name)]
                exact_val = exact_row['Macro_pAUC'].values[0] if len(exact_row) > 0 else np.nan
            else:
                exact_val = np.nan

            # Get fuzzy match value
            if fuzzy_df is not None:
                fuzzy_row = fuzzy_df[(fuzzy_df['Dataset'] == dataset) &
                                     (fuzzy_df['Model'] == display_name)]
                fuzzy_val = fuzzy_row['Macro_pAUC'].values[0] if len(fuzzy_row) > 0 else np.nan
            else:
                fuzzy_val = np.nan

            comparison_data.append({
                'Dataset': dataset,
                'Model': display_name,
                'Exact Match': exact_val,
                'Fuzzy Match': fuzzy_val,
                'Difference': exact_val - fuzzy_val if not (np.isnan(exact_val) or np.isnan(fuzzy_val)) else np.nan
            })

    comp_df = pd.DataFrame(comparison_data)

    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = get_model_colors()

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_df = comp_df[comp_df['Dataset'] == dataset]

        x = np.arange(len(MODEL_ORDER))
        width = 0.35

        # Get values in model order
        exact_vals = []
        fuzzy_vals = []
        for model in MODEL_ORDER:
            row = dataset_df[dataset_df['Model'] == format_model_name(model)]
            exact_vals.append(row['Exact Match'].values[0] if len(row) > 0 else np.nan)
            fuzzy_vals.append(row['Fuzzy Match'].values[0] if len(row) > 0 else np.nan)

        # Plot bars (custom palette: blue=#0000a2, yellow=#e9c716)
        bars1 = ax.bar(x - width/2, exact_vals, width, label='Exact Match', color='#0000a2', alpha=0.8)
        bars2 = ax.bar(x + width/2, fuzzy_vals, width, label='Fuzzy Match', color='#e9c716', alpha=0.8)

        # Formatting
        ax.set_xlabel('')
        ax.set_ylabel('Macro pAUC (0.1)')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels([format_model_name(m) for m in MODEL_ORDER], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        if i == 0:
            ax.legend()

    plt.suptitle('Exact Match vs Fuzzy Match Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'exact_vs_fuzzy_comparison.png'
    save_figure(fig, output_path)
    logger.info(f"Saved: {output_path}")

    # Save comparison table
    table_path = output_dir / 'summary_tables' / 'mode_comparison.csv'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(table_path, index=False)
    logger.info(f"Saved: {table_path}")


def generate_comparison():
    """Generate all comparison visualizations."""
    paths = get_paths(mode='exact_match')  # Mode doesn't matter for comparison dir
    output_dir = paths.COMPARISON_DIR

    logger.info("Generating comparison visualizations...")

    output_dir.mkdir(parents=True, exist_ok=True)

    create_mode_comparison_plot(output_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    args = parser.parse_args()

    logger = setup_logging("comparison", level="INFO")

    generate_comparison()


if __name__ == "__main__":
    main()
