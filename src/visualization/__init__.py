"""
Visualization modules for TCRP Benchmark V4.

Available visualizations:
- auc_boxplot: Per-peptide partial AUC box plots
- umi_scatter: UMI vs prediction probability scatter plots
- signal_fraction_scatter: UMI_Fraction vs prediction scatter plots
- log2fc_scatter: log2foldchange vs prediction scatter plots
- fingerprinting_heatmap: Per-TCR binding probability heatmaps
- fingerprinting_correlation: TCR × Model correlation heatmaps
- fingerprinting_dual_heatmap: Dual-panel (log2FC + prob) heatmaps per model
- fingerprinting_tcr_correlation: Per-TCR correlation heatmaps (8 models)
- fingerprinting_pauc_boxplot: Fingerprinting pAUC by R5-APL/Non-5-Pos groups
- fingerprinting_cutoff_analysis: Log2 cutoff sensitivity analysis (AUC/pAUC)
- fingerprinting_position_correlation: Position-based Spearman correlation (3x3 grids)
- tcr_model_heatmaps: 3x3 grid heatmaps per TCR (8 models + log2FC)
- comparison: Cross-mode (exact vs fuzzy) comparison plots
- compound_figure: Multi-panel publication figure combining multiple visualizations
"""

from .auc_boxplot import generate_all_boxplots
from .umi_scatter import generate_all_umi_plots
from .signal_fraction_scatter import generate_all_signal_plots
from .log2fc_scatter import generate_log2fc_plots
from .fingerprinting_heatmap import generate_all_heatmaps
from .fingerprinting_correlation import generate_fingerprinting_correlation_plots
from .fingerprinting_dual_heatmap import generate_all_dual_heatmaps
from .fingerprinting_tcr_correlation import generate_tcr_correlation_visualizations
from .fingerprinting_pauc_boxplot import create_fingerprinting_pauc_boxplot
from .fingerprinting_cutoff_analysis import generate_cutoff_analysis
from .fingerprinting_position_correlation import create_position_correlation_plot
from .tcr_model_heatmaps import generate_tcr_model_heatmaps
from .comparison import generate_comparison
from .compound_figure import generate_compound_figure

__all__ = [
    "generate_all_boxplots",
    "generate_all_umi_plots",
    "generate_all_signal_plots",
    "generate_log2fc_plots",
    "generate_all_heatmaps",
    "generate_fingerprinting_correlation_plots",
    "generate_all_dual_heatmaps",
    "generate_tcr_correlation_visualizations",
    "create_fingerprinting_pauc_boxplot",
    "generate_cutoff_analysis",
    "create_position_correlation_plot",
    "generate_tcr_model_heatmaps",
    "generate_comparison",
    "generate_compound_figure",
]
