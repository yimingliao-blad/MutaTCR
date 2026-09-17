"""
Visualization utilities for TCRP Benchmark V2.

Common plotting functions and style configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_config


def setup_plot_style():
    """Set up consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def get_model_colors() -> dict:
    """Get color mapping for models."""
    # Custom palette: #0000a2 (blue), #e9c716 (yellow), #bc272d (red), #50ad9f (teal)
    return {
        'ERGO': '#0000a2',
        'ERGO2': '#e9c716',
        'NetTCR': '#bc272d',
        'NetTCR-2.2': '#50ad9f',
        'TITAN': '#0000a2',
        'EPACT': '#e9c716',
        'PanPep': '#bc272d',
        'SCEPTR': '#50ad9f',
    }


def get_epitope_group_colors() -> dict:
    """Get color mapping for epitope groups."""
    config = get_config()
    return {
        'Viral': config.get_color('viral'),
        'self': config.get_color('self'),
        'T1D': config.get_color('T1D'),
        'seen': config.get_color('seen'),
        'unseen': config.get_color('unseen'),
        'special': config.get_color('special'),
        '5-R': config.get_color('5-R'),
    }


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with consistent styling.

    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (width, height) in inches

    Returns:
        Figure and axes array
    """
    setup_plot_style()

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Ensure axes is always an array
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def save_figure(fig: plt.Figure, filepath, close: bool = True):
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure
        filepath: Output path (str or Path)
        close: Whether to close figure after saving
    """
    config = get_config()

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=config.figure_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    if close:
        plt.close(fig)


def add_correlation_annotation(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    loc: str = 'upper right'
):
    """
    Add Spearman correlation annotation to plot.

    Args:
        ax: Matplotlib axes
        x: X values
        y: Y values
        loc: Location string
    """
    from scipy import stats

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return

    r, p = stats.spearmanr(x[mask], y[mask])

    # Format text
    if p < 0.001:
        text = f'ρ = {r:.3f}***'
    elif p < 0.01:
        text = f'ρ = {r:.3f}**'
    elif p < 0.05:
        text = f'ρ = {r:.3f}*'
    else:
        text = f'ρ = {r:.3f}'

    # Position
    if loc == 'upper right':
        x_pos, y_pos = 0.95, 0.95
        ha, va = 'right', 'top'
    elif loc == 'upper left':
        x_pos, y_pos = 0.05, 0.95
        ha, va = 'left', 'top'
    elif loc == 'lower right':
        x_pos, y_pos = 0.95, 0.05
        ha, va = 'right', 'bottom'
    else:
        x_pos, y_pos = 0.05, 0.05
        ha, va = 'left', 'bottom'

    ax.text(x_pos, y_pos, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def add_regression_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = 'red'):
    """
    Add regression line to scatter plot.

    Args:
        ax: Matplotlib axes
        x: X values
        y: Y values
        color: Line color
    """
    from scipy import stats

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return

    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])

    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=1.5, alpha=0.7)


def format_model_name(model: str) -> str:
    """Format model name for display."""
    from src.utils import get_model_display_name
    return get_model_display_name(model)
