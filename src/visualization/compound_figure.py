"""
Compound Figure Generation for Publication.

Creates a multi-panel figure combining multiple visualizations:
- Row 1: (a) Pipeline overview SVG, (b) TetTCR+IMMREP23 pAUC boxplot, (c) Fingerprinting pAUC bars
- Row 2: (d) Log2FC scatter 2x4, (e) TCR correlation 2x4
- Row 3: (f,g) ERGO-II dual heatmap, (h) NetTCR-2.2 heatmap, (i) EPACT heatmap

Note: The "Refined Flow.svg" file (panel a) must be present in the visualizations directory.
This file is provided externally and included in the repository, while other panels are generated.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False

import io

from ..utils import get_paths, get_config

logger = logging.getLogger(__name__)

# =====================================================================
# Configuration
# =====================================================================

MODEL_ORDER = ['ERGO', 'ERGO2', 'NetTCR', 'NetTCR22', 'TITAN', 'EPACT', 'PanPep', 'SCEPTR']

TCR_GROUPS = [
    'SVAR.1', 'SVAR.2', 'SVAR.3', 'SVAR.5', 'SVAR.6', 'SVAR.7',
    'SVAR.13', 'SVAR.14', 'SVAR.16', 'SVAR.21', 'SVAR.26', 'SVAR.56',
    'SVAR.57', 'SVAR.59', 'SVAR.73', 'SVAR.74', 'SVAR.75', 'SVAR.76',
    'SVAR.82', 'SVAR.85', 'SVAR.90'
]

REFERENCE_PEPTIDE = "YLQPRTFLL"
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'V', 'Y']

# Fingerprint heatmaps (f,g,h): imshow aspect = display width per x-unit / per y-unit (inverse of 1.5:1 W:H → 0.7)
HEATMAP_CELL_ASPECT_WH = 0.7

# Each heatmap sub-axes scaled about its center (1.0 = fills grid cell). 0.95 ≈ 5% smaller; use 0.9995 for 0.05% shrink
HEATMAP_SUBPLOT_SCALE = 0.95

# Font sizes (enlarged for legibility in compound figure export)
FONT_SIZES = {
    'title': 17,
    'axis_label': 14,
    'tick_label': 13,
    'annotation': 13,
    'legend': 13,
    'model_title': 16,
}

# Must match fig.savefig(..., dpi=...) for pixel↔figure conversion
_COMPOUND_FIG_SAVE_DPI = 300


def _figure_y_delta_from_pixels(fig, px: float) -> float:
    """Vertical shift in figure coordinates for a pixel offset at save DPI."""
    h_in = fig.get_figheight()
    return px / (h_in * _COMPOUND_FIG_SAVE_DPI)


def _axes_trans_x_left_for_pixels(fig, ax, px: float) -> float:
    """Negative Δ in axes x (transAxes) to move something left by px in figure space at save DPI."""
    w_in = fig.get_figwidth()
    pos = ax.get_position()
    if pos.width <= 1e-9:
        return 0.0
    dx_fig = -px / (w_in * _COMPOUND_FIG_SAVE_DPI)
    return dx_fig / pos.width


def _axes_trans_y_extra_for_pixels(fig, ax, px: float) -> float:
    """Δ in axes y (transAxes) to move something up by px in figure space at save DPI."""
    pos = ax.get_position()
    if pos.height <= 1e-9:
        return 0.0
    return _figure_y_delta_from_pixels(fig, px) / pos.height


def _shift_axes_up(axes, dy: float) -> None:
    """Move axes upward by dy (figure coordinates)."""
    for ax in axes:
        if ax is None:
            continue
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def _scale_axes_about_center(ax, scale: float) -> None:
    """Shrink or grow axes bbox about its center (figure coordinates)."""
    if ax is None or not ax.get_visible():
        return
    pos = ax.get_position()
    cx = pos.x0 + pos.width * 0.5
    cy = pos.y0 + pos.height * 0.5
    w = pos.width * scale
    h = pos.height * scale
    ax.set_position([cx - w * 0.5, cy - h * 0.5, w, h])


def _scale_heatmap_subaxes(axes_fg, axes_h, scale: float) -> None:
    """Apply same linear scale to all f/g/h heatmap subdiagrams."""
    for ax in list(axes_fg) + list(axes_h):
        _scale_axes_about_center(ax, scale)


def _union_bbox_axes(axes) -> tuple:
    """Bounding box (x0, y0, width, height) in figure coordinates."""
    pos = [a.get_position() for a in axes if a.get_visible()]
    if not pos:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(p.x0 for p in pos)
    y0 = min(p.y0 for p in pos)
    x1 = max(p.x1 for p in pos)
    y1 = max(p.y1 for p in pos)
    return (x0, y0, x1 - x0, y1 - y0)


def _add_cbar_right_of_union(fig, bbox, mappable, label, tick_params_kw, width=0.007, pad=0.01):
    """Vertical colorbar along the right edge of a union bbox."""
    x0, y0, w, h = bbox
    cax = fig.add_axes([x0 + w + pad, y0, width, h])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=FONT_SIZES['axis_label'])
    cb.ax.tick_params(**tick_params_kw)
    return cb


def _add_cbar_left_of_union(fig, bbox, mappable, label, tick_params_kw, width=0.007, pad=0.012):
    """Vertical colorbar along the left edge of a union bbox (e.g. Log2FC for panel f)."""
    x0, y0, w, h = bbox
    cax = fig.add_axes([x0 - pad - width, y0, width, h])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=FONT_SIZES['axis_label'])
    cb.ax.tick_params(**tick_params_kw)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    return cb


def _add_cbar_vertical_in_svar_column_above_cell(
    fig,
    ax_svar,
    bbox_union,
    mappable,
    label: str,
    tick_params_kw: dict,
    *,
    width_frac: float = 0.22,
    gap_above_svar: float = 0.004,
    length_frac: float = 0.7,
):
    """
    Vertical colorbar inside the panel, in the column of SVAR.90, above the SVAR.90 sub-axes.
    The 5×5 layout leaves column 4 empty above (4,4); this uses that strip in figure coords.
    Bar height is length_frac of the strip, vertically centered; bar is centered in column.
    Returns (colorbar, y_top_of_panel_union) for follow-up layout (e.g. legend).
    """
    pos_sv = ax_svar.get_position()
    x0_u, y0_u, w_u, h_u = bbox_union
    y_top = y0_u + h_u
    y_strip_bottom = pos_sv.y1 + gap_above_svar
    h_strip = max(y_top - y_strip_bottom, 0.04)
    h_c = max(h_strip * length_frac, 0.022)
    y_c = y_strip_bottom + (h_strip - h_c) * 0.5
    bar_w = max(pos_sv.width * width_frac, 0.004)
    x_c = pos_sv.x0 + pos_sv.width * 0.5 - bar_w * 0.5
    cax = fig.add_axes([x_c, y_c, bar_w, h_c])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=FONT_SIZES['axis_label'])
    cb.ax.tick_params(**tick_params_kw)
    return cb, y_top


# =====================================================================
# Helper Functions
# =====================================================================

def _get_display_name(model_key: str) -> str:
    """Get display name for model from config."""
    config = get_config()
    return config.get_model_display_name(model_key)


def _load_svg_as_image(svg_path: Path) -> np.ndarray:
    """Load SVG file and convert to numpy array."""
    if not HAS_CAIROSVG:
        raise ImportError("cairosvg is required for compound figure generation. Install with: pip install cairosvg")
    png_data = cairosvg.svg2png(url=str(svg_path), scale=2.0)
    img = Image.open(io.BytesIO(png_data))
    return np.array(img)


def _embed_png_fill_cell(ax, image_path: Path, anchor: str = "W") -> bool:
    """
    Embed PNG with aspect='auto' so it fills the axes (use for panel b to match row height with c).
    """
    if not image_path.exists():
        return False
    img = Image.open(image_path)
    logger.info("Embed PNG (fill cell) %s: %s", image_path.name, img.size)
    arr = np.asarray(img)
    ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.axis("off")
    ax.set_anchor(anchor)
    return True


def _embed_png_preserving_aspect(ax, image_path: Path, anchor: str = "C") -> bool:
    """
    Display a PNG without stretching: native aspect via imshow + set_box_aspect.

    anchor: 'W' west-aligns the axes in its cell so extra width goes to the right
    (helps panels b/c use full row width visually).
    """
    if not image_path.exists():
        return False
    img = Image.open(image_path)
    w_px, h_px = img.size
    logger.info(
        "Embed PNG %s: %d x %d px (w:h=%.4f)",
        image_path.name,
        w_px,
        h_px,
        w_px / h_px if h_px else 0.0,
    )
    arr = np.asarray(img)
    ax.imshow(arr, aspect="equal", interpolation="nearest")
    ax.axis("off")
    if w_px > 0:
        ax.set_box_aspect(h_px / w_px)
    ax.set_anchor(anchor)
    return True


def _add_panel_label(ax, label: str, x=0.0, y=1.02, fontsize=21):
    """Add a panel label (a), (b), etc. at the top-left of an axes."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='left',
            fontfamily='sans-serif')


def _add_panel_label_figure(fig, label: str, x: float, y: float, fontsize=21):
    """Deprecated: prefer _add_panel_label on specific axes."""
    fig.text(x, y, f'({label})', fontsize=fontsize, fontweight='bold',
             va='top', ha='left', fontfamily='sans-serif')


# =====================================================================
# Data Loading
# =====================================================================

def _load_fingerprinting_data(mode: str) -> pd.DataFrame:
    """Load merged fingerprinting data."""
    paths = get_paths(mode=mode)
    merged_file = paths.MERGED_DIR / 'fingerprinting_all_models.csv'
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_file}")
    df = pd.read_csv(merged_file)
    if 'Antigen_Status' in df.columns:
        df = df[df['Antigen_Status'] == 'valid']
    return df


def _load_tettcr_immrep_data(mode: str):
    """Load TetTCR and IMMREP23 data for boxplot."""
    paths = get_paths(mode=mode)
    tettcr_file = paths.MERGED_DIR / 'tettcr_all_models.csv'
    immrep23_file = paths.MERGED_DIR / 'immrep23_all_models.csv'
    
    tettcr_df = pd.read_csv(tettcr_file) if tettcr_file.exists() else None
    immrep23_df = pd.read_csv(immrep23_file) if immrep23_file.exists() else None
    
    return tettcr_df, immrep23_df


# =====================================================================
# Panel (b): Combined Seen pAUC Boxplot
# =====================================================================

def _calculate_per_peptide_pauc(df, prob_col, max_fpr=0.1):
    """Calculate per-peptide partial AUC."""
    results = []
    for peptide in df['Peptide'].unique():
        pep_df = df[df['Peptide'] == peptide]
        y_true = pep_df['Label'].values
        y_score = pep_df[prob_col].values
        mask = ~np.isnan(y_score)
        
        if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
            try:
                pauc = roc_auc_score(y_true[mask], y_score[mask], max_fpr=max_fpr)
                results.append({'Peptide': peptide, 'pAUC': pauc})
            except:
                pass
    return pd.DataFrame(results)


def _create_combined_boxplot(fig, gs, tettcr_df, immrep23_df):
    """Create combined TetTCR + IMMREP23 boxplot."""
    config = get_config()
    
    COLORS = {
        'TetTCR-SeqHD (Viral)': config.get_group_color('tettcr', 'Viral'),
        'TetTCR-SeqHD (Self)': config.get_group_color('tettcr', 'Self'),
        'IMMREP23 (Seen)': config.get_group_color('immrep23', 'Seen'),
        'IMMREP23 (Unseen)': config.get_group_color('immrep23', 'Unseen')
    }
    
    tettcr_viral = tettcr_df[tettcr_df['Epitope_Group'] == 'Viral'].copy()
    tettcr_self = tettcr_df[tettcr_df['Epitope_Group'] == 'self'].copy()
    immrep23_seen = immrep23_df[immrep23_df['Epitope_Group'] == 'seen'].copy()
    immrep23_unseen = immrep23_df[immrep23_df['Epitope_Group'] == 'unseen'].copy()
    
    plot_data = []
    for model in MODEL_ORDER:
        prob_col = f'{model}_Prob'
        display_name = _get_display_name(model)
        
        for df_subset, dataset_name in [
            (tettcr_viral, 'TetTCR-SeqHD (Viral)'),
            (tettcr_self, 'TetTCR-SeqHD (Self)'),
            (immrep23_seen, 'IMMREP23 (Seen)'),
            (immrep23_unseen, 'IMMREP23 (Unseen)')
        ]:
            if prob_col in df_subset.columns:
                pauc_df = _calculate_per_peptide_pauc(df_subset, prob_col)
                for _, row in pauc_df.iterrows():
                    plot_data.append({
                        'Model': display_name,
                        'Dataset': dataset_name,
                        'pAUC': row['pAUC'],
                        'Peptide': row['Peptide']
                    })
    
    plot_df = pd.DataFrame(plot_data)
    ax = fig.add_subplot(gs)
    
    datasets = ['TetTCR-SeqHD (Viral)', 'TetTCR-SeqHD (Self)', 'IMMREP23 (Seen)', 'IMMREP23 (Unseen)']
    model_display_names = [_get_display_name(m) for m in MODEL_ORDER]
    n_models = len(model_display_names)
    n_datasets = len(datasets)
    
    box_width = 0.18
    
    for i, model_name in enumerate(model_display_names):
        model_data = plot_df[plot_df['Model'] == model_name]
        base_pos = i * (n_datasets * box_width + 0.3)
        
        for j, dataset in enumerate(datasets):
            data = model_data[model_data['Dataset'] == dataset]['pAUC'].values
            if len(data) > 0:
                pos = base_pos + j * box_width
                bp = ax.boxplot([data], positions=[pos], widths=box_width * 0.8,
                               patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(COLORS[dataset])
                bp['boxes'][0].set_alpha(0.7)
                for element in ['whiskers', 'caps', 'medians']:
                    plt.setp(bp[element], color='black', linewidth=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    
    tick_positions = [i * (n_datasets * box_width + 0.3) + (n_datasets - 1) * box_width / 2 
                      for i in range(n_models)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=FONT_SIZES['tick_label'])
    
    ax.set_ylabel('Partial AUC$_{0.1}$', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])
    
    legend_elements = [mpatches.Patch(facecolor=COLORS[d], label=d, alpha=0.7) for d in datasets]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZES['legend'], ncol=2)
    
    return ax


# =====================================================================
# Panel (c): Fingerprinting Partial AUC Bars
# =====================================================================

def _create_fingerprinting_bars(fig, gs, df):
    """Create fingerprinting partial AUC bar chart."""
    ax = fig.add_subplot(gs)
    
    y_true = df['Label'].values
    pauc_values = {}
    
    for model in MODEL_ORDER:
        prob_col = f'{model}_Prob'
        if prob_col in df.columns:
            y_score = df[prob_col].values
            mask = ~np.isnan(y_score)
            if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
                try:
                    pauc = roc_auc_score(y_true[mask], y_score[mask], max_fpr=0.1)
                    pauc_values[model] = pauc
                except:
                    pauc_values[model] = 0.5
    
    display_names = [_get_display_name(m) for m in MODEL_ORDER]
    values = [pauc_values.get(m, 0.5) for m in MODEL_ORDER]
    
    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(x, values, width=0.6, color='#d62728', edgecolor='black', linewidth=0.5, alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=FONT_SIZES['annotation'], fontweight='bold')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=FONT_SIZES['tick_label'])
    ax.set_ylabel('Partial AUC$_{0.1}$', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])
    
    legend_elements = [
        mpatches.Patch(facecolor='#d62728', label='FingerPrinting', alpha=0.8),
        Line2D([0], [0], color='gray', linestyle='--', label='Random (0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZES['legend'])
    
    return ax


# =====================================================================
# Panel (d): Log2FC Scatter 2x4
# =====================================================================

def _add_regression_line(ax, x, y, color='gray'):
    """Add regression line to scatter plot."""
    from scipy import stats
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=1, alpha=0.7)


def _create_log2fc_scatter_2x4(fig, gs, df, threshold=1.0):
    """Create 2x4 grid of log2fc scatter plots. Returns (top-left axes, all axes in grid)."""
    inner_gs = gs.subgridspec(2, 4, hspace=0.35, wspace=0.20)
    
    pos_color = '#d62728'
    neg_color = '#1f77b4'
    
    df = df.copy()
    df['Binding_Group'] = df['log2foldchange'].apply(lambda x: 'Positive' if x >= threshold else 'Negative')
    
    first_ax = None
    all_axes = []
    for i, model_key in enumerate(MODEL_ORDER):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(inner_gs[row, col])
        all_axes.append(ax)
        if first_ax is None:
            first_ax = ax
        
        prob_col = f'{model_key}_Prob'
        if prob_col not in df.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(_get_display_name(model_key), fontsize=FONT_SIZES['model_title'])
            continue
        
        for group, color in [('Negative', neg_color), ('Positive', pos_color)]:
            group_df = df[df['Binding_Group'] == group]
            if len(group_df) > 0:
                ax.scatter(group_df[prob_col], group_df['log2foldchange'],
                          c=color, alpha=0.5, s=10, edgecolor='none')
        
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1)
        
        x, y = df[prob_col].values, df['log2foldchange'].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 3:
            r_all, _ = spearmanr(x[mask], y[mask])
            _add_regression_line(ax, x[mask], y[mask], color='gray')
            ax.text(0.98, 0.02, f'ρ={r_all:.2f}', transform=ax.transAxes,
                    fontsize=FONT_SIZES['annotation'], va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.2))
        
        ax.set_xlim(0, 1)
        ax.set_title(_get_display_name(model_key), fontsize=FONT_SIZES['model_title'], fontweight='bold')
        ax.tick_params(labelsize=FONT_SIZES['tick_label'])
        
        if row == 1:
            ax.set_xlabel('Pred. Prob.', fontsize=FONT_SIZES['axis_label'])
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('log2FC', fontsize=FONT_SIZES['axis_label'])
        else:
            ax.set_yticklabels([])
    
    return first_ax, all_axes


# =====================================================================
# Panel (e): TCR Correlation 2x4
# =====================================================================

def _calculate_tcr_correlations(df, prob_col):
    """Calculate Spearman correlation for each TCR group."""
    correlations = {}
    for tcr_group in df['TCR_Group'].unique():
        tcr_df = df[df['TCR_Group'] == tcr_group]
        pred_vals = tcr_df[prob_col].values
        log2fc_vals = tcr_df['log2foldchange'].values
        mask = ~(np.isnan(pred_vals) | np.isnan(log2fc_vals))
        if mask.sum() > 3:
            try:
                r, _ = spearmanr(pred_vals[mask], log2fc_vals[mask])
                correlations[tcr_group] = r if not np.isnan(r) else 0.0
            except:
                correlations[tcr_group] = 0.0
        else:
            correlations[tcr_group] = 0.0
    return correlations


def _create_tcr_correlation_2x4(fig, gs, df):
    """Create 2x4 grid of per-TCR correlation heatmaps. Returns (top-left axes, mappable, all axes)."""
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('custom_rwb', colors, N=256)
    
    # wspace: tight columns; hspace: enough gap so row-2 titles (2 lines) do not overlap row-1 heatmaps
    inner_gs = gs.subgridspec(2, 4, hspace=0.31, wspace=0.035)
    
    first_ax = None
    im = None
    all_axes = []
    
    for i, model in enumerate(MODEL_ORDER):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(inner_gs[row, col])
        all_axes.append(ax)
        if first_ax is None:
            first_ax = ax
        
        prob_col = f'{model}_Prob'
        if prob_col not in df.columns:
            ax.set_visible(False)
            continue
        
        correlations = _calculate_tcr_correlations(df, prob_col)
        corr_matrix = np.full((5, 5), np.nan)
        tcr_list = TCR_GROUPS[:-1]
        
        for j, tcr in enumerate(tcr_list):
            r, c = j // 4, j % 4
            corr_matrix[r, c] = correlations.get(tcr, 0.0)
        corr_matrix[4, 4] = correlations.get('SVAR.90', 0.0)
        
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
        
        for j, tcr in enumerate(tcr_list):
            r, c = j // 4, j % 4
            val = correlations.get(tcr, 0.0)
            color = 'white' if abs(val) > 0.5 else 'black'
            tcr_short = tcr.replace('SVAR.', '')
            ax.text(c, r, f'{tcr_short}\n{val:.2f}', ha='center', va='center',
                   fontsize=FONT_SIZES['annotation'], color=color, fontweight='bold')
        
        val_90 = correlations.get('SVAR.90', 0.0)
        color_90 = 'white' if abs(val_90) > 0.5 else 'black'
        ax.text(4, 4, f'90\n{val_90:.2f}', ha='center', va='center',
               fontsize=FONT_SIZES['annotation'], color=color_90, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        all_corrs = [correlations.get(tcr, 0.0) for tcr in TCR_GROUPS]
        overall_corr = np.mean(all_corrs)
        # Tighter pad on row 1: keeps the 2-line title lower in the gutter so it does not collide
        # with row-0 heatmaps (extra row gap is from inner_gs hspace).
        title_pad = 3.5 if row == 1 else 6.0
        ax.set_title(
            f'{_get_display_name(model)}\n(ρ={overall_corr:.2f})',
            fontsize=FONT_SIZES['model_title'],
            fontweight='bold',
            pad=title_pad,
        )
    
    return first_ax, im, all_axes


# =====================================================================
# Panels (f-i): Heatmaps
# =====================================================================

def _get_position_and_aa(peptide):
    """Extract position and amino acid from peptide variant."""
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


def _get_reference_positions():
    """Get reference amino acid positions."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    positions = []
    for col_idx, aa in enumerate(REFERENCE_PEPTIDE):
        if aa in aa_to_idx:
            positions.append((aa_to_idx[aa], col_idx))
    return positions


def _prepare_heatmap_matrix(df, tcr_group, value_col):
    """Prepare heatmap matrix for a TCR group."""
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
        pos, aa = _get_position_and_aa(peptide)
        if pos > 0 and aa in aa_to_idx:
            row_idx = aa_to_idx[aa]
            col_idx = pos - 1
            matrix[row_idx, col_idx] = value
            status_matrix[row_idx, col_idx] = status
    
    return matrix, status_matrix


def _create_single_tcr_heatmap_ax(ax, matrix, status_matrix, tcr_group, cmap, vmin, vmax, 
                                  ref_positions, show_ylabel=False, show_xlabel=False, norm=None):
    """Create a single heatmap on given axes."""
    # aspect = display width per x-unit / display height per y-unit (0.7 ≈ reciprocal of 1.5 W:H cells)
    if norm is not None:
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect=HEATMAP_CELL_ASPECT_WH)
    else:
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect=HEATMAP_CELL_ASPECT_WH)
    
    # Add white grid lines between cells (inner gridlines)
    for col_idx in range(10):
        ax.axvline(x=col_idx - 0.5, color='white', linewidth=0.5)
    for row_idx in range(21):
        ax.axhline(y=row_idx - 0.5, color='white', linewidth=0.5)
    
    # Grey overlay for unstable/wild type
    for row_idx in range(20):
        for col_idx in range(9):
            status = status_matrix[row_idx, col_idx]
            if status in ['unstable', 'wild type']:
                rect = mpatches.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                          linewidth=0, facecolor='#808080', alpha=1.0)
                ax.add_patch(rect)
    
    ax.set_xticks(range(9))
    ax.set_yticks(range(20))
    
    if show_xlabel:
        ax.set_xticklabels(range(1, 10), fontsize=FONT_SIZES['tick_label'])
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_yticklabels(AMINO_ACIDS, fontsize=FONT_SIZES['tick_label'])
    else:
        ax.set_yticklabels([])
    
    ax.set_title(tcr_group, fontsize=FONT_SIZES['annotation'], fontweight='bold', pad=2)
    
    # Reference amino acid borders (black)
    for row_idx, col_idx in ref_positions:
        rect = mpatches.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                  linewidth=0.8, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    
    # Dark/black border around the entire heatmap
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    
    return im


def _create_dual_heatmap_panel(fig, gs_left, gs_right, mode, model_key):
    """
    Create dual heatmap panels (log2fc and probability) for a model.
    Returns:
        im_log2fc, im_prob, ax_topleft_log2fc, ax_topleft_prob, all_axes, ax_svar90_log2fc, ax_svar90_prob
    """
    paths = get_paths(mode=mode)
    full_df = pd.read_csv(paths.MERGED_DIR / 'fingerprinting_all_models.csv')
    
    prob_col = f'{model_key}_Prob'
    ref_positions = _get_reference_positions()
    
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap_log2fc = LinearSegmentedColormap.from_list('rwb', colors, N=256)
    cmap_prob = LinearSegmentedColormap.from_list('rwb_prob', colors, N=256)
    
    log2fc_norm = TwoSlopeNorm(vmin=-1, vcenter=1, vmax=5)
    
    tcr_list = TCR_GROUPS[:-1]
    im_log2fc, im_prob = None, None
    ax_topleft_log2fc, ax_topleft_prob = None, None
    all_axes = []
    
    for i, tcr_group in enumerate(tcr_list):
        row, col = i // 4, i % 4
        show_ylabel = (col == 0)
        show_xlabel = (row == 4)
        
        ax_log2fc = fig.add_subplot(gs_left[row, col])
        all_axes.append(ax_log2fc)
        if ax_topleft_log2fc is None:
            ax_topleft_log2fc = ax_log2fc
        log2fc_matrix, log2fc_status = _prepare_heatmap_matrix(full_df, tcr_group, 'log2foldchange')
        log2fc_matrix_clipped = np.clip(log2fc_matrix, -1, 5)
        im_log2fc = _create_single_tcr_heatmap_ax(ax_log2fc, log2fc_matrix_clipped, log2fc_status,
                                                  tcr_group, cmap_log2fc, -1, 5, ref_positions,
                                                  show_ylabel=show_ylabel, show_xlabel=show_xlabel,
                                                  norm=log2fc_norm)
        
        ax_prob = fig.add_subplot(gs_right[row, col])
        all_axes.append(ax_prob)
        if ax_topleft_prob is None:
            ax_topleft_prob = ax_prob
        prob_matrix, prob_status = _prepare_heatmap_matrix(full_df, tcr_group, prob_col)
        im_prob = _create_single_tcr_heatmap_ax(ax_prob, prob_matrix, prob_status,
                                               tcr_group, cmap_prob, 0, 1, ref_positions,
                                               show_ylabel=show_ylabel, show_xlabel=show_xlabel)
    
    # SVAR.90
    ax_svar90_log2fc = fig.add_subplot(gs_left[4, 4])
    all_axes.append(ax_svar90_log2fc)
    log2fc_matrix_90, log2fc_status_90 = _prepare_heatmap_matrix(full_df, 'SVAR.90', 'log2foldchange')
    im_log2fc = _create_single_tcr_heatmap_ax(ax_svar90_log2fc, np.clip(log2fc_matrix_90, -1, 5),
                                              log2fc_status_90, 'SVAR.90', cmap_log2fc, -1, 5,
                                              ref_positions, show_xlabel=True, norm=log2fc_norm)
    
    ax_svar90_prob = fig.add_subplot(gs_right[4, 4])
    all_axes.append(ax_svar90_prob)
    prob_matrix_90, prob_status_90 = _prepare_heatmap_matrix(full_df, 'SVAR.90', prob_col)
    im_prob = _create_single_tcr_heatmap_ax(ax_svar90_prob, prob_matrix_90, prob_status_90,
                                           'SVAR.90', cmap_prob, 0, 1, ref_positions, show_xlabel=True)
    
    return im_log2fc, im_prob, ax_topleft_log2fc, ax_topleft_prob, all_axes, ax_svar90_log2fc, ax_svar90_prob


def _create_single_model_heatmap(fig, gs, mode, model_key):
    """Create single model heatmap with 5x5 TCR grid. Returns (image, top-left axes, all axes)."""
    paths = get_paths(mode=mode)
    full_df = pd.read_csv(paths.MERGED_DIR / 'fingerprinting_all_models.csv')
    
    prob_col = f'{model_key}_Prob'
    ref_positions = _get_reference_positions()
    cmap = plt.cm.RdBu_r
    
    inner_gs = gs.subgridspec(5, 5, hspace=0.135, wspace=0.055)
    tcr_list = TCR_GROUPS[:-1]
    
    im = None
    first_ax = None
    all_axes = []
    for i, tcr_group in enumerate(tcr_list):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(inner_gs[row, col])
        all_axes.append(ax)
        if first_ax is None:
            first_ax = ax
        
        matrix, status_matrix = _prepare_heatmap_matrix(full_df, tcr_group, prob_col)
        im = _create_single_tcr_heatmap_ax(ax, matrix, status_matrix, tcr_group, cmap, 0, 1,
                                          ref_positions, show_ylabel=(col == 0), show_xlabel=(row == 4))
    
    # SVAR.90
    ax_90 = fig.add_subplot(inner_gs[4, 4])
    all_axes.append(ax_90)
    matrix_90, status_90 = _prepare_heatmap_matrix(full_df, 'SVAR.90', prob_col)
    im = _create_single_tcr_heatmap_ax(ax_90, matrix_90, status_90, 'SVAR.90', cmap, 0, 1,
                                      ref_positions, show_xlabel=True)
    
    return im, first_ax, all_axes


# =====================================================================
# Main Compound Figure Creation
# =====================================================================

def generate_compound_figure(mode: str = "exact_match"):
    """
    Generate the compound figure for publication.
    
    Args:
        mode: Deduplication mode ("exact_match" or "fuzzy_match")
    
    Returns:
        Path to the output directory containing PNG and SVG files
    """
    logger.info(f"Creating compound figure for mode: {mode}")
    
    paths = get_paths(mode=mode)
    vis_dir = paths.VISUALIZATIONS_DIR
    output_dir = vis_dir / "compound_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    df_fp = _load_fingerprinting_data(mode)
    tettcr_df, immrep23_df = _load_tettcr_immrep_data(mode)
    logger.info(f"Loaded fingerprinting: {len(df_fp)} samples")
    
    # Large canvas: extra area for enlarged heatmap row (f,g,h)
    fig = plt.figure(figsize=(28, 41))
    
    # Nested layout: more vertical weight to heatmap row (f,g,h)
    outer_ratios = [3.85, 3.35]  # top (a)+(b,c) vs bottom (d,e)+(f,g,h)
    outer_gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=outer_ratios,
        # Smaller gap (b,c)→(d,e); was too large vs (d,e)→(f,g,h)
        hspace=0.076,
        top=0.985,
        bottom=0.02,
        left=0.005,
        right=0.995,
    )
    # Top: (a) then (b,c) — ~20% tighter gap (a)↔(b,c) vs hspace=0.10
    top_gs = outer_gs[0].subgridspec(2, 1, height_ratios=[3.15, 1.12], hspace=0.08)
    # (d,e)↔(f,g,h): need more gutter than before; pairs with dy_row2/dy_row3 balance below
    bottom_gs = outer_gs[1].subgridspec(2, 1, height_ratios=[1.0, 2.35], hspace=0.148)
    
    # =====================================================================
    # Row 0: (a) pipeline overview PNG (separate top row)
    # =====================================================================
    logger.info("Creating Row 0 (panel a)...")
    row0_gs = top_gs[0].subgridspec(1, 1)
    
    ax_a = fig.add_subplot(row0_gs[0, 0])
    new_process_path = vis_dir / "new_process.png"
    if not _embed_png_preserving_aspect(ax_a, new_process_path):
        ax_a.text(0.5, 0.5, 'Pipeline Overview\n(new_process.png missing)',
                  ha='center', va='center', fontsize=FONT_SIZES['annotation'], transform=ax_a.transAxes)
        ax_a.set_facecolor('#f0f0f0')
        ax_a.axis('off')
    _add_panel_label(ax_a, 'a')
    
    # =====================================================================
    # Row 1: (b,c) combined pAUC boxplot + fingerprinting pAUC bars
    # =====================================================================
    logger.info("Creating Row 1 (panels b,c)...")
    row1_gs = top_gs[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.055)
    
    # (b) Combined pAUC boxplot (load existing PNG to match standalone figure)
    ax_b = fig.add_subplot(row1_gs[0, 0])
    boxplot_path = vis_dir / "combined_seen_pauc_boxplot.png"
    if boxplot_path.exists():
        if not _embed_png_fill_cell(ax_b, boxplot_path, anchor="W"):
            ax_b.text(0.5, 0.5, 'Boxplot image failed', ha='center', va='center', transform=ax_b.transAxes)
            ax_b.axis('off')
    else:
        # Fallback: draw boxplot directly if PNG is missing
        logger.warning(f"combined_seen_pauc_boxplot.png not found at {boxplot_path}, falling back to plot recreation.")
        ax_b.remove()
        ax_b = _create_combined_boxplot(fig, row1_gs[0, 0], tettcr_df, immrep23_df)
    _add_panel_label(ax_b, 'b')

    # (c) Fingerprinting pAUC bars — west anchor to match (b), use full column width
    ax_c = _create_fingerprinting_bars(fig, row1_gs[0, 1], df_fp)
    ax_c.set_anchor("W")
    _add_panel_label(ax_c, 'c')
    
    # =====================================================================
    # Row 2: (d) log2fc scatter 2x4, (e) tcr correlation 2x4
    # =====================================================================
    logger.info("Creating Row 2 (panels d,e)...")
    row2_gs = bottom_gs[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.10)
    
    # (d) Log2FC scatter 2x4
    ax_d, axes_d = _create_log2fc_scatter_2x4(fig, row2_gs[0], df_fp)
    
    # (e) TCR correlation 2x4
    ax_e, im_corr, axes_e = _create_tcr_correlation_2x4(fig, row2_gs[1], df_fp)
    
    # =====================================================================
    # Row 3: (f,g) ERGO2 dual, (h) NetTCR-2.2
    # =====================================================================
    logger.info("Creating Row 3 (panels f,g,h)...")
    row3_gs = bottom_gs[1].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.045)
    
    # (f,g) ERGO2 dual heatmap — tighter subgrid spacing → larger square cells
    gs_f = row3_gs[0].subgridspec(5, 5, hspace=0.075, wspace=0.032)
    gs_g = row3_gs[1].subgridspec(5, 5, hspace=0.075, wspace=0.032)
    im_log2fc, im_prob, ax_f, ax_g, axes_fg, ax_svar90_f, ax_svar90_prob = _create_dual_heatmap_panel(
        fig, gs_f, gs_g, mode, 'ERGO2'
    )
    # (h) NetTCR-2.2 heatmap (last axes = SVAR.90 cell; im_h for inset colorbar)
    im_h, ax_h, axes_h = _create_single_model_heatmap(fig, row3_gs[2], mode, 'NetTCR22')
    
    # Nudge rows 2–4 ((b,c),(d,e),(f,g,h)) upward in figure space (same dy for all; keeps prior row-2/3 balance)
    fig.canvas.draw()
    dy_row2 = _figure_y_delta_from_pixels(fig, 30.0)
    # Heatmap row: keep closer to (d,e) than a huge dy_row3 (was 100px), which visually crushed that gap
    dy_row3 = _figure_y_delta_from_pixels(fig, 52.0)
    dy_d_only = _figure_y_delta_from_pixels(fig, 20.0)  # panel (d) only, above shared row-2 nudge
    dy_rows_bcd = _figure_y_delta_from_pixels(fig, 14.0)  # uniform lift: below (a), same for all three rows
    _shift_axes_up([ax_b, ax_c], dy_rows_bcd)
    _shift_axes_up(axes_d, dy_row2 + dy_d_only + dy_rows_bcd)
    _shift_axes_up(axes_e, dy_row2 + dy_rows_bcd)
    _shift_axes_up(axes_fg, dy_row3 + dy_rows_bcd)
    _shift_axes_up(axes_h, dy_row3 + dy_rows_bcd)
    
    _scale_heatmap_subaxes(axes_fg, axes_h, HEATMAP_SUBPLOT_SCALE)
    
    # Final layout before union bboxes (full-panel colorbars, not first sub-axes only)
    fig.canvas.draw()
    
    # Panel labels (d–h): after layout shifts; extra px = clearance above panel content
    _add_panel_label(
        ax_d, 'd', y=1.02 + _axes_trans_y_extra_for_pixels(fig, ax_d, 58.0),
    )
    _add_panel_label(
        ax_e, 'e', y=1.02 + _axes_trans_y_extra_for_pixels(fig, ax_e, 58.0),
    )
    _add_panel_label(
        ax_f,
        'f',
        x=_axes_trans_x_left_for_pixels(fig, ax_f, 10.0),
        y=1.04 + _axes_trans_y_extra_for_pixels(fig, ax_f, 40.0),
    )
    _add_panel_label(
        ax_g, 'g', y=1.04 + _axes_trans_y_extra_for_pixels(fig, ax_g, 40.0),
    )
    _add_panel_label(
        ax_h,
        'h',
        x=_axes_trans_x_left_for_pixels(fig, ax_h, 10.0),
        y=1.04 + _axes_trans_y_extra_for_pixels(fig, ax_h, 40.0),
    )
    
    colors_log2fc = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                     '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    cmap_log2fc = LinearSegmentedColormap.from_list('rwb', colors_log2fc, N=256)
    log2fc_norm = TwoSlopeNorm(vmin=-1, vcenter=1, vmax=5)
    sm_log2fc = cm.ScalarMappable(cmap=cmap_log2fc, norm=log2fc_norm)
    sm_log2fc.set_array([])

    bbox_e = _union_bbox_axes(axes_e)
    _add_cbar_right_of_union(
        fig, bbox_e, im_corr, 'Spearman ρ (e)',
        {'labelsize': FONT_SIZES['tick_label']},
    )

    axes_f_only = axes_fg[0::2]
    axes_g_only = axes_fg[1::2]
    bbox_f = _union_bbox_axes(axes_f_only)
    bbox_g = _union_bbox_axes(axes_g_only)
    bbox_h = _union_bbox_axes(axes_h)

    _cbar_kw = {'labelsize': FONT_SIZES['tick_label']}
    _cbar_len = 0.7

    # Log2FC (f): SVAR.90 column (left ERGO2 grid), above SVAR.90; bar 70% strip height, centered
    cbar_f, y_top_f = _add_cbar_vertical_in_svar_column_above_cell(
        fig, ax_svar90_f, bbox_f, sm_log2fc, 'Log2FC (f)', _cbar_kw, length_frac=_cbar_len,
    )
    cbar_f.set_ticks([-1, 0, 1, 2, 3, 4, 5])

    # Pred. prob. (g): SVAR.90 column (right ERGO2 grid), same inset layout
    cbar_g, _ = _add_cbar_vertical_in_svar_column_above_cell(
        fig, ax_svar90_prob, bbox_g, im_prob, 'Pred. Prob. (g)', _cbar_kw, length_frac=_cbar_len,
    )
    cbar_g.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Pred. prob. (h): panel h SVAR.90 column
    cbar_h, _ = _add_cbar_vertical_in_svar_column_above_cell(
        fig, axes_h[-1], bbox_h, im_h, 'Pred. Prob. (h)', _cbar_kw, length_frac=_cbar_len,
    )
    cbar_h.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Unstable/WT legend: above panel (f) colorbar strip (column above SVAR.90)
    pos_sv = ax_svar90_f.get_position()
    legend_elements = [mpatches.Patch(facecolor='#808080', edgecolor='black', label='Unstable/WT')]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        fontsize=FONT_SIZES['legend'],
        bbox_to_anchor=(pos_sv.x0 + pos_sv.width * 0.5, y_top_f + 0.012),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=True,
    )
    
    # =====================================================================
    # Save outputs
    # =====================================================================
    logger.info("Saving outputs...")
    
    png_path = output_dir / "compound_figure.png"
    fig.savefig(
        png_path,
        dpi=_COMPOUND_FIG_SAVE_DPI,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    logger.info(f"Saved: {png_path}")
    
    svg_path_out = output_dir / "compound_figure.svg"
    fig.savefig(
        svg_path_out,
        format='svg',
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    logger.info(f"Saved: {svg_path_out}")
    
    plt.close(fig)
    logger.info("Compound figure generation complete!")
    
    return output_dir


def main():
    """Main entry point for standalone execution."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils import setup_logging
    setup_logging("compound_figure", level="INFO")
    
    generate_compound_figure(mode="exact_match")


if __name__ == "__main__":
    main()

