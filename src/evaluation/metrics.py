"""
Evaluation Metrics for TCRP Benchmark V2.

Provides functions to calculate:
- Partial AUC (with FPR threshold)
- Mean per-peptide partial AUC
- Full AUC metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def partial_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.1
) -> float:
    """
    Calculate partial AUC up to a maximum false positive rate.

    Uses sklearn's roc_auc_score with max_fpr parameter for consistency
    with the legacy benchmark implementation.

    Args:
        y_true: True binary labels
        y_score: Prediction scores/probabilities
        max_fpr: Maximum FPR threshold (default 0.1)

    Returns:
        Partial AUC score (normalized by sklearn's McClish correction)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Handle edge cases
    if len(y_true) < 2:
        return 0.5

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # No discrimination possible

    try:
        # Use sklearn's built-in partial AUC calculation
        # This uses McClish correction for normalization
        return roc_auc_score(y_true, y_score, max_fpr=max_fpr)

    except Exception as e:
        logger.warning(f"Error calculating partial AUC: {e}")
        return 0.5


def mean_per_peptide_partial_auc(
    df: pd.DataFrame,
    peptide_col: str = 'Peptide',
    label_col: str = 'Label',
    prob_col: str = 'Prediction_Prob',
    max_fpr: float = 0.1,
    min_samples: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate macro-averaged partial AUC across peptides.

    Args:
        df: DataFrame with predictions
        peptide_col: Column name for peptide sequences
        label_col: Column name for true labels
        prob_col: Column name for prediction probabilities
        max_fpr: Maximum FPR for partial AUC
        min_samples: Minimum samples per peptide to calculate

    Returns:
        Tuple of (mean_pauc, per_peptide_dict)
    """
    peptide_aucs = {}

    for peptide in df[peptide_col].unique():
        peptide_df = df[df[peptide_col] == peptide]

        # Need sufficient samples with both classes
        n_pos = (peptide_df[label_col] == 1).sum()
        n_neg = (peptide_df[label_col] == 0).sum()

        if len(peptide_df) < min_samples or n_pos == 0 or n_neg == 0:
            continue

        pauc = partial_auc(
            peptide_df[label_col].values,
            peptide_df[prob_col].values,
            max_fpr=max_fpr
        )

        peptide_aucs[peptide] = pauc

    if len(peptide_aucs) == 0:
        return 0.5, {}

    mean_pauc = np.mean(list(peptide_aucs.values()))
    return mean_pauc, peptide_aucs


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.1
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for predictions.

    Args:
        y_true: True binary labels
        y_score: Prediction scores/probabilities
        max_fpr: Maximum FPR for partial AUC

    Returns:
        Dictionary of metric names to values
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    metrics = {}

    # Handle edge cases
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if len(y_true) < 2 or n_pos == 0 or n_neg == 0:
        return {
            'ROC_AUC': 0.5,
            'Partial_AUC': 0.5,
            'PR_AUC': n_pos / len(y_true) if len(y_true) > 0 else 0.5,
            'N_Samples': len(y_true),
            'N_Positive': n_pos,
            'N_Negative': n_neg,
        }

    try:
        # ROC AUC
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_score)
    except Exception:
        metrics['ROC_AUC'] = 0.5

    # Partial AUC
    metrics['Partial_AUC'] = partial_auc(y_true, y_score, max_fpr)

    try:
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['PR_AUC'] = auc(recall, precision)
    except Exception:
        metrics['PR_AUC'] = n_pos / len(y_true)

    # Sample counts
    metrics['N_Samples'] = len(y_true)
    metrics['N_Positive'] = n_pos
    metrics['N_Negative'] = n_neg

    return metrics


def calculate_all_metrics(
    df: pd.DataFrame,
    label_col: str = 'Label',
    prob_col: str = 'Prediction_Prob',
    peptide_col: str = 'Peptide',
    peptide_type_col: Optional[str] = 'Peptide_Type',
    max_fpr: float = 0.1,
    filter_antigen_status: Optional[str] = 'valid'
) -> Dict[str, float]:
    """
    Calculate all benchmark metrics including per-peptide and seen/unseen breakdowns.

    Args:
        df: DataFrame with predictions
        label_col: Column name for true labels
        prob_col: Column name for prediction probabilities
        peptide_col: Column name for peptide sequences
        peptide_type_col: Column name for seen/unseen classification
        max_fpr: Maximum FPR for partial AUC
        filter_antigen_status: Filter to this Antigen_Status value (default: 'valid').
                               Set to None to disable filtering.

    Returns:
        Dictionary with all metrics
    """
    # Filter by Antigen_Status if column exists and filtering is enabled
    if filter_antigen_status and 'Antigen_Status' in df.columns:
        original_len = len(df)
        df = df[df['Antigen_Status'] == filter_antigen_status]
        if len(df) < original_len:
            logger.debug(f"Filtered from {original_len} to {len(df)} samples (Antigen_Status={filter_antigen_status})")

    results = {}

    # Overall metrics
    overall_metrics = calculate_metrics(
        df[label_col].values,
        df[prob_col].values,
        max_fpr
    )
    for key, value in overall_metrics.items():
        results[f'Overall_{key}'] = value

    # Mean per-peptide partial AUC
    mean_pauc, peptide_aucs = mean_per_peptide_partial_auc(
        df, peptide_col, label_col, prob_col, max_fpr
    )
    results['Mean_Per_Peptide_pAUC'] = mean_pauc
    results['N_Peptides_Evaluated'] = len(peptide_aucs)

    # Seen/Unseen breakdown if available
    if peptide_type_col and peptide_type_col in df.columns:
        for ptype in ['seen', 'unseen']:
            type_df = df[df[peptide_type_col] == ptype]
            if len(type_df) > 0:
                type_metrics = calculate_metrics(
                    type_df[label_col].values,
                    type_df[prob_col].values,
                    max_fpr
                )
                for key, value in type_metrics.items():
                    results[f'{ptype.capitalize()}_{key}'] = value

                # Per-peptide for this type
                type_mean_pauc, _ = mean_per_peptide_partial_auc(
                    type_df, peptide_col, label_col, prob_col, max_fpr
                )
                results[f'{ptype.capitalize()}_Mean_Per_Peptide_pAUC'] = type_mean_pauc

    return results


def format_metrics_table(
    all_results: Dict[str, Dict[str, float]],
    metric_key: str = 'Mean_Per_Peptide_pAUC'
) -> pd.DataFrame:
    """
    Format metrics results as a comparison table.

    Args:
        all_results: Dictionary mapping model names to their metrics
        metric_key: Which metric to use for comparison

    Returns:
        DataFrame with models as rows and datasets as columns
    """
    table_data = []

    for model_name, model_metrics in all_results.items():
        row = {'Model': model_name}

        # Extract the specific metric for each dataset
        for key, value in model_metrics.items():
            if metric_key in key:
                dataset = key.replace(f'_{metric_key}', '').replace(metric_key, 'Overall')
                row[dataset] = value

        table_data.append(row)

    return pd.DataFrame(table_data)
