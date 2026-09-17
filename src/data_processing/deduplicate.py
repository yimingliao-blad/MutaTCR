"""
CDR3 Deduplication Module for TCRP Benchmark V2.

This module removes test samples that overlap with model training data
using two deduplication modes:
1. Exact match: Remove samples with identical (CDR3b, Peptide) - CDR3a is less important
2. Fuzzy match: Remove samples with CDR3b Levenshtein distance ≤ 3

Each model has its own training data, so deduplication is model-specific.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import (
    get_paths, get_config, setup_logging,
    levenshtein_distance, get_all_model_keys
)

logger = logging.getLogger(__name__)


class TrainingDataDeduplicator:
    """
    Deduplicator that removes test samples overlapping with training data.

    Supports two modes:
    - exact_match: CDR3b + Peptide must match exactly (CDR3a is less important)
    - fuzzy_match: CDR3b within 3 edit distance of any training CDR3b
    """

    def __init__(self, mode: str = "exact_match"):
        """
        Initialize deduplicator.

        Args:
            mode: "exact_match" or "fuzzy_match"
        """
        if mode not in ["exact_match", "fuzzy_match"]:
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.paths = get_paths(mode=mode)
        self.config = get_config()

        # Training data storage
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.training_cdr3s: Dict[str, Set[str]] = {}  # CDR3b sequences only for fuzzy match
        self.training_tuples: Dict[str, Set[Tuple[str, str]]] = {}  # (CDR3b, Peptide) tuples for exact match

        logger.info(f"Initialized deduplicator in {mode} mode")

    def load_training_data(self, models_dir: Path) -> None:
        """
        Load training data for all models.

        Args:
            models_dir: Path to models directory
        """
        logger.info("Loading training data for all models...")

        # Define training data locations for each model
        # Paths handle both naming conventions (NetTCR vs NETTCR2)
        # Try both directory names for compatibility
        def find_model_dir(name_options):
            """Find first existing model directory from name options."""
            for name in name_options:
                if (models_dir / name).exists():
                    return models_dir / name
            return models_dir / name_options[0]  # Default to first option

        nettcr_dir = find_model_dir(['NetTCR', 'NETTCR2'])
        nettcr22_dir = find_model_dir(['NetTCR22', 'NETTCR22'])

        training_files = {
            'ERGO': models_dir / 'ERGO' / 'data' / 'VDJDB_complete.tsv',
            'ERGO2': models_dir / 'ERGO2' / 'Samples' / 'mcpas_train_samples.pickle',
            'NetTCR': nettcr_dir / 'data' / 'train_ab_95_alphabeta.csv',
            'NetTCR22': nettcr22_dir / 'data' / 'nettcr_2_2_full_dataset.csv',
            'TITAN': models_dir / 'TITAN' / 'datasets' / 'full_data+covid.csv',
            'EPACT': models_dir / 'EPACT' / 'sample' / 'VDJdb-GLCTLVAML.csv',  # Sample data
            'PanPep': models_dir / 'PanPep' / 'Data' / 'majority_training_dataset.csv',  # Use training data, not ground truth
            # SCEPTR uses runtime-generated training data (top VDJDB peptides)
        }

        for model, train_file in training_files.items():
            if train_file.exists():
                try:
                    df = self._load_training_file(model, train_file)
                    self._extract_training_features(model, df)
                    logger.info(f"  {model}: {len(df)} training samples")
                except Exception as e:
                    logger.warning(f"  {model}: Failed to load ({e})")
            else:
                logger.warning(f"  {model}: Training file not found ({train_file})")

        # Special handling for SCEPTR (top 6 VDJDB peptides)
        self._load_sceptr_training_data()

    def _load_training_file(self, model: str, filepath: Path) -> pd.DataFrame:
        """Load training data file based on format."""
        if filepath.suffix == '.pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            # Handle ERGO2 pickle format
            if isinstance(data, dict):
                return pd.DataFrame(data)
            return pd.DataFrame(data)
        elif filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.tsv':
            return pd.read_csv(filepath, sep='\t')
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")

    def _extract_training_features(self, model: str, df: pd.DataFrame) -> None:
        """Extract CDR3b sequences and (CDR3b, Peptide) tuples from training data."""
        self.training_data[model] = df

        # Extract CDR3b sequences (case-insensitive) for fuzzy matching
        cdr3b_set = set()
        tuple_set = set()  # (CDR3b, Peptide) tuples for exact matching

        # Try different column name conventions
        cdr3b_cols = ['CDR3b', 'cdr3b', 'CDR3.beta', 'CDR3.beta.aa', 'TRB_cdr3', 'CDR3', 'beta', 'tcrb', 'B3', 'binding_TCR']
        peptide_cols = ['Peptide', 'peptide', 'Epitope', 'epitope', 'Epitope.peptide', 'antigen.epitope']

        cdr3b_col = self._find_column(df, cdr3b_cols)
        peptide_col = self._find_column(df, peptide_cols)

        for _, row in df.iterrows():
            cdr3b = str(row.get(cdr3b_col, '')).upper() if cdr3b_col else ''
            peptide = str(row.get(peptide_col, '')).upper() if peptide_col else ''

            # Store CDR3b for fuzzy matching
            if cdr3b and cdr3b != 'NAN':
                cdr3b_set.add(cdr3b)

            # Store (CDR3b, Peptide) tuple for exact matching
            if cdr3b and cdr3b != 'NAN' and peptide and peptide != 'NAN':
                tuple_set.add((cdr3b, peptide))

        self.training_cdr3s[model] = cdr3b_set
        self.training_tuples[model] = tuple_set

        logger.debug(f"  {model}: {len(cdr3b_set)} unique CDR3b, {len(tuple_set)} unique (CDR3b, Peptide) tuples")

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _load_sceptr_training_data(self) -> None:
        """Load SCEPTR training data (top 6 VDJDB peptides)."""
        # SCEPTR uses embeddings trained on top 6 VDJDB peptides
        # We'll mark this as empty for now - actual implementation depends on SCEPTR setup
        sceptr_peptides = [
            'GILGFVFTL', 'GLCTLVAML', 'NLVPMVATV',
            'RAKFKQLL', 'YLQPRTFLL', 'LLWNGPMAV'
        ]
        self.training_cdr3s['SCEPTR'] = set()
        self.training_tuples['SCEPTR'] = set()
        logger.info(f"  SCEPTR: Using top 6 VDJDB peptides for context")

    def deduplicate(
        self,
        df: pd.DataFrame,
        model: str,
        cdr3a_col: str = 'CDR3a',
        cdr3b_col: str = 'CDR3b',
        peptide_col: str = 'Peptide'
    ) -> pd.DataFrame:
        """
        Remove samples that overlap with model's training data.

        Args:
            df: DataFrame to deduplicate
            model: Model name
            cdr3a_col: CDR3 alpha column name
            cdr3b_col: CDR3 beta column name
            peptide_col: Peptide column name

        Returns:
            Deduplicated DataFrame
        """
        if model not in self.training_cdr3s:
            logger.warning(f"No training data for {model}, skipping deduplication")
            return df

        initial_count = len(df)

        if self.mode == "exact_match":
            df_dedup = self._exact_dedup(df, model, cdr3a_col, cdr3b_col, peptide_col)
        else:
            df_dedup = self._fuzzy_dedup(df, model, cdr3a_col, cdr3b_col)

        removed = initial_count - len(df_dedup)
        logger.info(f"  {model}: Removed {removed}/{initial_count} samples ({removed/initial_count*100:.1f}%)")

        return df_dedup

    def _exact_dedup(
        self,
        df: pd.DataFrame,
        model: str,
        cdr3a_col: str,
        cdr3b_col: str,
        peptide_col: str
    ) -> pd.DataFrame:
        """Remove exact matches of (CDR3b, Peptide) - CDR3a is not used for deduplication."""
        training_tuples = self.training_tuples.get(model, set())

        def is_duplicate(row):
            cdr3b = str(row[cdr3b_col]).upper() if pd.notna(row[cdr3b_col]) else ''
            peptide = str(row[peptide_col]).upper() if pd.notna(row[peptide_col]) else ''
            return (cdr3b, peptide) in training_tuples

        mask = ~df.apply(is_duplicate, axis=1)
        return df[mask].copy()

    def _fuzzy_dedup(
        self,
        df: pd.DataFrame,
        model: str,
        cdr3a_col: str,
        cdr3b_col: str
    ) -> pd.DataFrame:
        """Remove samples with CDR3b within 3 edits of training CDR3b sequences."""
        training_cdr3s = self.training_cdr3s.get(model, set())
        max_distance = self.config.fuzzy_max_distance

        # Convert to list for faster iteration
        training_list = list(training_cdr3s)

        def is_fuzzy_match(seq: str) -> bool:
            """Check if CDR3b sequence is within max_distance of any training CDR3b."""
            if not seq or seq == 'NAN':
                return False
            seq = seq.upper()
            for train_seq in training_list:
                if abs(len(seq) - len(train_seq)) <= max_distance:
                    if levenshtein_distance(seq, train_seq) <= max_distance:
                        return True
            return False

        def is_duplicate(row):
            # Only check CDR3b for fuzzy matching (CDR3a is less important)
            cdr3b = str(row[cdr3b_col]).upper() if pd.notna(row[cdr3b_col]) else ''
            return is_fuzzy_match(cdr3b)

        mask = ~df.apply(is_duplicate, axis=1)
        return df[mask].copy()

    def deduplicate_all_models(
        self,
        df: pd.DataFrame,
        cdr3a_col: str = 'CDR3a',
        cdr3b_col: str = 'CDR3b',
        peptide_col: str = 'Peptide'
    ) -> Dict[str, pd.DataFrame]:
        """
        Deduplicate against all models' training data.

        Args:
            df: DataFrame to deduplicate
            cdr3a_col: CDR3 alpha column
            cdr3b_col: CDR3 beta column
            peptide_col: Peptide column

        Returns:
            Dictionary mapping model names to deduplicated DataFrames
        """
        results = {}
        for model in get_all_model_keys():
            results[model] = self.deduplicate(df, model, cdr3a_col, cdr3b_col, peptide_col)
        return results


def sample_negatives_by_epitope(
    df: pd.DataFrame,
    max_neg_ratio: int = 10,
    epitope_col: str = 'Epitope_Group',
    label_col: str = 'Label',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample negatives to maintain pos:neg ratio per epitope group.

    For TetTCR dataset, reduces negatives to max pos:neg = 1:10 per epitope.
    If there aren't enough negatives, keep all available.

    Args:
        df: DataFrame with Label and Epitope_Group columns
        max_neg_ratio: Maximum negative samples per positive (default: 10)
        epitope_col: Column name for epitope grouping
        label_col: Column name for labels (1=positive, 0=negative)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with sampled negatives
    """
    np.random.seed(random_state)

    result_dfs = []

    for epitope in df[epitope_col].unique():
        epitope_df = df[df[epitope_col] == epitope]

        positives = epitope_df[epitope_df[label_col] == 1]
        negatives = epitope_df[epitope_df[label_col] == 0]

        n_pos = len(positives)
        n_neg = len(negatives)
        max_neg = n_pos * max_neg_ratio

        # Always keep all positives
        result_dfs.append(positives)

        # Sample negatives if too many
        if n_neg > max_neg and max_neg > 0:
            sampled_neg = negatives.sample(n=max_neg, random_state=random_state)
            result_dfs.append(sampled_neg)
            logger.info(f"    {epitope}: {n_pos} pos, {n_neg} neg -> sampled to {max_neg} neg (1:{max_neg_ratio})")
        else:
            result_dfs.append(negatives)
            if n_pos > 0:
                actual_ratio = n_neg / n_pos if n_pos > 0 else 0
                logger.info(f"    {epitope}: {n_pos} pos, {n_neg} neg (ratio 1:{actual_ratio:.1f})")
            else:
                logger.info(f"    {epitope}: {n_pos} pos, {n_neg} neg (no positives)")

    return pd.concat(result_dfs, ignore_index=True)


def deduplicate_unified_data(
    mode: str = "exact_match",
    models_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Deduplicate all unified datasets against all models.

    Args:
        mode: "exact_match" or "fuzzy_match"
        models_dir: Path to models directory

    Returns:
        Nested dict: {dataset: {model: output_path}}
    """
    paths = get_paths(mode=mode)
    config = get_config()

    if models_dir is None:
        models_dir = paths.MODELS_DIR

    # Initialize deduplicator
    dedup = TrainingDataDeduplicator(mode=mode)
    dedup.load_training_data(models_dir)

    results = {}
    datasets = ['tettcr', 'immrep23', 'fingerprinting']

    for dataset in datasets:
        logger.info(f"\nDeduplicating {dataset}...")

        # Load unified data
        unified_file = paths.get_unified_file(dataset)
        if not unified_file.exists():
            logger.warning(f"  Unified file not found: {unified_file}")
            continue

        df = pd.read_csv(unified_file)
        logger.info(f"  Loaded {len(df)} records")

        # Deduplicate against each model
        results[dataset] = {}
        model_results = dedup.deduplicate_all_models(df)

        for model, df_dedup in model_results.items():
            # For TetTCR: sample negatives to maintain pos:neg = 1:10 per epitope
            if dataset == 'tettcr' and 'Epitope_Group' in df_dedup.columns and 'Label' in df_dedup.columns:
                logger.info(f"  Sampling negatives for {model} (pos:neg = 1:10 max)...")
                df_dedup = sample_negatives_by_epitope(
                    df_dedup,
                    max_neg_ratio=10,
                    epitope_col='Epitope_Group',
                    label_col='Label'
                )
                logger.info(f"    Final: {len(df_dedup)} samples")

            # Save deduplicated data
            output_dir = paths.DEDUP_DATA_DIR / mode / model
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{dataset}_deduplicated.csv"
            df_dedup.to_csv(output_file, index=False)
            results[dataset][model] = output_file

    return results


def main():
    """Main entry point for deduplication."""
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate unified data against model training data")
    parser.add_argument("--mode", choices=["exact_match", "fuzzy_match"], default="exact_match",
                        help="Deduplication mode")
    parser.add_argument("--models-dir", type=Path, help="Path to models directory")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("deduplicate", level="INFO")

    # Run deduplication
    results = deduplicate_unified_data(mode=args.mode, models_dir=args.models_dir)

    logger.info("\nDeduplication complete!")
    for dataset, model_files in results.items():
        logger.info(f"\n{dataset}:")
        for model, filepath in model_files.items():
            logger.info(f"  {model}: {filepath}")


if __name__ == "__main__":
    main()
