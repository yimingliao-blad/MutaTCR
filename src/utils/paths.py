"""
Path configuration for TCRP Benchmark V4.

This module provides centralized path management for all data processing,
model execution, and visualization operations.

Integrates with the Config system for environment-aware path resolution.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from .config import get_config
except ImportError:
    from config import get_config


class PathManager:
    """Centralized path management for TCRP Benchmark V4."""

    def __init__(self, base_path: Optional[Path] = None, mode: str = "exact_match"):
        """
        Initialize path manager.

        Args:
            base_path: Base project path. If None, auto-detected from config.
            mode: Deduplication mode - either "exact_match" or "fuzzy_match"
        """
        self._config = get_config()

        # Use config's project root if no base_path provided
        if base_path is None:
            base_path = self._config.project_root

        self.base_path = base_path
        self.mode = mode
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Set up all directory paths using config."""
        # Configuration paths
        self.CONFIG_DIR = self.base_path / "config"

        # Data paths - use config for subdirectory names
        self.DATA_DIR = self.base_path / "data"
        self.RAW_DATA_DIR = self._config.get_data_dir('raw') or (self.DATA_DIR / "raw")
        self.UNIFIED_DATA_DIR = self._config.get_data_dir('unified') or (self.DATA_DIR / "unified")
        self.DEDUP_DATA_DIR = self._config.get_data_dir('deduplicated') or (self.DATA_DIR / "deduplicated")
        self.MODEL_INPUT_DIR = self.DATA_DIR / "model_input"

        # Raw dataset paths (symlinks)
        self.TETTCR_RAW_DIR = self.RAW_DATA_DIR / "TetTCR-SeqHD"
        self.IMMREP23_RAW_DIR = self.RAW_DATA_DIR / "IMMREP23"
        self.FINGERPRINTING_RAW_DIR = self.RAW_DATA_DIR / "FingerPrinting"

        # Model paths - use config
        self.MODELS_DIR = self._config.get_models_dir() or (self.base_path / "models")

        # Conda paths - use config
        self.CONDA_PATH = self._config.get_conda_path() or (self.base_path / "tools" / "miniconda3")

        # Output paths (mode-specific)
        self.OUTPUT_DIR = self._config.get_output_dir() or (self.base_path / "output")
        self.MODE_OUTPUT_DIR = self.OUTPUT_DIR / self.mode
        self.PREDICTIONS_DIR = self.MODE_OUTPUT_DIR
        self.MERGED_DIR = self.MODE_OUTPUT_DIR / "merged"
        self.AGGREGATED_DIR = self.MODE_OUTPUT_DIR / "aggregated"
        self.VISUALIZATIONS_DIR = self.MODE_OUTPUT_DIR / "visualizations"
        self.HEATMAPS_DIR = self.VISUALIZATIONS_DIR / "fingerprinting_heatmaps"

        # Comparison output (cross-mode)
        self.COMPARISON_DIR = self.OUTPUT_DIR / "comparison"

        # Other paths
        self.DOCS_DIR = self.base_path / "docs"
        self.TESTS_DIR = self.base_path / "tests"
        self.ENVS_DIR = self.base_path / "envs"
        self.LOGS_DIR = self._config.get_logs_dir() or (self.base_path / "logs")

    def set_mode(self, mode: str) -> None:
        """
        Change the deduplication mode.

        Args:
            mode: "exact_match" or "fuzzy_match"
        """
        if mode not in ["exact_match", "fuzzy_match"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'exact_match' or 'fuzzy_match'")
        self.mode = mode
        self._setup_paths()

    # Conda executable
    def get_conda_executable(self) -> Path:
        """Get path to conda executable."""
        return self.CONDA_PATH / "bin" / "conda"

    # Raw data file paths
    def get_tettcr_tcr_file(self) -> Path:
        """Get path to Kevin's TCR annotations file."""
        return self.TETTCR_RAW_DIR / "Kevin's Publication TCRs Updated.xlsx"

    def get_tettcr_umi_file(self) -> Path:
        """Get path to UMI count matrix file."""
        return self.TETTCR_RAW_DIR / "41590_2021_1073_MOESM12_ESM.xlsx"

    def get_immrep23_file(self) -> Path:
        """Get path to IMMREP23 dataset file."""
        return self.IMMREP23_RAW_DIR / "immref23.csv"

    def get_fingerprinting_file(self) -> Path:
        """Get path to FingerPrinting dataset file."""
        return self.FINGERPRINTING_RAW_DIR / "Fingerprinting_TCR_Data_Combined_final.xlsx"

    def get_fingerprinting_log2fc_file(self) -> Path:
        """Get path to log2fc reference file."""
        # Located at project root, not in raw data
        return self.base_path.parent / "finger_print.xlsx"

    # Unified data paths
    def get_unified_file(self, dataset: str) -> Path:
        """
        Get path to unified CSV file.

        Args:
            dataset: "tettcr", "immrep23", or "fingerprinting"
        """
        return self.UNIFIED_DATA_DIR / f"{dataset}_unified.csv"

    # Deduplicated data paths
    def get_dedup_file(self, model: str, dataset: str) -> Path:
        """
        Get path to deduplicated data file for a model.

        Args:
            model: Model name
            dataset: Dataset name
        """
        return self.DEDUP_DATA_DIR / self.mode / model / f"{dataset}_dedup.csv"

    # Model prediction paths
    def get_model_predictions_dir(self, model: str) -> Path:
        """
        Get path to model predictions directory.

        Args:
            model: Model name (e.g., "ERGO", "ERGO2")
        """
        return self.MODE_OUTPUT_DIR / model / "predictions"

    def get_model_prediction_file(self, model: str, dataset: str) -> Path:
        """
        Get path to specific model prediction file.

        Args:
            model: Model name
            dataset: Dataset name
        """
        return self.get_model_predictions_dir(model) / f"{dataset}_predictions.csv"

    # Model directory paths
    def get_model_dir(self, model: str) -> Path:
        """
        Get path to model installation directory.

        Args:
            model: Model name
        """
        # Handle folder name variations
        folder_names = {
            'ERGO': 'ERGO',
            'ERGO2': 'ERGO2',
            'NetTCR': 'NetTCR',
            'NetTCR22': 'NetTCR22',
            'TITAN': 'TITAN',
            'EPACT': 'EPACT',
            'PanPep': 'PanPep',
            'SCEPTR': 'SCEPTR'
        }
        folder = folder_names.get(model, model)
        return self.MODELS_DIR / folder

    # Merged data paths
    def get_merged_file(self, dataset: str) -> Path:
        """
        Get path to merged predictions file (all models + unified data).

        Args:
            dataset: Dataset name
        """
        return self.MERGED_DIR / f"{dataset}_all_models.csv"

    # Aggregated results paths
    def get_metrics_summary_file(self) -> Path:
        """Get path to metrics summary file."""
        return self.AGGREGATED_DIR / "metrics_summary.csv"

    def get_per_peptide_auc_file(self) -> Path:
        """Get path to per-peptide AUC file."""
        return self.AGGREGATED_DIR / "per_peptide_auc.csv"

    # Visualization paths
    def get_visualization_file(self, name: str) -> Path:
        """
        Get path to visualization file.

        Args:
            name: Visualization filename (e.g., "auc_boxplot_tettcr_viral.png")
        """
        return self.VISUALIZATIONS_DIR / name

    def get_heatmap_file(self, model: str) -> Path:
        """
        Get path to fingerprinting heatmap file.

        Args:
            model: Model name
        """
        return self.HEATMAPS_DIR / f"{model}_heatmap.png"

    # Directory creation
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.CONFIG_DIR,
            self.DATA_DIR,
            self.UNIFIED_DATA_DIR,
            self.DEDUP_DATA_DIR / "exact_match",
            self.DEDUP_DATA_DIR / "fuzzy_match",
            self.MODEL_INPUT_DIR,
            self.MODELS_DIR,
            self.OUTPUT_DIR / "exact_match" / "merged",
            self.OUTPUT_DIR / "exact_match" / "aggregated",
            self.OUTPUT_DIR / "exact_match" / "visualizations" / "fingerprinting_heatmaps",
            self.OUTPUT_DIR / "fuzzy_match" / "merged",
            self.OUTPUT_DIR / "fuzzy_match" / "aggregated",
            self.OUTPUT_DIR / "fuzzy_match" / "visualizations" / "fingerprinting_heatmaps",
            self.COMPARISON_DIR / "summary_tables",
            self.DOCS_DIR,
            self.TESTS_DIR / "fixtures",
            self.ENVS_DIR,
            self.LOGS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_model_dirs(self, models: list) -> None:
        """
        Create model-specific directories for both modes.

        Args:
            models: List of model names
        """
        for mode in ["exact_match", "fuzzy_match"]:
            for model in models:
                (self.OUTPUT_DIR / mode / model / "predictions").mkdir(parents=True, exist_ok=True)
                (self.DEDUP_DATA_DIR / mode / model).mkdir(parents=True, exist_ok=True)


# Global path manager instance
_path_manager: Optional[PathManager] = None


def get_paths(base_path: Optional[Path] = None, mode: str = "exact_match") -> PathManager:
    """
    Get path manager instance for a specific mode.

    Note: Each mode should use its own PathManager to avoid state issues.
    This function creates a new PathManager when the mode changes.

    Args:
        base_path: Base project path (optional)
        mode: Deduplication mode ("exact_match" or "fuzzy_match")

    Returns:
        PathManager instance configured for the specified mode
    """
    global _path_manager

    # Always create new instance when mode differs or base_path specified
    if _path_manager is None or base_path is not None or _path_manager.mode != mode:
        _path_manager = PathManager(base_path, mode)

    return _path_manager


# Convenience function for getting paths from environment variable
def get_paths_from_env() -> PathManager:
    """
    Get path manager with mode from MATCHING_MODE environment variable.

    Returns:
        PathManager instance
    """
    mode = os.environ.get("MATCHING_MODE", "exact_match")
    return get_paths(mode=mode)


if __name__ == "__main__":
    # Test path manager
    print("Testing PathManager...")

    paths = PathManager()
    print(f"Base path: {paths.base_path}")
    print(f"Mode: {paths.mode}")
    print(f"\nKey paths:")
    print(f"  Config: {paths.CONFIG_DIR}")
    print(f"  Conda: {paths.CONDA_PATH}")
    print(f"  Models: {paths.MODELS_DIR}")
    print(f"  Unified data: {paths.UNIFIED_DATA_DIR}")
    print(f"  Predictions: {paths.PREDICTIONS_DIR}")
    print(f"  Visualizations: {paths.VISUALIZATIONS_DIR}")
    print(f"  Heatmaps: {paths.HEATMAPS_DIR}")

    print(f"\nRaw data files:")
    print(f"  TetTCR TCRs: {paths.get_tettcr_tcr_file()}")
    print(f"  TetTCR UMI: {paths.get_tettcr_umi_file()}")
    print(f"  IMMREP23: {paths.get_immrep23_file()}")
    print(f"  FingerPrinting: {paths.get_fingerprinting_file()}")

    print(f"\nModel directories:")
    for model in ['ERGO', 'NetTCR', 'TITAN', 'EPACT', 'PanPep']:
        print(f"  {model}: {paths.get_model_dir(model)}")

    print(f"\nModel paths:")
    print(f"  ERGO predictions: {paths.get_model_predictions_dir('ERGO')}")
    print(f"  ERGO tettcr file: {paths.get_model_prediction_file('ERGO', 'tettcr')}")

    print("\nPath manager test completed!")
