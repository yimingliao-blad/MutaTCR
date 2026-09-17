"""
Base Model Runner for TCRP Benchmark V4.

Abstract base class that all model runners inherit from.
Provides common functionality for data loading, prediction saving, etc.

V4 Features:
- Config-based path resolution
- Configurable timeouts, GPU settings
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time

try:
    from ..utils.paths import get_paths
    from ..utils.config import get_config
except ImportError:
    from utils.paths import get_paths
    from utils.config import get_config

logger = logging.getLogger(__name__)


class BaseModelRunner(ABC):
    """
    Abstract base class for model runners.

    Each model runner must implement:
    - setup(): Install dependencies and load model
    - predict(df): Generate predictions for a DataFrame
    - get_required_columns(): Return list of required input columns
    """

    def __init__(self, model_key: str, mode: str = "exact_match"):
        """
        Initialize model runner.

        Args:
            model_key: Model key (e.g., "ERGO", "NetTCR22")
            mode: Deduplication mode
        """
        self.model_key = model_key
        self.mode = mode
        self.paths = get_paths(mode=mode)
        self._config = get_config()

        self.logger = logging.getLogger(f"tcrp_benchmark.{model_key}")
        self._model = None
        self._is_setup = False

    @property
    def model_name(self) -> str:
        """Get model display name."""
        return self._config.get_model_display_name(self.model_key) or self.model_key

    @property
    def env_name(self) -> str:
        """Get conda environment name."""
        return self._config.get_model_env(self.model_key)

    @property
    def timeout(self) -> int:
        """Get model timeout in seconds."""
        return self._config.get_model_timeout(self.model_key)

    @abstractmethod
    def setup(self) -> bool:
        """
        Set up the model (load weights, initialize, etc.).

        Returns:
            True if setup successful, False otherwise
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for input data.

        Args:
            df: Input DataFrame with required columns

        Returns:
            DataFrame with predictions (must include 'Prediction_Prob' column)
        """
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required input columns.

        Returns:
            List of column names
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame has required columns.

        Args:
            df: Input DataFrame

        Returns:
            True if valid, raises ValueError otherwise
        """
        required = self.get_required_columns()
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns for {self.model_name}: {missing}")

        return True

    def load_unified_data(self, dataset: str) -> pd.DataFrame:
        """
        Load unified data for a dataset.

        Args:
            dataset: Dataset name ('tettcr', 'immrep23', 'fingerprinting')

        Returns:
            DataFrame
        """
        unified_file = self.paths.get_unified_file(dataset)

        if not unified_file.exists():
            raise FileNotFoundError(f"Unified file not found: {unified_file}")

        df = pd.read_csv(unified_file)
        self.logger.info(f"Loaded {len(df)} records from {dataset}")

        return df

    def save_predictions(self, df: pd.DataFrame, dataset: str) -> Path:
        """
        Save predictions to output file.

        Args:
            df: DataFrame with predictions
            dataset: Dataset name

        Returns:
            Path to saved file
        """
        output_dir = self.paths.get_model_predictions_dir(self.model_key)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{dataset}_predictions.csv"
        df.to_csv(output_file, index=False)

        self.logger.info(f"Saved predictions to {output_file}")
        return output_file

    def run_dataset(self, dataset: str) -> Optional[Path]:
        """
        Run predictions for a single dataset.

        Args:
            dataset: Dataset name

        Returns:
            Path to predictions file, or None if failed
        """
        self.logger.info(f"Processing {dataset} with {self.model_name}...")

        try:
            # Load data
            df = self.load_unified_data(dataset)

            # Validate
            self.validate_input(df)

            # Predict
            start_time = time.time()
            df_pred = self.predict(df)
            elapsed = time.time() - start_time

            self.logger.info(f"Prediction completed in {elapsed:.1f}s")

            # Save
            output_path = self.save_predictions(df_pred, dataset)

            return output_path

        except Exception as e:
            self.logger.error(f"Error processing {dataset}: {e}")
            return None

    def run_all_datasets(self) -> Dict[str, Optional[Path]]:
        """
        Run predictions for all datasets.

        Returns:
            Dictionary mapping dataset names to output paths
        """
        if not self._is_setup:
            self.logger.info("Setting up model...")
            if not self.setup():
                self.logger.error("Model setup failed")
                return {}
            self._is_setup = True

        datasets = ['tettcr', 'immrep23', 'fingerprinting']
        results = {}

        for dataset in datasets:
            results[dataset] = self.run_dataset(dataset)

        return results


class DummyModelRunner(BaseModelRunner):
    """
    Dummy model runner for testing purposes.

    Generates random predictions.
    """

    def setup(self) -> bool:
        """Setup dummy model (always succeeds)."""
        self.logger.info("Dummy model setup complete")
        return True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate random predictions."""
        df_pred = df.copy()
        df_pred['Prediction_Prob'] = np.random.random(len(df))
        return df_pred

    def get_required_columns(self) -> List[str]:
        """Dummy model requires only Peptide and CDR3b."""
        return ['Peptide', 'CDR3b']


def create_runner(model_key: str, mode: str = "exact_match") -> BaseModelRunner:
    """
    Factory function to create appropriate model runner.

    Args:
        model_key: Model key
        mode: Deduplication mode

    Returns:
        ModelRunner instance
    """
    # Import specific runners here to avoid circular imports
    runners = {
        'ERGO': 'src.model_runners.ergo_runner.ERGORunner',
        'ERGO2': 'src.model_runners.ergo2_runner.ERGO2Runner',
        'NetTCR': 'src.model_runners.nettcr_runner.NetTCRRunner',
        'NetTCR22': 'src.model_runners.nettcr22_runner.NetTCR22Runner',
        'TITAN': 'src.model_runners.titan_runner.TITANRunner',
        'EPACT': 'src.model_runners.epact_runner.EPACTRunner',
        'PanPep': 'src.model_runners.panpep_runner.PanPepRunner',
        'SCEPTR': 'src.model_runners.sceptr_runner.SCEPTRRunner',
    }

    if model_key not in runners:
        logger.warning(f"No specific runner for {model_key}, using DummyRunner")
        return DummyModelRunner(model_key, mode)

    # Dynamic import
    module_path, class_name = runners[model_key].rsplit('.', 1)
    try:
        import importlib
        module = importlib.import_module(module_path)
        runner_class = getattr(module, class_name)
        return runner_class(mode)
    except ImportError as e:
        logger.warning(f"Could not import runner for {model_key}: {e}")
        return DummyModelRunner(model_key, mode)


if __name__ == "__main__":
    # Test dummy runner
    logging.basicConfig(level=logging.INFO)

    runner = DummyModelRunner("ERGO", "exact_match")
    print(f"Model: {runner.model_name}")
    print(f"Env: {runner.env_name}")
    print(f"Timeout: {runner.timeout}s")
    print(f"Required columns: {runner.get_required_columns()}")

    # Test with sample data
    test_df = pd.DataFrame({
        'Peptide': ['YLQPRTFLL', 'GILGFVFTL'],
        'CDR3b': ['CASSQDRG', 'CASSLVGG'],
        'Label': [1, 0]
    })

    runner.setup()
    result = runner.predict(test_df)
    print(f"\nPredictions:\n{result[['Peptide', 'CDR3b', 'Prediction_Prob']]}")
