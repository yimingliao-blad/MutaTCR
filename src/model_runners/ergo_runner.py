"""
ERGO Model Runner for TCRP Benchmark V3.

ERGO (Epitope Recognition by Graph Operators) uses autoencoder
or LSTM to predict TCR-peptide binding.

Native Input Format: CSV with 2 columns (tcr, pep) - NO HEADER
Native Output Format: Tab-separated to stdout (tcr\tpep\tprediction)

This runner:
1. Preserves ID column for 1-to-1 merge
2. Runs ERGO in its proper conda environment (ergo-benchmark)
3. Matches predictions back by row index
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import subprocess
import logging
import tempfile
import os

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)


class ERGORunner(BaseModelRunner):
    """
    Runner for ERGO model.

    ERGO uses autoencoder (ae) or LSTM architecture.

    Native interface:
    python ERGO.py predict ae|lstm vdjdb specific cpu --test_data_file=input.csv
    Output is printed to stdout as tab-separated: tcr\tpep\tprediction
    """

    def __init__(self, mode: str = "exact_match", model_type: str = "ae"):
        """
        Initialize ERGO runner.

        Args:
            mode: Deduplication mode
            model_type: 'ae' (autoencoder) or 'lstm'
        """
        super().__init__("ERGO", mode)
        self.model_type = model_type
        self.ergo_path = None
        self._ergo_config = get_config()

    def setup(self) -> bool:
        """
        Set up ERGO model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            ergo_dir = models_dir / "ERGO"

            if not ergo_dir.exists():
                self.logger.error(f"ERGO model not found at {ergo_dir}")
                return False

            self.ergo_path = ergo_dir

            # Check for ERGO.py
            ergo_script = ergo_dir / "ERGO.py"
            if not ergo_script.exists():
                self.logger.error(f"ERGO.py not found at {ergo_script}")
                return False

            # Check conda environment exists
            env_name = self._ergo_config.get_model_env('ERGO')
            env_path = self.paths.CONDA_PATH / "envs" / env_name
            if not env_path.exists():
                self.logger.warning(f"ERGO conda env not found at {env_path}")

            self.logger.info(f"ERGO setup complete: {ergo_dir}")
            return True

        except Exception as e:
            self.logger.error(f"ERGO setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns."""
        return ['Peptide', 'CDR3b']

    def _convert_to_ergo_format(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert input DataFrame to ERGO native format.

        ERGO expects: CSV with 2 columns (tcr, pep) - NO HEADER

        Returns:
            Tuple of (ergo_format_df, index_mapping_df)
        """
        # Keep track of original index
        df_indexed = df.copy()
        df_indexed['_orig_idx'] = range(len(df))

        # Create ERGO format (just tcr and pep columns, no header)
        ergo_df = pd.DataFrame({
            'tcr': df['CDR3b'],
            'pep': df['Peptide'],
            '_orig_idx': df_indexed['_orig_idx']
        })

        return ergo_df, df_indexed

    def _run_ergo_prediction(self, input_file: str) -> Tuple[bool, List[float]]:
        """
        Run ERGO prediction using conda environment.

        Args:
            input_file: Path to input CSV (tcr, pep - no header)

        Returns:
            Tuple of (success, predictions_list)
        """
        ergo_script = self.ergo_path / "ERGO.py"

        # Build command to run in ERGO environment
        # python ERGO.py predict ae vdjdb specific cpu --test_data_file=input.csv
        conda_exe = self.paths.get_conda_executable()
        env_name = self._ergo_config.get_model_env('ERGO')

        # Model file path
        model_file = self.ergo_path / "models" / f"{self.model_type}_vdjdb1.pt"
        ae_file = self.ergo_path / "TCR_Autoencoder" / "tcr_ae_dim_100.pt"

        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", str(ergo_script),
            "predict",
            self.model_type,  # ae or lstm
            "vdjdb",          # dataset (used for model selection)
            "specific",       # sampling strategy
            "cpu",            # device
            f"--test_data_file={input_file}",
            f"--model_file={model_file}",
            f"--ae_file={ae_file}"
        ]

        self.logger.info(f"Running ERGO: {' '.join(cmd[:8])}...")

        try:
            timeout = self._ergo_config.get_model_timeout('ERGO')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.ergo_path),
                timeout=timeout
            )

            if result.returncode != 0:
                self.logger.error(f"ERGO failed: {result.stderr[:500]}")
                return False, []

            # Parse stdout (tab-separated: tcr\tpep\tprediction)
            predictions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        try:
                            predictions.append(float(parts[2]))
                        except ValueError:
                            pass

            self.logger.info(f"ERGO completed: {len(predictions)} predictions")
            return True, predictions

        except subprocess.TimeoutExpired:
            self.logger.error("ERGO prediction timed out")
            return False, []
        except Exception as e:
            self.logger.error(f"ERGO error: {e}")
            return False, []

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using ERGO.

        Strategy:
        1. Convert to ERGO format (tcr, pep CSV without header)
        2. Run prediction
        3. Match predictions back by row order
        4. Return original df with Prediction_Prob added

        Args:
            df: Input DataFrame with ID, Peptide, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        # Start with copy of original (preserves ID and all columns)
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))

        try:
            # Convert to ERGO format
            ergo_df, _ = self._convert_to_ergo_format(df)

            # Create temp input file (NO HEADER, just tcr,pep)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_input = f.name
                # Write without header and without index, just tcr and pep columns
                for _, row in ergo_df.iterrows():
                    f.write(f"{row['tcr']},{row['pep']}\n")

            # Run prediction
            success, predictions = self._run_ergo_prediction(temp_input)

            if success and len(predictions) > 0:
                if len(predictions) == len(df):
                    # Same length - predictions are in order
                    df_result['Prediction_Prob'] = predictions
                    self.logger.info("Matched predictions by position")
                elif len(predictions) < len(df):
                    # ERGO filters out TCRs >= 28 amino acids
                    self.logger.warning(f"Row count mismatch: {len(predictions)} vs {len(df)}")
                    self.logger.warning("Some TCRs may have been filtered (len >= 28)")
                    # Match by rebuilding filter logic
                    filtered_indices = []
                    for i, row in ergo_df.iterrows():
                        if self.model_type == 'ae' and len(row['tcr']) < 28:
                            filtered_indices.append(i)
                        elif self.model_type == 'lstm':
                            filtered_indices.append(i)

                    # Assign predictions to filtered rows, 0.5 to others
                    df_result['Prediction_Prob'] = 0.5
                    for i, pred in zip(filtered_indices[:len(predictions)], predictions):
                        df_result.loc[df_result['_orig_idx'] == i, 'Prediction_Prob'] = pred
                    self.logger.info(f"Assigned {len(predictions)} predictions, {len(df) - len(predictions)} defaulted to 0.5")
                else:
                    self.logger.warning(f"More predictions than inputs: {len(predictions)} vs {len(df)}")
                    df_result['Prediction_Prob'] = predictions[:len(df)]
            else:
                self.logger.error("ERGO prediction failed or no output")
                df_result['Prediction_Prob'] = 0.5

            # Cleanup
            if os.path.exists(temp_input):
                os.unlink(temp_input)

        except Exception as e:
            self.logger.error(f"ERGO prediction error: {e}")
            import traceback
            traceback.print_exc()
            df_result['Prediction_Prob'] = 0.5

        # Remove helper column and restore original order
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to ERGO training data for deduplication."""
        if self.ergo_path:
            training_file = self.ergo_path / "data" / "VDJDB_complete.tsv"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load ERGO training data."""
        path = self.get_training_data_path()
        if path and path.exists():
            return pd.read_csv(path, sep='\t')
        return None
