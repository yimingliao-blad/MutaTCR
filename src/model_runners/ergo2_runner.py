"""
ERGO2 Model Runner for TCRP Benchmark V3.

ERGO2 uses deep learning with multiple feature configurations
(TCRb, +TCRa, +V,J genes) for TCR-peptide binding prediction.

Native Input Format: CSV with columns TRA,TRB,TRAV,TRAJ,TRBV,TRBJ,T-Cell-Type,Peptide,MHC
Native Output Format: Input DataFrame with 'Score' column added

This runner:
1. Preserves ID column for 1-to-1 merge
2. Runs ERGO2 in its proper conda environment (ergo2-benchmark)
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
import pickle

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)


class ERGO2Runner(BaseModelRunner):
    """
    Runner for ERGO2 model.

    ERGO2 uses deep learning with multiple feature configurations.

    Native interface:
    python Predict.py vdjdb|mcpas input.csv
    Output is printed to stdout (DataFrame with Score column)
    """

    def __init__(self, mode: str = "exact_match", dataset: str = "vdjdb"):
        """
        Initialize ERGO2 runner.

        Args:
            mode: Deduplication mode
            dataset: 'vdjdb' or 'mcpas' for model selection
        """
        super().__init__("ERGO2", mode)
        self.dataset = dataset
        self.ergo2_path = None
        self._ergo2_config = get_config()

    def setup(self) -> bool:
        """
        Set up ERGO2 model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            ergo2_dir = models_dir / "ERGO2"

            if not ergo2_dir.exists():
                self.logger.error(f"ERGO2 model not found at {ergo2_dir}")
                return False

            self.ergo2_path = ergo2_dir

            # Check for Predict.py
            predict_script = ergo2_dir / "Predict.py"
            if not predict_script.exists():
                self.logger.error(f"Predict.py not found at {predict_script}")
                return False

            # Check conda environment exists
            env_name = self._ergo2_config.get_model_env('ERGO2')
            env_path = self.paths.CONDA_PATH / "envs" / env_name
            if not env_path.exists():
                self.logger.warning(f"ERGO2 conda env not found at {env_path}")

            self.logger.info(f"ERGO2 setup complete: {ergo2_dir}")
            return True

        except Exception as e:
            self.logger.error(f"ERGO2 setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns."""
        return ['Peptide', 'CDR3b']

    def _convert_to_ergo2_format(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert input DataFrame to ERGO2 native format.

        ERGO2 expects: TRA, TRB, TRAV, TRAJ, TRBV, TRBJ, T-Cell-Type, Peptide, MHC

        Returns:
            Tuple of (ergo2_format_df, index_mapping_df)
        """
        # Keep track of original index
        df_indexed = df.copy()
        df_indexed['_orig_idx'] = range(len(df))

        # Create ERGO2 format
        ergo2_df = pd.DataFrame({
            'TRA': df.get('CDR3a', 'UNK'),
            'TRB': df['CDR3b'],
            'TRAV': df.get('Va', 'UNK'),
            'TRAJ': df.get('Ja', 'UNK'),
            'TRBV': df.get('Vb', 'UNK'),
            'TRBJ': df.get('Jb', 'UNK'),
            'T-Cell-Type': 'CD8',  # Default T-cell type
            'Peptide': df['Peptide'],
            'MHC': df.get('HLA', 'HLA-A*02:01'),
            '_orig_idx': df_indexed['_orig_idx']
        })

        # Fill NaN values
        ergo2_df['TRA'] = ergo2_df['TRA'].fillna('UNK')
        ergo2_df['TRAV'] = ergo2_df['TRAV'].fillna('UNK')
        ergo2_df['TRAJ'] = ergo2_df['TRAJ'].fillna('UNK')
        ergo2_df['TRBV'] = ergo2_df['TRBV'].fillna('UNK')
        ergo2_df['TRBJ'] = ergo2_df['TRBJ'].fillna('UNK')
        ergo2_df['MHC'] = ergo2_df['MHC'].fillna('HLA-A*02:01')

        return ergo2_df, df_indexed

    def _run_ergo2_prediction(self, input_file: str, output_file: str) -> bool:
        """
        Run ERGO2 prediction using conda environment.

        Args:
            input_file: Path to input CSV
            output_file: Path to output CSV

        Returns:
            True if prediction successful
        """
        # Build command to run in ERGO2 environment
        conda_exe = self.paths.get_conda_executable()
        env_name = self._ergo2_config.get_model_env('ERGO2')

        # ERGO2 needs to run from its own directory because it looks for
        # relative paths like TCR_Autoencoder/tcra_ae_dim_100.pt
        # Create a temp script to ensure correct directory
        script_content = f"""
import os
import sys
os.chdir('{self.ergo2_path}')
sys.path.insert(0, '.')
from Predict import predict
df = predict('{self.dataset}', '{input_file}')
df.to_csv('{output_file}', index=False)
"""
        temp_script = tempfile.mktemp(suffix='.py')
        with open(temp_script, 'w') as f:
            f.write(script_content)

        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", temp_script
        ]

        self.logger.info(f"Running ERGO2: predict({self.dataset}, input)...")

        try:
            timeout = self._ergo2_config.get_model_timeout('ERGO2')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Cleanup temp script
            if os.path.exists(temp_script):
                os.unlink(temp_script)

            if result.returncode != 0:
                self.logger.error(f"ERGO2 failed: {result.stderr}")
                return False

            self.logger.info("ERGO2 prediction completed")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("ERGO2 prediction timed out")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return False
        except Exception as e:
            self.logger.error(f"ERGO2 error: {e}")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using ERGO2.

        Strategy:
        1. Convert to ERGO2 format with _orig_idx
        2. Run prediction
        3. Match predictions back using _orig_idx or row position
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
            # Convert to ERGO2 format
            ergo2_df, _ = self._convert_to_ergo2_format(df)

            # Create temp input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_input = f.name
                ergo2_df.to_csv(f, index=False)

            temp_output = tempfile.mktemp(suffix='.csv')

            # Run prediction
            success = self._run_ergo2_prediction(temp_input, temp_output)

            if success and os.path.exists(temp_output):
                pred_df = pd.read_csv(temp_output)
                self.logger.info(f"Loaded {len(pred_df)} predictions")

                # Check for Score column
                if 'Score' in pred_df.columns:
                    if '_orig_idx' in pred_df.columns:
                        # Match by original index
                        pred_df = pred_df.sort_values('_orig_idx')
                        df_result = df_result.sort_values('_orig_idx')
                        df_result['Prediction_Prob'] = pred_df['Score'].values
                        self.logger.info("Matched predictions by _orig_idx")
                    elif len(pred_df) == len(df):
                        # Same length - assume same order
                        df_result['Prediction_Prob'] = pred_df['Score'].values
                        self.logger.info("Matched predictions by position")
                    else:
                        # Row count mismatch - some samples were filtered
                        self.logger.warning(f"Row count mismatch: {len(pred_df)} vs {len(df)}")
                        # ERGO2 filters invalid sequences, try matching by TRB + Peptide
                        pred_dict = {}
                        for _, row in pred_df.iterrows():
                            key = (row.get('TRB', ''), row.get('Peptide', ''))
                            pred_dict[key] = row['Score']

                        predictions = []
                        for _, row in df_result.iterrows():
                            key = (row['CDR3b'], row['Peptide'])
                            predictions.append(pred_dict.get(key, 0.5))
                        df_result['Prediction_Prob'] = predictions
                        self.logger.info("Matched predictions by (CDR3b, Peptide)")
                else:
                    self.logger.error(f"No 'Score' column. Columns: {list(pred_df.columns)}")
                    df_result['Prediction_Prob'] = 0.5

                # Cleanup
                try:
                    os.unlink(temp_output)
                except:
                    pass
            else:
                self.logger.error("ERGO2 prediction failed or no output")
                df_result['Prediction_Prob'] = 0.5

            # Cleanup input file
            if os.path.exists(temp_input):
                os.unlink(temp_input)

        except Exception as e:
            self.logger.error(f"ERGO2 prediction error: {e}")
            import traceback
            traceback.print_exc()
            df_result['Prediction_Prob'] = 0.5

        # Remove helper column and restore original order
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to ERGO2 training data for deduplication."""
        if self.ergo2_path:
            training_file = self.ergo2_path / "Samples" / f"{self.dataset}_train_samples.pickle"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load ERGO2 training data."""
        path = self.get_training_data_path()
        if path and path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # Convert to DataFrame
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, list):
                return pd.DataFrame(data)
        return None
