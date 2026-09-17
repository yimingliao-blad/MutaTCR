"""
NetTCR-2.2 Model Runner for TCRP Benchmark V4.

NetTCR-2.2 uses all 6 CDR loops for prediction.

Native Input Format: CSV with columns A1, A2, A3, B1, B2, B3, peptide
Native Output Format: Input + 'prediction' column

This runner:
1. Preserves ID column for 1-to-1 merge
2. Uses 20-model ensemble (5 folds × 4 validation splits) like the original webserver
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
import shutil
import sys

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)


class NetTCR22Runner(BaseModelRunner):
    """
    Runner for NetTCR-2.2 model.

    NetTCR-2.2 requires all 6 CDR loops (A1, A2, A3, B1, B2, B3).

    Uses 20-model ensemble (5 folds × 4 validation splits) like the original webserver.
    """

    def __init__(self, mode: str = "exact_match", model_type: str = "pan", use_ensemble: bool = True):
        """
        Initialize NetTCR-2.2 runner.

        Args:
            mode: Deduplication mode
            model_type: 'pan', 'peptide', or 'pretrained'
            use_ensemble: If True, use 20-model ensemble (like original webserver).
                         If False, use single model t.0.v.1 (legacy behavior).
        """
        super().__init__("NetTCR22", mode)
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        self.trained_model_name = "t.0.v.1"  # Fallback for single-model mode
        # Ensemble: 5 folds × 4 validation splits = 20 models
        self.train_parts = {0, 1, 2, 3, 4}
        self.nettcr22_path = None
        self._nettcr22_config = get_config()

    def setup(self) -> bool:
        """
        Set up NetTCR-2.2 model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            nettcr22_dir = models_dir / "NETTCR22"

            if not nettcr22_dir.exists():
                nettcr22_dir = models_dir / "NetTCR22"

            if not nettcr22_dir.exists():
                self.logger.error(f"NetTCR-2.2 model not found")
                return False

            self.nettcr22_path = nettcr22_dir

            # Check if predict.py exists
            predict_script = nettcr22_dir / "src" / "predict.py"
            if not predict_script.exists():
                self.logger.error(f"predict.py not found at {predict_script}")
                return False

            # Check conda environment exists
            env_name = self._nettcr22_config.get_model_env('NetTCR22')
            env_path = self.paths.CONDA_PATH / "envs" / env_name
            if not env_path.exists():
                self.logger.warning(f"NetTCR-2.2 conda env not found at {env_path}")

            self.logger.info(f"NetTCR-2.2 setup complete: {nettcr22_dir}")
            return True

        except Exception as e:
            self.logger.error(f"NetTCR-2.2 setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns."""
        return ['Peptide', 'CDR3a', 'CDR3b']

    def _convert_to_nettcr22_format(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert input DataFrame to NetTCR-2.2 native format.

        NetTCR-2.2 expects: A1, A2, A3, B1, B2, B3, peptide, binder
        Where A1-A3 are CDR1a-CDR3a, B1-B3 are CDR1b-CDR3b

        Returns:
            Tuple of (nettcr22_format_df, index_mapping)
        """
        # Keep track of original index
        df_indexed = df.copy()
        df_indexed['_orig_idx'] = range(len(df))

        nettcr22_df = pd.DataFrame()

        # CDR1a, CDR2a - default to 'A' if not available
        nettcr22_df['A1'] = df.get('CDR1a', pd.Series(['A'] * len(df)))
        nettcr22_df['A1'] = nettcr22_df['A1'].fillna('A').replace('', 'A')

        nettcr22_df['A2'] = df.get('CDR2a', pd.Series(['A'] * len(df)))
        nettcr22_df['A2'] = nettcr22_df['A2'].fillna('A').replace('', 'A')

        # CDR3a - required, fill NaN with 'A'
        nettcr22_df['A3'] = df.get('CDR3a', pd.Series(['A'] * len(df)))
        nettcr22_df['A3'] = nettcr22_df['A3'].fillna('A').replace('', 'A')

        # CDR1b, CDR2b - default to 'A' if not available
        nettcr22_df['B1'] = df.get('CDR1b', pd.Series(['A'] * len(df)))
        nettcr22_df['B1'] = nettcr22_df['B1'].fillna('A').replace('', 'A')

        nettcr22_df['B2'] = df.get('CDR2b', pd.Series(['A'] * len(df)))
        nettcr22_df['B2'] = nettcr22_df['B2'].fillna('A').replace('', 'A')

        # CDR3b - required
        nettcr22_df['B3'] = df['CDR3b']

        # Peptide and label
        nettcr22_df['peptide'] = df['Peptide']
        nettcr22_df['binder'] = df.get('Label', 1)

        # Add original index for matching back
        nettcr22_df['_orig_idx'] = df_indexed['_orig_idx']

        return nettcr22_df, df_indexed

    def _convert_to_webserver_format(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert input DataFrame to make_webserver_prediction.py format.

        The webserver script expects:
        - NO header
        - Columns in order: peptide, A1, A2, A3, B1, B2, B3, binder

        Returns:
            Tuple of (webserver_format_df, index_mapping)
        """
        df_indexed = df.copy()
        df_indexed['_orig_idx'] = range(len(df))

        # Create DataFrame with exact column order expected by webserver
        webserver_df = pd.DataFrame()
        webserver_df['peptide'] = df['Peptide']

        # CDR1a, CDR2a - default to 'A' if not available
        webserver_df['A1'] = df.get('CDR1a', pd.Series(['A'] * len(df)))
        webserver_df['A1'] = webserver_df['A1'].fillna('A').replace('', 'A')

        webserver_df['A2'] = df.get('CDR2a', pd.Series(['A'] * len(df)))
        webserver_df['A2'] = webserver_df['A2'].fillna('A').replace('', 'A')

        # CDR3a - required, fill NaN with 'A'
        webserver_df['A3'] = df.get('CDR3a', pd.Series(['A'] * len(df)))
        webserver_df['A3'] = webserver_df['A3'].fillna('A').replace('', 'A')

        # CDR1b, CDR2b - default to 'A' if not available
        webserver_df['B1'] = df.get('CDR1b', pd.Series(['A'] * len(df)))
        webserver_df['B1'] = webserver_df['B1'].fillna('A').replace('', 'A')

        webserver_df['B2'] = df.get('CDR2b', pd.Series(['A'] * len(df)))
        webserver_df['B2'] = webserver_df['B2'].fillna('A').replace('', 'A')

        # CDR3b - required
        webserver_df['B3'] = df['CDR3b']

        # Binder label
        webserver_df['binder'] = df.get('Label', 1)

        return webserver_df, df_indexed

    def _run_nettcr22_prediction(self, input_file: str, model_name: str) -> Tuple[bool, Path]:
        """
        Run NetTCR-2.2 prediction using conda environment.

        Args:
            input_file: Path to input CSV
            model_name: Model checkpoint name (e.g., 't.0.v.1')

        Returns:
            Tuple of (success, output_file_path)
        """
        predict_script = self.nettcr22_path / "src" / "predict.py"

        # Get model directory based on model_type
        # For pan: models/nettcr_2_2_pan (contains checkpoint/*.tflite)
        # For peptide: models/nettcr_2_2_peptide
        # For pretrained: models/nettcr_2_2_pretrained
        model_dir = self.nettcr22_path / "models" / f"nettcr_2_2_{self.model_type}"

        # Build command to run in NetTCR-2.2 environment
        conda_exe = self.paths.get_conda_executable()
        env_name = self._nettcr22_config.get_model_env('NetTCR22')

        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", str(predict_script),
            "--test_data", input_file,
            "--outdir", str(model_dir),
            "--model_name", model_name,
            "--model_type", self.model_type
        ]

        # Expected output file
        output_file = model_dir / f"{model_name}_prediction.csv"

        self.logger.info(f"Running NetTCR-2.2: {' '.join(cmd[:8])}...")

        try:
            timeout = self._nettcr22_config.get_model_timeout('NetTCR22')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.nettcr22_path),
                timeout=timeout
            )

            if result.returncode != 0:
                self.logger.error(f"NetTCR-2.2 failed: {result.stderr[:500]}")
                return False, output_file

            self.logger.info("NetTCR-2.2 prediction completed")
            return True, output_file

        except subprocess.TimeoutExpired:
            self.logger.error("NetTCR-2.2 prediction timed out")
            return False, output_file
        except Exception as e:
            self.logger.error(f"NetTCR-2.2 error: {e}")
            return False, output_file

    def _run_ensemble_prediction(self, input_file: str) -> Tuple[bool, Path]:
        """
        Run NetTCR-2.2 20-model ensemble prediction using make_webserver_prediction.py.

        This uses the original webserver script which:
        - Loads all 20 models (5 folds × 4 validation splits)
        - Averages predictions across all models
        - Produces results identical to the official NetTCR-2.2 webserver

        Args:
            input_file: Path to input CSV with A1-A3, B1-B3, peptide, binder columns

        Returns:
            Tuple of (success, output_file_path)
        """
        webserver_script = self.nettcr22_path / "src" / "make_webserver_prediction.py"

        if not webserver_script.exists():
            self.logger.warning("make_webserver_prediction.py not found, falling back to single model")
            return False, None

        # Create output directory for webserver predictions
        output_dir = self.nettcr22_path / "webserver_output"
        output_dir.mkdir(exist_ok=True)

        output_filename = "nettcr22_ensemble_predictions.csv"

        # Build command to run in NetTCR-2.2 environment
        conda_exe = self.paths.get_conda_executable()
        env_name = self._nettcr22_config.get_model_env('NetTCR22')

        # make_webserver_prediction.py arguments:
        # -d/--dir: GitHub repo directory (where models/ folder is)
        # -i/--infile: input CSV
        # -o/--output_dir: output directory
        # --output_file: output filename
        # -a/--alpha: TCRbase scaling factor (0 = disabled)
        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", str(webserver_script),
            "-d", str(self.nettcr22_path),
            "-i", input_file,
            "-o", str(output_dir),
            "--output_file", output_filename,
            "-a", "0"  # Disable TCRbase scaling for fair comparison
        ]

        # Expected output file
        output_file = output_dir / output_filename

        self.logger.info(f"Running NetTCR-2.2 20-model ensemble...")

        try:
            timeout = self._nettcr22_config.get_model_timeout('NetTCR22')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.nettcr22_path),
                timeout=timeout
            )

            if result.returncode != 0:
                self.logger.error(f"NetTCR-2.2 ensemble failed: {result.stderr[:500]}")
                return False, output_file

            self.logger.info("NetTCR-2.2 ensemble prediction completed (20 models)")
            return True, output_file

        except subprocess.TimeoutExpired:
            self.logger.error("NetTCR-2.2 ensemble prediction timed out")
            return False, output_file
        except Exception as e:
            self.logger.error(f"NetTCR-2.2 ensemble error: {e}")
            return False, output_file

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using NetTCR-2.2.

        Strategy:
        1. Convert to NetTCR-2.2 format with _orig_idx
        2. Run prediction (ensemble of 20 models or single model)
        3. Match predictions back using _orig_idx or row position
        4. Return original df with Prediction_Prob added

        Args:
            df: Input DataFrame with ID, Peptide, CDR3a, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        # Start with copy of original (preserves ID and all columns)
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))

        try:
            # Handle missing CDR3a
            if 'CDR3a' not in df.columns:
                df_temp = df.copy()
                df_temp['CDR3a'] = 'A'
            else:
                df_temp = df

            # Choose prediction method: ensemble (20 models) or single model
            if self.use_ensemble:
                self.logger.info("Using 20-model ensemble prediction (like original webserver)")

                # Convert to webserver format (no header, specific column order)
                webserver_df, _ = self._convert_to_webserver_format(df_temp)

                # Create temp input file WITHOUT header (as webserver script expects)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    temp_input = f.name
                    webserver_df.to_csv(f, index=False, header=False)

                success, pred_file = self._run_ensemble_prediction(temp_input)
                if not success or pred_file is None or not pred_file.exists():
                    # Fallback to single model if ensemble fails
                    self.logger.warning("Ensemble failed, falling back to single model")
                    # Re-convert to nettcr22 format for predict.py
                    nettcr22_df, _ = self._convert_to_nettcr22_format(df_temp)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                        temp_input = f.name
                        nettcr22_df.to_csv(f, index=False)
                    model_name = self.trained_model_name
                    success, pred_file = self._run_nettcr22_prediction(temp_input, model_name)
            else:
                # Legacy single-model behavior using predict.py
                nettcr22_df, _ = self._convert_to_nettcr22_format(df_temp)

                # Create temp input file with header for predict.py
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    temp_input = f.name
                    nettcr22_df.to_csv(f, index=False)

                model_name = self.trained_model_name  # e.g., 't.0.v.1'
                success, pred_file = self._run_nettcr22_prediction(temp_input, model_name)

            if success and pred_file.exists():
                pred_df = pd.read_csv(pred_file)
                self.logger.info(f"Loaded {len(pred_df)} predictions")

                # Check for prediction column
                if 'prediction' in pred_df.columns:
                    if '_orig_idx' in pred_df.columns:
                        # Match by original index
                        pred_df = pred_df.sort_values('_orig_idx')
                        df_result = df_result.sort_values('_orig_idx')
                        df_result['Prediction_Prob'] = pred_df['prediction'].values
                        self.logger.info("Matched predictions by _orig_idx")
                    elif len(pred_df) == len(df):
                        # Same length - assume same order
                        df_result['Prediction_Prob'] = pred_df['prediction'].values
                        self.logger.info("Matched predictions by position")
                    else:
                        # Row count mismatch - try matching by B3 + peptide
                        self.logger.warning(f"Row count mismatch: {len(pred_df)} vs {len(df)}")
                        pred_dict = {}
                        for _, row in pred_df.iterrows():
                            key = (row.get('B3', ''), row.get('peptide', ''))
                            pred_dict[key] = row['prediction']

                        predictions = []
                        for _, row in df_result.iterrows():
                            key = (row['CDR3b'], row['Peptide'])
                            predictions.append(pred_dict.get(key, 0.5))
                        df_result['Prediction_Prob'] = predictions
                        self.logger.info("Matched predictions by (CDR3b, Peptide)")
                else:
                    self.logger.error(f"No 'prediction' column. Columns: {list(pred_df.columns)}")
                    df_result['Prediction_Prob'] = 0.5

                # Cleanup prediction file
                try:
                    os.unlink(pred_file)
                except:
                    pass
            else:
                self.logger.error("NetTCR-2.2 prediction failed or no output")
                df_result['Prediction_Prob'] = 0.5

            # Cleanup input file
            if os.path.exists(temp_input):
                os.unlink(temp_input)

        except Exception as e:
            self.logger.error(f"NetTCR-2.2 prediction error: {e}")
            import traceback
            traceback.print_exc()
            df_result['Prediction_Prob'] = 0.5

        # Remove helper column and restore original order
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to NetTCR-2.2 training data."""
        if self.nettcr22_path:
            training_file = self.nettcr22_path / "data" / "IMMREP" / "train" / "all_peptides_redundancy_reduced.csv"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load NetTCR-2.2 training data."""
        path = self.get_training_data_path()
        if path and path.exists():
            return pd.read_csv(path)
        return None
