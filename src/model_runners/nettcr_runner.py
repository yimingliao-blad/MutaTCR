"""
NetTCR (NetTCR-2.0) Model Runner for TCRP Benchmark V3.

NetTCR uses deep learning for TCR-peptide binding prediction.
Supports beta-only or alpha+beta chain configurations.

Native Input Format: CSV with columns CDR3a, CDR3b, peptide, binder
Native Output Format: Input CSV with 'prediction' column added

This runner:
1. Preserves ID column for 1-to-1 merge
2. Runs NetTCR in its proper conda environment (nettcr-benchmark)
3. Matches predictions back by row index

Constraints:
- Max peptide length: 9 amino acids
- Max CDR3 length: 30 amino acids
- Valid amino acids: ACDEFGHIKLMNPQRSTVWY
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
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

# Valid amino acids for NetTCR
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


class NetTCRRunner(BaseModelRunner):
    """
    Runner for NetTCR (NetTCR-2.0) model.

    NetTCR supports beta-only or alpha+beta chain modes.

    Native interface:
    python nettcr.py --trainfile data.csv --testfile test.csv --chain b --outfile out.csv
    """

    def __init__(self, mode: str = "exact_match", chain: str = "b"):
        """
        Initialize NetTCR runner.

        Args:
            mode: Deduplication mode
            chain: 'b' (beta only) or 'ab' (alpha+beta)
        """
        super().__init__("NetTCR", mode)
        self.chain = chain
        self.nettcr_path = None
        self._nettcr_config = get_config()

    def setup(self) -> bool:
        """
        Set up NetTCR model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            # Try different possible names
            for name in ["NETTCR2", "NetTCR2", "NetTCR"]:
                nettcr_dir = models_dir / name
                if nettcr_dir.exists():
                    break
            else:
                self.logger.error(f"NetTCR model not found in {models_dir}")
                return False

            self.nettcr_path = nettcr_dir

            # Check for nettcr.py
            nettcr_script = nettcr_dir / "nettcr.py"
            if not nettcr_script.exists():
                self.logger.error(f"nettcr.py not found at {nettcr_script}")
                return False

            # Check for training file
            train_file = nettcr_dir / "data" / "train_ab_95_alphabeta.csv"
            if not train_file.exists():
                self.logger.warning(f"Training file not found at {train_file}")

            self.logger.info(f"NetTCR setup complete: {nettcr_dir}")
            return True

        except Exception as e:
            self.logger.error(f"NetTCR setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns based on chain config."""
        if self.chain == 'ab':
            return ['Peptide', 'CDR3a', 'CDR3b']
        return ['Peptide', 'CDR3b']

    def _validate_sequence(self, seq: str, max_len: int) -> bool:
        """Validate sequence has valid amino acids and length."""
        if pd.isna(seq) or not isinstance(seq, str):
            return False
        if len(seq) > max_len:
            return False
        return all(aa in VALID_AA for aa in seq.upper())

    def _convert_to_nettcr_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert input DataFrame to NetTCR format.

        NetTCR expects: CDR3a, CDR3b, peptide, binder
        """
        nettcr_df = pd.DataFrame()
        nettcr_df['peptide'] = df['Peptide'].str.upper()
        nettcr_df['CDR3b'] = df['CDR3b'].str.upper()

        if self.chain == 'ab' and 'CDR3a' in df.columns:
            nettcr_df['CDR3a'] = df['CDR3a'].fillna('').str.upper()
        else:
            nettcr_df['CDR3a'] = ''

        nettcr_df['binder'] = df.get('Label', 1)
        nettcr_df['_orig_idx'] = range(len(df))

        return nettcr_df

    def _run_nettcr_prediction(self, input_file: str, output_file: str) -> bool:
        """
        Run NetTCR prediction using conda environment.

        Args:
            input_file: Path to input CSV
            output_file: Path to output CSV

        Returns:
            True if prediction successful
        """
        conda_exe = self.paths.get_conda_executable()
        env_name = self._nettcr_config.get_model_env('NetTCR')
        train_file = self.nettcr_path / "data" / "train_ab_95_alphabeta.csv"

        # NetTCR requires running from its directory
        script_content = f"""
import os
import sys
os.chdir('{self.nettcr_path}')
sys.path.insert(0, '.')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Run nettcr via subprocess to handle TF properly
import subprocess
cmd = [
    sys.executable, 'nettcr.py',
    '--trainfile', '{train_file}',
    '--testfile', '{input_file}',
    '--chain', '{self.chain}',
    '--outfile', '{output_file}'
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"NetTCR error: {{result.stderr}}", file=sys.stderr)
    sys.exit(1)
"""
        temp_script = tempfile.mktemp(suffix='.py')
        with open(temp_script, 'w') as f:
            f.write(script_content)

        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", temp_script
        ]

        self.logger.info(f"Running NetTCR: chain={self.chain}...")

        try:
            timeout = self._nettcr_config.get_model_timeout('NetTCR')
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
                self.logger.error(f"NetTCR failed: {result.stderr}")
                return False

            self.logger.info("NetTCR prediction completed")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("NetTCR prediction timed out")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return False
        except Exception as e:
            self.logger.error(f"NetTCR error: {e}")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using NetTCR.

        Args:
            df: Input DataFrame with ID, Peptide, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))

        try:
            # Filter valid sequences (peptide max 9 AA, CDR3 max 30 AA)
            valid_mask = (
                df['Peptide'].apply(lambda x: self._validate_sequence(x, 9)) &
                df['CDR3b'].apply(lambda x: self._validate_sequence(x, 30))
            )
            if self.chain == 'ab' and 'CDR3a' in df.columns:
                valid_mask &= df['CDR3a'].apply(lambda x: pd.isna(x) or self._validate_sequence(x, 30))

            valid_df = df[valid_mask].copy()
            invalid_count = len(df) - len(valid_df)

            if invalid_count > 0:
                self.logger.warning(f"Filtered {invalid_count} invalid sequences")

            if len(valid_df) == 0:
                self.logger.warning("No valid sequences for NetTCR")
                df_result['Prediction_Prob'] = 0.5
                df_result = df_result.drop(columns=['_orig_idx'])
                return df_result

            # Convert to NetTCR format
            nettcr_df = self._convert_to_nettcr_format(valid_df)
            valid_orig_idx = nettcr_df['_orig_idx'].values

            # Create temp files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_input = f.name
                nettcr_df.drop(columns=['_orig_idx']).to_csv(f, index=False)

            temp_output = tempfile.mktemp(suffix='.csv')

            # Run prediction
            success = self._run_nettcr_prediction(temp_input, temp_output)

            # Initialize predictions
            df_result['Prediction_Prob'] = 0.5  # Default for invalid sequences

            if success and os.path.exists(temp_output):
                pred_df = pd.read_csv(temp_output)
                self.logger.info(f"Loaded {len(pred_df)} predictions")

                if 'prediction' in pred_df.columns:
                    # Match by position (NetTCR preserves order)
                    if len(pred_df) == len(valid_df):
                        for i, orig_idx in enumerate(valid_orig_idx):
                            df_result.loc[df_result['_orig_idx'] == orig_idx, 'Prediction_Prob'] = pred_df['prediction'].iloc[i]
                        self.logger.info("Matched predictions by position")
                    else:
                        self.logger.warning(f"Row count mismatch: {len(pred_df)} vs {len(valid_df)}")
                        # Try matching by CDR3b + peptide
                        pred_df['_key'] = pred_df['CDR3b'] + '_' + pred_df['peptide']
                        pred_dict = dict(zip(pred_df['_key'], pred_df['prediction']))
                        for orig_idx, row in zip(valid_orig_idx, valid_df.itertuples()):
                            key = row.CDR3b.upper() + '_' + row.Peptide.upper()
                            if key in pred_dict:
                                df_result.loc[df_result['_orig_idx'] == orig_idx, 'Prediction_Prob'] = pred_dict[key]
                else:
                    self.logger.error(f"No 'prediction' column. Columns: {list(pred_df.columns)}")

                # Cleanup
                try:
                    os.unlink(temp_output)
                except:
                    pass
            else:
                self.logger.error("NetTCR prediction failed or no output")

            # Cleanup input
            if os.path.exists(temp_input):
                os.unlink(temp_input)

        except Exception as e:
            self.logger.error(f"NetTCR prediction error: {e}")
            import traceback
            traceback.print_exc()
            df_result['Prediction_Prob'] = 0.5

        # Restore order and cleanup
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to NetTCR training data."""
        if self.nettcr_path:
            training_file = self.nettcr_path / "data" / "train_ab_95_alphabeta.csv"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load NetTCR training data."""
        path = self.get_training_data_path()
        if path and path.exists():
            return pd.read_csv(path)
        return None
