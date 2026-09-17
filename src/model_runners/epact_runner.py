"""
EPACT Model Runner for TCRP Benchmark V4.

EPACT (Epitope-anchored Contrastive Transfer Learning) uses
attention mechanisms for TCR-peptide binding prediction.

V4 Features:
- Config-based conda path, timeout, GPU device, batch size
- Uses paths from config system

Native Input Format: CSV with VDJdb columns (CDR3.alpha.aa, CDR3.beta.aa, Epitope.peptide, MHC, etc.)
Native Output Format: predictions.csv with 'Pred' column

This runner:
1. Converts to VDJdb format
2. Runs EPACT in its proper conda environment (epact-benchmark)
3. Matches predictions back by position
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import subprocess
import logging
import tempfile
import os
import re

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)


class EPACTRunner(BaseModelRunner):
    """
    Runner for EPACT model.

    EPACT uses attention-based contrastive transfer learning with VDJdb format.

    Native interface:
    python scripts/predict/predict_tcr_pmhc_binding.py --config config.yml --input_data_path data.csv --model_location model_path --log_dir output_dir
    Output: predictions.csv with 'Pred' column
    """

    # Default MHC allele when not provided
    DEFAULT_MHC = "HLA-A*02:01"

    def __init__(self, mode: str = "exact_match"):
        """
        Initialize EPACT runner.

        Args:
            mode: Deduplication mode
        """
        super().__init__("EPACT", mode)
        self.epact_path = None
        self._epact_config = get_config()

    def setup(self) -> bool:
        """
        Set up EPACT model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            epact_dir = models_dir / "EPACT"

            if not epact_dir.exists():
                self.logger.error(f"EPACT model not found at {epact_dir}")
                return False

            self.epact_path = epact_dir

            # Check for predict script
            predict_script = epact_dir / "scripts" / "predict" / "predict_tcr_pmhc_binding.py"
            if not predict_script.exists():
                self.logger.error(f"predict_tcr_pmhc_binding.py not found at {predict_script}")
                return False

            self.logger.info(f"EPACT setup complete: {epact_dir}")
            return True

        except Exception as e:
            self.logger.error(f"EPACT setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns."""
        return ['Peptide', 'CDR3b']

    def _convert_hla_format(self, hla):
        """Convert various HLA formats to standard format (e.g., A0201 -> HLA-A*02:01)."""
        if pd.isna(hla) or hla == '' or hla == 'Unknown':
            return self.DEFAULT_MHC

        hla = str(hla).strip()

        # Already in proper format (HLA-A*02:01)
        if hla.startswith('HLA-') and '*' in hla and ':' in hla:
            return hla

        # Format: A0201, B0702, etc. -> HLA-A*02:01, HLA-B*07:02
        match = re.match(r'^([ABC])(\d{2})(\d{2})$', hla)
        if match:
            gene = match.group(1)
            group = match.group(2)
            protein = match.group(3)
            return f'HLA-{gene}*{group}:{protein}'

        # Format: A*02:01 -> HLA-A*02:01
        if '*' in hla and ':' in hla and not hla.startswith('HLA-'):
            return f'HLA-{hla}'

        # Can't parse, use default
        return self.DEFAULT_MHC

    def _convert_to_epact_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert input DataFrame to EPACT VDJdb format.

        EPACT expects: CDR3.alpha.aa, CDR3.beta.aa, Epitope.peptide, MHC, Target, etc.
        """
        epact_df = pd.DataFrame()

        # Map to VDJdb column names (EPACT expects original VDJdb format)
        epact_df['CDR3.alpha.aa'] = df.get('CDR3a', '')
        epact_df['CDR3.alpha.aa'] = epact_df['CDR3.alpha.aa'].fillna('')

        epact_df['CDR3.beta.aa'] = df['CDR3b']
        epact_df['Epitope.peptide'] = df['Peptide']

        # Handle HLA/MHC mapping
        hla_values = df.get('HLA', self.DEFAULT_MHC)
        if isinstance(hla_values, str):
            hla_values = pd.Series([hla_values] * len(df))
        epact_df['MHC'] = hla_values.apply(self._convert_hla_format)

        # Handle missing CDR columns
        epact_df['CDR1.alpha.aa'] = df.get('CDR1a', '').fillna('') if 'CDR1a' in df.columns else ''
        epact_df['CDR2.alpha.aa'] = df.get('CDR2a', '').fillna('') if 'CDR2a' in df.columns else ''
        epact_df['CDR1.beta.aa'] = df.get('CDR1b', '').fillna('') if 'CDR1b' in df.columns else ''
        epact_df['CDR2.beta.aa'] = df.get('CDR2b', '').fillna('') if 'CDR2b' in df.columns else ''

        epact_df['Target'] = df.get('Label', 1)

        return epact_df

    def _create_config_file(self, output_dir: str) -> str:
        """Create EPACT config file using the original config from EPACT repo."""
        import yaml

        # Use the original config file from EPACT if it exists
        original_config = self.epact_path / "configs" / "config-paired-cdr3-pmhc-binding.yml"
        if original_config.exists():
            config_file = os.path.join(output_dir, 'config.yml')
            # Load and modify the original config
            with open(original_config, 'r') as f:
                config = yaml.safe_load(f)
            # Set GPU device and batch size from config
            if 'training' in config:
                config['training']['gpu_device'] = self._epact_config.get_gpu_device()
                config['training']['test_batch_size'] = self._epact_config.get_model_batch_size('EPACT')
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            return config_file

        # Fallback: create config with ALL required parameters
        config = {
            'training': {
                'gpu_device': self._epact_config.get_gpu_device(),
                'max_epochs': 50,
                'log_dir': 'logs/paired-cdr3-pmhc-binding/',
                'lr': 2.5e-4,
                'weight_decay': 1.e-2,
                'warm_epochs': 5,
                'lr_scheduler': 'cosine',
                'lr_decay_steps': 50,
                'lr_decay_rate': 0.5,
                'lr_decay_min_lr': 1.e-6,
                'patience': 20,
                'non_binding_ratio': 5,
                'contrasruce_loss_type': 'simclr',
                'contrastive_loss_coef': 0.3,
                'pretrained_pmhc_model': 'checkpoints/pretrained/pmhc-BA-model-medium.pt',
                'pretrained_tcr_model': 'checkpoints/pretrained/paired-cdr3-model-medium.pt',
                'train_batch_size': 100,
                'test_batch_size': self._epact_config.get_model_batch_size('EPACT'),
                'num_workers': 4,
                'seed': 42,
                'temperature': 0.5,
                'margin': 0.4
            },
            'model': {
                'num_epi_layers': 6,
                'num_epi_heads': 4,
                'embed_epi_dim': 512,
                'num_mhc_layers': 6,
                'in_mhc_dim': 45,
                'embed_mhc_dim': 256,
                'mhc_seq_len': 366,
                'num_tcr_layers': 6,
                'num_tcr_heads': 4,
                'embed_tcr_dim': 512,
                'cross_attn_heads': 4,
                'embed_hid_dim': 512,
                'num_conv_layers': 2,
                'attn_dropout': 0.05,
                'dropout': 0.3,
                'projector_type': 'mlp',
                'agg': 'cls'
            },
            'data': {
                'use_cdr123': False,
                'train_pmhc_path': 'data/binding/Paired-TCR/pMHC-train-data.tsv',
                'train_tcr_feat_path': 'data/binding/Paired-TCR/train_paired_cdr3_seq.pt',
                'train_pos_data_path': 'data/binding/Paired-TCR/Paired-TCR-pMHC-Binding-train-data.csv',
                'test_data_path': 'data/binding/Paired-TCR/test_paired_tcr_pmhc_VDJdb_data.csv',
                'hla_lib_path': 'data/hla_library.json',
                'kfold_data_path': 'data/binding/Paired-TCR/k-fold-data'
            }
        }

        config_file = os.path.join(output_dir, 'config.yml')
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_file

    def _run_epact_prediction(self, input_file: str, output_dir: str) -> Optional[np.ndarray]:
        """
        Run EPACT prediction using conda environment.

        Returns:
            np.ndarray of predictions or None if failed
        """
        conda_exe = self.paths.get_conda_executable()

        # Create config file
        config_file = self._create_config_file(output_dir)
        model_path = self.epact_path / "paired-cdr3-pmhc-binding" / "paired-cdr3-pmhc-binding-model-fold-1.pt"

        # EPACT requires running from its directory
        script_content = f"""
import os
import sys
os.chdir('{self.epact_path}')
sys.path.insert(0, '.')

# Set environment variables for MKL
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import subprocess
cmd = [
    sys.executable, 'scripts/predict/predict_tcr_pmhc_binding.py',
    '--config', '{config_file}',
    '--input_data_path', '{input_file}',
    '--model_location', '{model_path}',
    '--log_dir', '{output_dir}'
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"EPACT error: {{result.stderr}}", file=sys.stderr)
    sys.exit(1)
"""
        temp_script = tempfile.mktemp(suffix='.py')
        with open(temp_script, 'w') as f:
            f.write(script_content)

        env_name = self._epact_config.get_model_env('EPACT')
        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", temp_script
        ]

        self.logger.info(f"Running EPACT prediction...")

        try:
            timeout = self._epact_config.get_model_timeout('EPACT')
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
                self.logger.error(f"EPACT failed: {result.stderr}")
                return None

            # Load predictions from output directory
            pred_file = os.path.join(output_dir, 'predictions.csv')
            if not os.path.exists(pred_file):
                self.logger.error(f"Predictions file not found: {pred_file}")
                return None

            pred_df = pd.read_csv(pred_file)
            self.logger.info(f"EPACT prediction completed: {len(pred_df)} predictions")

            if 'Pred' in pred_df.columns:
                return pred_df['Pred'].values
            elif 'prediction' in pred_df.columns:
                return pred_df['prediction'].values
            else:
                self.logger.error(f"No 'Pred' column. Columns: {list(pred_df.columns)}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error("EPACT prediction timed out")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return None
        except Exception as e:
            self.logger.error(f"EPACT error: {e}")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return None

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using EPACT.

        Args:
            df: Input DataFrame with ID, Peptide, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))

        try:
            # Convert to EPACT format
            epact_df = self._convert_to_epact_format(df)

            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save input file
                input_file = os.path.join(temp_dir, 'input.csv')
                epact_df.to_csv(input_file, index=False)

                # Run prediction
                predictions = self._run_epact_prediction(input_file, temp_dir)

                if predictions is not None and len(predictions) == len(df):
                    df_result['Prediction_Prob'] = predictions
                    self.logger.info("Matched predictions by position")
                elif predictions is not None:
                    self.logger.warning(f"Prediction count mismatch: {len(predictions)} vs {len(df)}")
                    df_result['Prediction_Prob'] = 0.5
                    for i in range(min(len(predictions), len(df))):
                        df_result.loc[df_result['_orig_idx'] == i, 'Prediction_Prob'] = predictions[i]
                else:
                    self.logger.error("EPACT prediction failed")
                    df_result['Prediction_Prob'] = 0.5

        except Exception as e:
            self.logger.error(f"EPACT prediction error: {e}")
            import traceback
            traceback.print_exc()
            df_result['Prediction_Prob'] = 0.5

        # Restore order and cleanup
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to EPACT training data."""
        if self.epact_path:
            training_file = self.epact_path / "data" / "vdjdb_train.csv"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load EPACT training data."""
        path = self.get_training_data_path()
        if path and path.exists():
            return pd.read_csv(path)
        return None
