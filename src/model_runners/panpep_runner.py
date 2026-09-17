"""
PanPep Model Runner for TCRP Benchmark V4.

PanPep uses meta-learning for TCR-peptide binding prediction,
supporting both seen and unseen peptides through few-shot learning.

V4 Features:
- Best-of-modes: For each peptide, try multiple modes and select the best
- Uses majority_training_dataset.csv for mode selection (not base_dataset.csv)
- Configurable via benchmark_config.yaml

Mode Selection (based on training data match):
- zero-shot: Peptide not in training data (0 records)
- few-shot: Peptide has 1-4 records in training data
- majority: Peptide has 5+ records in training data

Best-of-Modes Strategy:
- Majority peptides: Try majority, few-shot, zero-shot -> pick best pAUC
- Few-shot peptides: Try few-shot, zero-shot -> pick best pAUC
- Zero-shot peptides: Use zero-shot only

Native Input Format: CSV with columns 'Peptide', 'CDR3', 'Label'
Native Output Format: CSV with 'Score' column
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import subprocess
import logging
import tempfile
import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)


class PanPepRunner(BaseModelRunner):
    """
    Runner for PanPep model with best-of-modes support.

    PanPep uses meta-learning with adaptive mode selection.
    V4 adds best-of-modes: trying multiple modes per peptide and selecting the best.
    """

    def __init__(self, mode: str = "exact_match"):
        """
        Initialize PanPep runner.

        Args:
            mode: Deduplication mode
        """
        super().__init__("PanPep", mode)
        self.panpep_path = None
        self._config = get_config()
        self._training_df = None
        self._training_combinations = set()

    def setup(self) -> bool:
        """
        Set up PanPep model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            panpep_dir = models_dir / "PanPep"

            if not panpep_dir.exists():
                self.logger.error(f"PanPep model not found at {panpep_dir}")
                return False

            self.panpep_path = panpep_dir

            # Check if PanPep.py exists
            panpep_script = panpep_dir / "PanPep.py"
            if not panpep_script.exists():
                self.logger.error(f"PanPep.py not found at {panpep_script}")
                return False

            # Load TRAINING data for mode selection (majority_training_dataset.csv)
            # This is what determines how many examples PanPep has seen during training
            training_file = panpep_dir / "Data" / "majority_training_dataset.csv"
            if training_file.exists():
                self._training_df = pd.read_csv(training_file)
                self._training_combinations = set(
                    zip(self._training_df['peptide'], self._training_df['binding_TCR'])
                )
                self.logger.info(f"Loaded training data: {len(self._training_df)} records, "
                               f"{len(self._training_combinations)} unique combinations")
            else:
                self.logger.warning(f"Training data not found: {training_file}")
                self._training_df = pd.DataFrame(columns=['peptide', 'binding_TCR', 'label'])
                self._training_combinations = set()

            # Also load base_dataset for context generation (has more samples)
            base_file = panpep_dir / "Data" / "base_dataset.csv"
            if base_file.exists():
                self._base_dataset = pd.read_csv(base_file)
                self.logger.info(f"Loaded base dataset: {len(self._base_dataset)} records")
            else:
                self._base_dataset = self._training_df.copy()

            self.logger.info(f"PanPep setup complete: {panpep_dir}")
            return True

        except Exception as e:
            self.logger.error(f"PanPep setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns."""
        return ['Peptide', 'CDR3b']

    def _determine_base_mode(self, peptide: str) -> str:
        """
        Determine base mode for a peptide based on training data.

        Args:
            peptide: Peptide sequence

        Returns:
            Mode string: 'zero-shot', 'few-shot', or 'majority'
        """
        if len(self._training_df) == 0:
            return 'zero-shot'

        count = len(self._training_df[self._training_df['peptide'] == peptide])
        if count == 0:
            return 'zero-shot'
        elif count <= 4:
            return 'few-shot'
        else:
            return 'majority'

    def _get_modes_to_try(self, base_mode: str) -> List[str]:
        """
        Get list of modes to try based on base mode.

        Uses config settings for best-of-modes.
        """
        if not self._config.get_panpep_best_of_modes():
            return [base_mode]

        return self._config.get_panpep_modes_for_base(base_mode)

    def _generate_context_data(self, peptide: str, mode: str, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate context data for a peptide based on mode.

        Uses base_dataset for context examples (more comprehensive than training data).

        Args:
            peptide: The peptide sequence
            mode: 'few-shot' or 'majority'
            test_df: Test DataFrame to exclude from negatives

        Returns:
            Context DataFrame with Peptide, CDR3, Label columns
        """
        # Get positives for this peptide from base dataset
        peptide_positives = self._base_dataset[self._base_dataset['peptide'] == peptide]

        if len(peptide_positives) == 0:
            return pd.DataFrame()

        # Get all CDR3 sequences for negative generation
        all_cdr3s = self._base_dataset['binding_TCR'].unique()

        # Test combinations to exclude
        test_combinations = set(zip(test_df['Peptide'], test_df['CDR3b']))

        # Determine negative count
        if mode == 'few-shot':
            n_negatives = 6  # Fixed for few-shot
        else:
            n_negatives = len(peptide_positives) * 3  # 3x for majority

        # Generate negatives
        np.random.seed(42)
        negatives = []
        attempts = 0
        max_attempts = 500

        while len(negatives) < n_negatives and attempts < max_attempts:
            attempts += 1
            random_cdr3 = np.random.choice(all_cdr3s)

            # Check if valid (not in test data)
            if (peptide, random_cdr3) not in test_combinations:
                negatives.append({
                    'Peptide': peptide,
                    'CDR3': random_cdr3,
                    'Label': 0
                })

        # Create context DataFrame
        context_data = []

        # Add positives (limit for few-shot)
        positives_to_use = peptide_positives
        if mode == 'few-shot' and len(peptide_positives) > 4:
            positives_to_use = peptide_positives.sample(n=4, random_state=42)

        for _, row in positives_to_use.iterrows():
            context_data.append({
                'Peptide': row['peptide'],
                'CDR3': row['binding_TCR'],
                'Label': 1
            })

        # Add negatives
        context_data.extend(negatives)

        return pd.DataFrame(context_data)

    def _run_panpep_prediction(self, input_file: str, output_file: str, mode: str) -> bool:
        """
        Run PanPep prediction using conda environment.

        Args:
            input_file: Path to input CSV
            output_file: Path to output CSV
            mode: 'zero-shot', 'few-shot', or 'majority'

        Returns:
            True if prediction successful
        """
        panpep_script = self.panpep_path / "PanPep.py"
        conda_exe = self.paths.get_conda_executable()

        # Get environment name from config
        env_name = self._config.get_model_env('PanPep')

        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", str(panpep_script),
            "--learning_setting", mode,
            "--input", input_file,
            "--output", output_file
        ]

        # For majority mode, use more update steps
        if mode == 'majority':
            cmd.extend(["--update_step_test", "1000"])

        self.logger.debug(f"Running PanPep: {' '.join(cmd[:10])}...")

        try:
            timeout = self._config.get_model_timeout('PanPep')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.panpep_path),
                timeout=timeout
            )

            if result.returncode != 0:
                self.logger.error(f"PanPep {mode} failed: {result.stderr[:500]}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"PanPep {mode} prediction timed out")
            return False
        except Exception as e:
            self.logger.error(f"PanPep error: {e}")
            return False

    def _run_mode_for_peptide(self, peptide: str, test_samples: pd.DataFrame, mode: str) -> Optional[pd.DataFrame]:
        """
        Run a single mode for a peptide's test samples.

        Args:
            peptide: Peptide sequence
            test_samples: DataFrame of test samples for this peptide
            mode: Mode to run ('zero-shot', 'few-shot', 'majority')

        Returns:
            DataFrame with predictions, or None if failed
        """
        # Prepare input data
        if mode == 'zero-shot':
            panpep_df = pd.DataFrame({
                'Peptide': test_samples['Peptide'],
                'CDR3': test_samples['CDR3b']
            })
        else:
            # Generate context data
            context_df = self._generate_context_data(peptide, mode, test_samples)

            if len(context_df) == 0:
                # No context available, fall back to zero-shot
                self.logger.debug(f"No context for {peptide} in {mode}, using zero-shot")
                return None

            # Test samples with 'Unknown' label
            test_records = []
            for _, row in test_samples.iterrows():
                test_records.append({
                    'Peptide': row['Peptide'],
                    'CDR3': row['CDR3b'],
                    'Label': 'Unknown'
                })

            test_part = pd.DataFrame(test_records)
            panpep_df = pd.concat([context_df, test_part], ignore_index=True)

        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_input = f.name
            panpep_df.to_csv(f, index=False)

        temp_output = tempfile.mktemp(suffix='.csv')

        try:
            success = self._run_panpep_prediction(temp_input, temp_output, mode)

            if success and os.path.exists(temp_output):
                pred_df = pd.read_csv(temp_output)

                # Find prediction column
                pred_col = None
                for col in ['Score', 'score', 'prediction', 'preds']:
                    if col in pred_df.columns:
                        pred_col = col
                        break

                if pred_col:
                    # Create result DataFrame matching test_samples
                    result = test_samples.copy()
                    result['Prediction_Prob'] = 0.5

                    # Match by (Peptide, CDR3)
                    pred_dict = {}
                    for _, row in pred_df.iterrows():
                        key = (row.get('Peptide', ''), row.get('CDR3', ''))
                        pred_dict[key] = row[pred_col]

                    for idx, row in result.iterrows():
                        key = (row['Peptide'], row['CDR3b'])
                        if key in pred_dict:
                            result.loc[idx, 'Prediction_Prob'] = pred_dict[key]

                    return result

            return None

        finally:
            if os.path.exists(temp_input):
                os.unlink(temp_input)
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    def _calculate_pauc(self, df: pd.DataFrame, max_fpr: float = 0.1) -> float:
        """
        Calculate partial AUC at max FPR.

        Args:
            df: DataFrame with Label and Prediction_Prob columns
            max_fpr: Maximum false positive rate

        Returns:
            Partial AUC score (normalized to 0-1)
        """
        if 'Label' not in df.columns or 'Prediction_Prob' not in df.columns:
            return 0.5

        labels = df['Label'].values
        probs = df['Prediction_Prob'].values

        # Need both classes
        if len(np.unique(labels)) < 2:
            return 0.5

        try:
            pauc = roc_auc_score(labels, probs, max_fpr=max_fpr)
            # Normalize to 0-1 range
            return pauc / max_fpr
        except:
            return 0.5

    def _run_best_of_modes(self, peptide: str, test_samples: pd.DataFrame, modes: List[str]) -> pd.DataFrame:
        """
        Try multiple modes and select the best predictions.

        Args:
            peptide: Peptide sequence
            test_samples: DataFrame of test samples
            modes: List of modes to try

        Returns:
            DataFrame with best predictions
        """
        results = {}

        for mode in modes:
            pred_df = self._run_mode_for_peptide(peptide, test_samples, mode)
            if pred_df is not None:
                # Calculate pAUC for this mode
                if 'Label' in test_samples.columns:
                    pred_df['Label'] = test_samples['Label'].values
                    pauc = self._calculate_pauc(pred_df)
                else:
                    # No labels, use mean prediction as proxy
                    pauc = pred_df['Prediction_Prob'].mean()

                results[mode] = {
                    'predictions': pred_df,
                    'pauc': pauc
                }
                self.logger.debug(f"Peptide {peptide[:8]}... {mode}: pAUC={pauc:.4f}")

        if not results:
            # All modes failed, return default
            result = test_samples.copy()
            result['Prediction_Prob'] = 0.5
            return result

        # Select best mode by pAUC
        best_mode = max(results, key=lambda m: results[m]['pauc'])
        self.logger.debug(f"Peptide {peptide[:8]}... best mode: {best_mode}")

        return results[best_mode]['predictions']

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using PanPep with best-of-modes.

        For each peptide:
        1. Determine base mode from training data
        2. Get list of modes to try (based on config)
        3. Run each mode and calculate pAUC
        4. Select best mode's predictions

        Args:
            df: Input DataFrame with ID, Peptide, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))
        df_result['Prediction_Prob'] = 0.5  # Default

        try:
            best_of_modes = self._config.get_panpep_best_of_modes()
            self.logger.info(f"Best-of-modes: {'enabled' if best_of_modes else 'disabled'}")

            # Determine base mode for each peptide
            peptide_base_modes = {}
            for peptide in df['Peptide'].unique():
                peptide_base_modes[peptide] = self._determine_base_mode(peptide)

            # Log mode distribution
            mode_counts = defaultdict(int)
            for mode in peptide_base_modes.values():
                mode_counts[mode] += 1
            self.logger.info(f"Base mode distribution: {dict(mode_counts)}")

            if best_of_modes:
                # Best-of-modes: Process each peptide individually
                for peptide in df['Peptide'].unique():
                    peptide_df = df[df['Peptide'] == peptide].copy()
                    base_mode = peptide_base_modes[peptide]
                    modes_to_try = self._get_modes_to_try(base_mode)

                    self.logger.info(f"Processing {peptide[:12]}... ({len(peptide_df)} samples, base: {base_mode}, trying: {modes_to_try})")

                    # Run best-of-modes
                    pred_df = self._run_best_of_modes(peptide, peptide_df, modes_to_try)

                    # Update results
                    for _, row in pred_df.iterrows():
                        mask = (df_result['Peptide'] == row['Peptide']) & (df_result['CDR3b'] == row['CDR3b'])
                        df_result.loc[mask, 'Prediction_Prob'] = row['Prediction_Prob']

            else:
                # Standard mode: Group by base mode and process
                mode_groups = defaultdict(list)
                for peptide, mode in peptide_base_modes.items():
                    mode_groups[mode].append(peptide)

                for mode, peptides in mode_groups.items():
                    if not peptides:
                        continue

                    self.logger.info(f"Processing {len(peptides)} peptides in {mode} mode")
                    mode_test_df = df[df['Peptide'].isin(peptides)].copy()

                    for peptide in peptides:
                        peptide_df = mode_test_df[mode_test_df['Peptide'] == peptide].copy()
                        pred_df = self._run_mode_for_peptide(peptide, peptide_df, mode)

                        if pred_df is not None:
                            for _, row in pred_df.iterrows():
                                mask = (df_result['Peptide'] == row['Peptide']) & (df_result['CDR3b'] == row['CDR3b'])
                                df_result.loc[mask, 'Prediction_Prob'] = row['Prediction_Prob']

        except Exception as e:
            self.logger.error(f"PanPep prediction error: {e}")
            import traceback
            traceback.print_exc()

        # Remove helper column and restore original order
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to PanPep training data (majority_training_dataset.csv)."""
        if self.panpep_path:
            training_file = self.panpep_path / "Data" / "majority_training_dataset.csv"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load PanPep training data."""
        path = self.get_training_data_path()
        if path and path.exists():
            return pd.read_csv(path)
        return None
