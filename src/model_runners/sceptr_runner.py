"""
SCEPTR Model Runner for TCRP Benchmark V3.

SCEPTR (Single-cell TCR Embedding Prediction of T-cell Receptor binding)
uses pre-trained TCR embeddings with a classifier (KNN or Random Forest).

IMPORTANT: SCEPTR is NOT a pre-trained classifier. It:
1. Generates TCR embeddings using its transformer model
2. Encodes peptide features (AA composition + length)
3. Trains a classifier (KNN or RF) on training data embeddings
4. Uses the trained classifier for prediction

Native Input Format: DataFrame with TRAV, CDR3A, TRAJ, TRBV, CDR3B, TRBJ columns
Native Output Format: TCR embeddings (N, 64) numpy array

This runner:
1. Loads VDJDB training data (6 peptides × 300 samples)
2. Generates training negatives (1:5 ratio)
3. Trains classifier on SCEPTR embeddings + peptide features
4. Runs prediction in sceptr_env conda environment
5. Matches predictions back by position

Based on V1 benchmark_sceptr.py implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import subprocess
import logging
import tempfile
import os
import json
import pickle

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)

# Training configuration (matches V1 benchmark)
TRAINING_PEPTIDES = ['KLGGALQAK', 'GILGFVFTL', 'NLVPMVATV', 'AVFDRKSDAK', 'RAKFKQLL', 'ELAGIGILTV']
SAMPLES_PER_PEPTIDE = 300
NEGATIVE_RATIO = 5

# Valid TRBV genes for SCEPTR (matches V1)
VALID_TRBV_GENES = [
    'TRBV19*01', 'TRBV20-1*01', 'TRBV7-9*01', 'TRBV27*01', 'TRBV9*01',
    'TRBV29-1*01', 'TRBV2*01', 'TRBV4-1*01', 'TRBV12-3*01', 'TRBV6-5*01',
    'TRBV11-2*01', 'TRBV28*01', 'TRBV14*01', 'TRBV7-2*01'
]


class SCEPTRRunner(BaseModelRunner):
    """
    Runner for SCEPTR model.

    SCEPTR uses pre-trained TCR embeddings + trained classifier (KNN/RF).
    Training: VDJDB top 6 peptides × 300 samples + 5x negatives

    Native interface:
    - sceptr.calc_vector_representations(df) -> (N, 64) embeddings
    - Classifier trained on embeddings + peptide features
    """

    CLASSIFIER_TYPES = ['knn', 'random_forest']

    def __init__(self, mode: str = "exact_match", classifier_type: str = "knn"):
        """
        Initialize SCEPTR runner.

        Args:
            mode: Deduplication mode
            classifier_type: 'knn' or 'random_forest'
        """
        super().__init__("SCEPTR", mode)
        self.classifier_type = classifier_type
        self.sceptr_path = None
        self._sceptr_config = get_config()
        self._classifier = None
        self._training_df = None

    def setup(self) -> bool:
        """
        Set up SCEPTR model.

        Returns:
            True if setup successful
        """
        try:
            models_dir = self.paths.MODELS_DIR
            sceptr_dir = models_dir / "SCEPTR"

            if not sceptr_dir.exists():
                self.logger.error(f"SCEPTR model not found at {sceptr_dir}")
                return False

            self.sceptr_path = sceptr_dir

            # Check for SCEPTR model files
            model_dir = sceptr_dir / "src" / "sceptr" / "_model_saves" / "SCEPTR"
            if not model_dir.exists():
                self.logger.error(f"SCEPTR model saves not found at {model_dir}")
                return False

            # Check for VDJDB training data
            vdjdb_path = models_dir / "ERGO" / "data" / "VDJDB_complete.tsv"
            if not vdjdb_path.exists():
                self.logger.error(f"VDJDB training data not found at {vdjdb_path}")
                return False

            self.vdjdb_path = vdjdb_path
            self.logger.info(f"SCEPTR setup complete: {sceptr_dir}")
            return True

        except Exception as e:
            self.logger.error(f"SCEPTR setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        """Return required input columns."""
        return ['Peptide', 'CDR3b']

    def _load_training_data(self) -> Optional[pd.DataFrame]:
        """Load training data from VDJDB for specified peptides."""
        try:
            df = pd.read_csv(self.vdjdb_path, sep='\t')

            training_data = []
            for peptide in TRAINING_PEPTIDES:
                # Select records with this peptide
                peptide_records = df[df['Epitope'] == peptide].copy()

                if len(peptide_records) == 0:
                    continue

                # Filter to only valid TRBV genes
                peptide_records = peptide_records[peptide_records['V'].isin(VALID_TRBV_GENES)]

                if len(peptide_records) == 0:
                    continue

                # Sample records
                if len(peptide_records) > SAMPLES_PER_PEPTIDE:
                    peptide_records = peptide_records.sample(n=SAMPLES_PER_PEPTIDE, random_state=42)

                # Convert to standard format
                peptide_data = pd.DataFrame({
                    'Peptide': [peptide] * len(peptide_records),
                    'CDR3b': peptide_records['CDR3'],
                    'Vb': peptide_records['V'],
                    'Jb': peptide_records['J'],
                    'Label': [1] * len(peptide_records)
                })

                training_data.append(peptide_data)

            if not training_data:
                return None

            return pd.concat(training_data, ignore_index=True)

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return None

    def _generate_training_negatives(self, positive_df: pd.DataFrame) -> pd.DataFrame:
        """Generate negatives for training data by peptide swapping."""
        np.random.seed(42)

        all_tcrs = positive_df['CDR3b'].unique()
        all_peptides = positive_df['Peptide'].unique()
        existing_combinations = set(zip(positive_df['Peptide'], positive_df['CDR3b']))

        negatives = []
        target_count = len(positive_df) * NEGATIVE_RATIO
        attempts = 0
        max_attempts = target_count * 10

        while len(negatives) < target_count and attempts < max_attempts:
            attempts += 1

            random_peptide = np.random.choice(all_peptides)
            random_tcr = np.random.choice(all_tcrs)

            if (random_peptide, random_tcr) not in existing_combinations:
                tcr_info = positive_df[positive_df['CDR3b'] == random_tcr]
                if len(tcr_info) > 0:
                    v_gene = tcr_info['Vb'].mode().iloc[0]
                    j_gene = tcr_info['Jb'].mode().iloc[0]

                    negatives.append({
                        'Peptide': random_peptide,
                        'CDR3b': random_tcr,
                        'Vb': v_gene,
                        'Jb': j_gene,
                        'Label': 0
                    })
                    existing_combinations.add((random_peptide, random_tcr))

        return pd.DataFrame(negatives)

    def _convert_to_sceptr_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame to SCEPTR format.

        SCEPTR expects: TRAV, CDR3A, TRAJ, TRBV, CDR3B, TRBJ
        """
        sceptr_data = pd.DataFrame({
            'TRAV': df.get('Va', 'TRAV12-2*01').fillna('TRAV12-2*01') if 'Va' in df.columns else ['TRAV12-2*01'] * len(df),
            'CDR3A': df.get('CDR3a', 'CAVKASGSRLTF').fillna('CAVKASGSRLTF') if 'CDR3a' in df.columns else ['CAVKASGSRLTF'] * len(df),
            'TRAJ': df.get('Ja', 'TRAJ1*01').fillna('TRAJ1*01') if 'Ja' in df.columns else ['TRAJ1*01'] * len(df),
            'TRBV': df['Vb'].fillna('TRBV19*01') if 'Vb' in df.columns else ['TRBV19*01'] * len(df),
            'CDR3B': df['CDR3b'],
            'TRBJ': df['Jb'].fillna('TRBJ1-1*01') if 'Jb' in df.columns else ['TRBJ1-1*01'] * len(df)
        }, index=df.index)

        # Filter to valid TRBV genes
        valid_mask = sceptr_data['TRBV'].isin(VALID_TRBV_GENES)
        return sceptr_data[valid_mask]

    def _encode_peptide_features(self, peptides: List[str]) -> np.ndarray:
        """Encode peptide features (AA composition + length)."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        # Amino acid composition (20 features)
        composition = np.zeros((len(peptides), len(amino_acids)))
        for i, peptide in enumerate(peptides):
            for j, aa in enumerate(amino_acids):
                composition[i, j] = peptide.count(aa) / len(peptide) if len(peptide) > 0 else 0

        # Peptide length (1 feature)
        lengths = np.array([[len(p)] for p in peptides])

        return np.hstack([composition, lengths])

    def _run_sceptr_embeddings(self, sceptr_df: pd.DataFrame, output_file: str) -> bool:
        """
        Run SCEPTR embedding generation in conda environment.

        Returns:
            True if successful
        """
        conda_exe = self.paths.get_conda_executable()
        env_name = self._sceptr_config.get_model_env('SCEPTR')

        # Save input data
        input_file = output_file.replace('.npy', '_input.csv')
        sceptr_df.to_csv(input_file, index=False)

        # Script to run in SCEPTR environment
        script_content = f"""
import os
import sys
import numpy as np
import pandas as pd

# Add SCEPTR to path
sceptr_src = '{self.sceptr_path}/src'
sys.path.insert(0, sceptr_src)

import sceptr

# Load input data
df = pd.read_csv('{input_file}')

# Generate embeddings
embeddings = sceptr.calc_vector_representations(df)

# Save embeddings
np.save('{output_file}', embeddings)
print(f"Saved {{len(embeddings)}} embeddings to {output_file}")
"""
        temp_script = tempfile.mktemp(suffix='.py')
        with open(temp_script, 'w') as f:
            f.write(script_content)

        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", temp_script
        ]

        try:
            timeout = self._sceptr_config.get_model_timeout('SCEPTR')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Cleanup
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            if os.path.exists(input_file):
                os.unlink(input_file)

            if result.returncode != 0:
                self.logger.error(f"SCEPTR embedding failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("SCEPTR embedding timed out")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return False
        except Exception as e:
            self.logger.error(f"SCEPTR error: {e}")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return False

    def _train_classifier(self, training_df: pd.DataFrame) -> bool:
        """
        Train classifier on SCEPTR embeddings + peptide features.

        Returns:
            True if successful
        """
        self.logger.info("Training SCEPTR classifier...")

        try:
            # Convert to SCEPTR format
            sceptr_data = self._convert_to_sceptr_format(training_df)

            if len(sceptr_data) == 0:
                self.logger.error("No valid records for SCEPTR")
                return False

            # Get embeddings
            with tempfile.TemporaryDirectory() as temp_dir:
                embeddings_file = os.path.join(temp_dir, 'train_embeddings.npy')

                if not self._run_sceptr_embeddings(sceptr_data, embeddings_file):
                    return False

                embeddings = np.load(embeddings_file)

            # Get filtered training data (matching SCEPTR conversion)
            filtered_df = training_df.iloc[sceptr_data.index]

            # Encode peptide features
            peptide_features = self._encode_peptide_features(filtered_df['Peptide'].tolist())

            # Combine features (64 TCR dims + 21 peptide dims = 85 dims)
            combined_features = np.hstack([embeddings, peptide_features])
            labels = filtered_df['Label'].values

            self.logger.info(f"Training on {len(combined_features)} samples ({np.sum(labels)} pos, {len(labels) - np.sum(labels)} neg)")

            # Train classifier
            if self.classifier_type == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                self._classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
            else:
                from sklearn.ensemble import RandomForestClassifier
                self._classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

            self._classifier.fit(combined_features, labels)
            self._training_df = training_df

            self.logger.info("Classifier training completed")
            return True

        except Exception as e:
            self.logger.error(f"Classifier training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using SCEPTR.

        The prediction workflow:
        1. Train classifier if not already trained
        2. Generate SCEPTR embeddings for test data
        3. Encode peptide features
        4. Use trained classifier for prediction

        Args:
            df: Input DataFrame with ID, Peptide, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))

        try:
            # Train classifier if needed
            if self._classifier is None:
                self.logger.info("Loading training data and training classifier...")

                # Load training data
                training_pos = self._load_training_data()
                if training_pos is None:
                    self.logger.error("Failed to load training data")
                    df_result['Prediction_Prob'] = 0.5
                    df_result = df_result.drop(columns=['_orig_idx'])
                    return df_result

                # Generate negatives
                training_neg = self._generate_training_negatives(training_pos)
                training_df = pd.concat([training_pos, training_neg], ignore_index=True)

                # Train classifier
                if not self._train_classifier(training_df):
                    self.logger.error("Failed to train classifier")
                    df_result['Prediction_Prob'] = 0.5
                    df_result = df_result.drop(columns=['_orig_idx'])
                    return df_result

            # Convert test data to SCEPTR format
            sceptr_data = self._convert_to_sceptr_format(df)
            valid_indices = sceptr_data.index.tolist()

            if len(sceptr_data) == 0:
                self.logger.warning("No valid records after SCEPTR conversion")
                df_result['Prediction_Prob'] = 0.5
                df_result = df_result.drop(columns=['_orig_idx'])
                return df_result

            # Generate embeddings for test data
            with tempfile.TemporaryDirectory() as temp_dir:
                embeddings_file = os.path.join(temp_dir, 'test_embeddings.npy')

                if not self._run_sceptr_embeddings(sceptr_data, embeddings_file):
                    self.logger.error("Failed to generate test embeddings")
                    df_result['Prediction_Prob'] = 0.5
                    df_result = df_result.drop(columns=['_orig_idx'])
                    return df_result

                embeddings = np.load(embeddings_file)

            # Get filtered test data
            filtered_df = df.iloc[valid_indices]

            # Encode peptide features
            peptide_features = self._encode_peptide_features(filtered_df['Peptide'].tolist())

            # Combine features
            combined_features = np.hstack([embeddings, peptide_features])

            # Predict probabilities
            predictions = self._classifier.predict_proba(combined_features)[:, 1]

            self.logger.info(f"Generated {len(predictions)} predictions")

            # Initialize all as 0.5 (default for filtered out records)
            df_result['Prediction_Prob'] = 0.5

            # Assign predictions to valid indices
            for i, orig_idx in enumerate(valid_indices):
                df_result.loc[df_result['_orig_idx'] == orig_idx, 'Prediction_Prob'] = predictions[i]

            self.logger.info(f"Sample predictions: {predictions[:3]}")

        except Exception as e:
            self.logger.error(f"SCEPTR prediction error: {e}")
            import traceback
            traceback.print_exc()
            df_result['Prediction_Prob'] = 0.5

        # Restore order and cleanup
        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])

        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        """Get path to SCEPTR training data (VDJDB)."""
        if hasattr(self, 'vdjdb_path') and self.vdjdb_path:
            return Path(self.vdjdb_path)
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load SCEPTR training data."""
        return self._load_training_data()
