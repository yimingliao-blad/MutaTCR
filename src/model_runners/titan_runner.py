"""
TITAN Model Runner for TCRP Benchmark V4.

TITAN (TCR–Epitope Binding Affinity Network) uses a transformer
architecture with peptide SMILES representation.

V4 Features:
- Config-based conda path and timeout
- Uses paths from config system

This runner follows the original benchmark_titan.py approach:
1. Generates TITAN format files dynamically (epitopes.smi, tcr.csv with CDR3b only)
2. Uses pytoda's aas_to_smiles for SMILES conversion
3. Runs TITAN prediction via paccmann_predictor
4. Maps predictions back to original rows

Note: Uses CDR3b sequences only (not full TCR with V+J genes) to match
the original working benchmark approach.
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

try:
    from .base_runner import BaseModelRunner
    from ..utils.config import get_config
except ImportError:
    from base_runner import BaseModelRunner
    from utils.config import get_config

logger = logging.getLogger(__name__)


class TITANRunner(BaseModelRunner):
    """
    Runner for TITAN model.

    TITAN uses transformer architecture with SMILES peptide representation
    and CDR3b sequences.
    """

    def __init__(self, mode: str = "exact_match"):
        super().__init__("TITAN", mode)
        self.titan_path = None
        self._titan_config = get_config()

    def setup(self) -> bool:
        try:
            models_dir = self.paths.MODELS_DIR
            titan_dir = models_dir / "TITAN"

            if not titan_dir.exists():
                self.logger.error(f"TITAN model not found at {titan_dir}")
                return False

            self.titan_path = titan_dir

            # Check for model files
            model_path = titan_dir / "trained_model"
            if not (model_path / "model_params.json").exists():
                self.logger.error(f"model_params.json not found at {model_path}")
                return False

            self.logger.info(f"TITAN setup complete: {titan_dir}")
            return True

        except Exception as e:
            self.logger.error(f"TITAN setup failed: {e}")
            return False

    def get_required_columns(self) -> List[str]:
        return ['Peptide', 'CDR3b']

    def _generate_titan_files(self, df: pd.DataFrame, output_dir: str) -> Tuple[Dict, Dict]:
        """
        Generate TITAN format files: epitopes.csv, epitopes.smi, tcr.csv, and test.csv.

        This follows the original benchmark_titan.py approach which generates
        ID mappings dynamically and uses CDR3b only (not full TCR sequences).

        Args:
            df: DataFrame with Peptide, CDR3b, Label columns
            output_dir: Directory to save TITAN files

        Returns:
            tuple: (epitope_to_id, tcr_to_id) mappings
        """
        self.logger.info("Generating TITAN format files...")

        # Step 1: Generate unique epitope and TCR IDs
        unique_epitopes = df['Peptide'].unique()
        epitope_to_id = {ep: i+1 for i, ep in enumerate(unique_epitopes)}

        unique_tcrs = df['CDR3b'].unique()
        tcr_to_id = {tcr: i+1 for i, tcr in enumerate(unique_tcrs)}

        # Step 2: Create epitopes.csv (amino acid sequences)
        epitopes_file = os.path.join(output_dir, 'epitopes.csv')
        with open(epitopes_file, 'w') as f:
            for epitope, ep_id in epitope_to_id.items():
                f.write(f"{epitope}\t{ep_id}\n")

        # Step 3: Create epitopes.smi (SMILES) - will be generated in subprocess
        # The SMILES conversion requires pytoda which is in the TITAN environment

        # Step 4: Create tcr.csv (CDR3b sequences only - not full TCR)
        tcr_file = os.path.join(output_dir, 'tcr.csv')
        with open(tcr_file, 'w') as f:
            for cdr3b, tcr_id in tcr_to_id.items():
                f.write(f"{cdr3b}\t{tcr_id}\n")

        # Step 5: Create test data file with correct format
        test_data = []
        for _, row in df.iterrows():
            epitope = row['Peptide']
            cdr3b = row['CDR3b']
            if epitope in epitope_to_id and cdr3b in tcr_to_id:
                test_data.append({
                    'ligand_name': epitope_to_id[epitope],
                    'sequence_id': tcr_to_id[cdr3b],
                    'label': row.get('Label', 1)
                })

        test_df = pd.DataFrame(test_data)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=True)

        self.logger.info(f"Generated TITAN files: {len(epitope_to_id)} epitopes, {len(tcr_to_id)} TCRs, {len(test_df)} test records")
        return epitope_to_id, tcr_to_id

    def _run_prediction(self, test_file: str, tcr_file: str, epitope_ids: Dict, output_dir: str) -> Optional[np.ndarray]:
        """
        Run TITAN prediction using paccmann_predictor directly.

        This script:
        1. Generates epitopes.smi from epitope_ids using pytoda's aas_to_smiles
        2. Loads the pre-trained model
        3. Runs predictions on the test data
        4. Returns prediction scores
        """
        conda_exe = self.paths.get_conda_executable()
        model_path = str(self.titan_path / "trained_model")
        titan_path = str(self.titan_path)
        epitope_file = os.path.join(output_dir, 'epitopes.smi')

        # Serialize epitope_ids for script - escape single quotes for safe embedding in Python string
        epitope_ids_json = json.dumps(epitope_ids).replace("'", "\\'")

        # Build script that generates SMILES and runs prediction
        script = f'''
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, '{titan_path}')

from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import DrugAffinityDataset
from pytoda.proteins import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.proteins import aas_to_smiles

torch.manual_seed(123456)

model_path = '{model_path}'
test_file = '{test_file}'
tcr_file = '{tcr_file}'
epitope_file = '{epitope_file}'
output_dir = '{output_dir}'

# Step 1: Generate epitopes.smi using pytoda's aas_to_smiles
epitope_ids = json.loads('{epitope_ids_json}')
with open(epitope_file, 'w') as f:
    for epitope, ep_id in epitope_ids.items():
        try:
            smiles = aas_to_smiles(epitope)
            f.write(f"{{smiles}}\\t{{ep_id}}\\n")
        except Exception as e:
            print(f"Warning: Failed to convert epitope {{epitope}}: {{e}}", file=sys.stderr)
            continue

print(f"Generated epitopes.smi with {{len(epitope_ids)}} epitopes", file=sys.stderr)

# Step 2: Load model params
with open(os.path.join(model_path, 'model_params.json')) as fp:
    params = json.load(fp)

device = get_device()

# Step 3: Load languages
smiles_language = SMILESTokenizer.from_pretrained(model_path)
smiles_language.set_encoding_transforms(
    randomize=None,
    add_start_and_stop=params.get('ligand_start_stop_token', True),
    padding=params.get('ligand_padding', True),
    padding_length=params.get('ligand_padding_length', True),
)
smiles_language.set_smiles_transforms(
    augment=False,
    canonical=params.get('smiles_canonical', False),
    kekulize=params.get('smiles_kekulize', False),
    all_bonds_explicit=params.get('smiles_bonds_explicit', False),
    all_hs_explicit=params.get('smiles_all_hs_explicit', False),
    remove_bonddir=params.get('smiles_remove_bonddir', False),
    remove_chirality=params.get('smiles_remove_chirality', False),
    selfies=params.get('selfies', False),
    sanitize=params.get('sanitize', False)
)

protein_language = ProteinLanguage.load(
    os.path.join(model_path, 'protein_language.pkl')
)

# Step 4: Create dataset
test_dataset = DrugAffinityDataset(
    drug_affinity_filepath=test_file,
    smi_filepath=epitope_file,
    protein_filepath=tcr_file,
    smiles_language=smiles_language,
    protein_language=protein_language,
    smiles_padding=params.get('ligand_padding', True),
    smiles_padding_length=params.get('ligand_padding_length', None),
    smiles_add_start_and_stop=params.get('ligand_add_start_stop', True),
    smiles_augment=False,
    smiles_canonical=params.get('test_smiles_canonical', False),
    smiles_kekulize=params.get('smiles_kekulize', False),
    smiles_all_bonds_explicit=params.get('smiles_bonds_explicit', False),
    smiles_all_hs_explicit=params.get('smiles_all_hs_explicit', False),
    smiles_remove_bonddir=params.get('smiles_remove_bonddir', False),
    smiles_remove_chirality=params.get('smiles_remove_chirality', False),
    smiles_selfies=params.get('selfies', False),
    protein_amino_acid_dict=params.get('protein_amino_acid_dict', 'iupac'),
    protein_padding=params.get('receptor_padding', True),
    protein_padding_length=params.get('receptor_padding_length', None),
    protein_add_start_and_stop=params.get('receptor_add_start_stop', True),
    protein_augment_by_revert=False,
    drug_affinity_dtype=torch.float,
    backend='eager',
    iterate_dataset=True
)

print(f"Dataset size: {{len(test_dataset)}}", file=sys.stderr)

# Step 5: DataLoader with drop_last=False
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

# Step 6: Load model
model_fn = params.get('model_fn', 'bimodal_mca')
model = MODEL_FACTORY[model_fn](params).to(device)
model._associate_language(smiles_language)
model._associate_language(protein_language)

model_file = os.path.join(model_path, 'weights', 'best_ROC-AUC_bimodal_mca.pt')
if os.path.isfile(model_file):
    model.load(model_file, map_location=device)
    print(f"Loaded model from {{model_file}}", file=sys.stderr)
else:
    print(f"Warning: Model file not found at {{model_file}}", file=sys.stderr)

# Step 7: Predict
model.eval()
predictions = []
with torch.no_grad():
    for ligand, receptors, y in test_loader:
        y_hat, _ = model(ligand.to(device), receptors.to(device))
        predictions.append(y_hat.cpu())

if predictions:
    predictions = torch.cat(predictions, dim=0).flatten().numpy()
    print(json.dumps(predictions.tolist()))
else:
    print("[]")
'''
        # Write script to temp file
        temp_script = tempfile.mktemp(suffix='.py')
        with open(temp_script, 'w') as f:
            f.write(script)

        env_name = self._titan_config.get_model_env('TITAN')
        cmd = [
            str(conda_exe), "run", "-n", env_name,
            "python", temp_script
        ]

        self.logger.info("Running TITAN prediction...")

        try:
            timeout = self._titan_config.get_model_timeout('TITAN')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.titan_path),
                timeout=timeout
            )

            os.unlink(temp_script)

            if result.returncode != 0:
                self.logger.error(f"Prediction failed: {result.stderr[-1000:]}")
                return None

            # Log stderr for debugging info
            if result.stderr:
                for line in result.stderr.split('\n')[-10:]:
                    if line.strip():
                        self.logger.info(f"TITAN: {line}")

            # Parse predictions from stdout
            if result.stdout.strip():
                predictions = np.array(json.loads(result.stdout.strip()))
                self.logger.info(f"Got {len(predictions)} predictions")
                return predictions
            else:
                self.logger.error("No predictions returned")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error("Prediction timed out")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return None
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return None

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using TITAN.

        Args:
            df: Input DataFrame with ID, Peptide, CDR3b, etc.

        Returns:
            DataFrame with all original columns + Prediction_Prob
        """
        df_result = df.copy()
        df_result['_orig_idx'] = range(len(df))
        df_result['Prediction_Prob'] = 0.5  # Default

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Step 1: Generate TITAN format files
                epitope_to_id, tcr_to_id = self._generate_titan_files(df, temp_dir)

                test_file = os.path.join(temp_dir, 'test.csv')
                tcr_file = os.path.join(temp_dir, 'tcr.csv')

                # Step 2: Run prediction (generates epitopes.smi and runs model)
                predictions = self._run_prediction(test_file, tcr_file, epitope_to_id, temp_dir)

                if predictions is not None:
                    # Load test file to get mapping
                    test_df = pd.read_csv(test_file)

                    if len(predictions) == len(test_df):
                        # Map predictions back - order should be preserved
                        for i in range(min(len(predictions), len(df))):
                            df_result.loc[df_result['_orig_idx'] == i, 'Prediction_Prob'] = predictions[i]
                        self.logger.info(f"Matched {len(predictions)} predictions")
                    else:
                        self.logger.warning(f"Count mismatch: {len(predictions)} predictions vs {len(test_df)} test samples")
                else:
                    self.logger.error("Prediction failed")

        except Exception as e:
            self.logger.error(f"TITAN error: {e}")
            import traceback
            traceback.print_exc()

        df_result = df_result.sort_values('_orig_idx')
        df_result = df_result.drop(columns=['_orig_idx'])
        return df_result

    def get_training_data_path(self) -> Optional[Path]:
        if self.titan_path:
            training_file = self.titan_path / "datasets" / "trained_model_train.csv"
            if training_file.exists():
                return training_file
        return None

    def load_training_data(self) -> Optional[pd.DataFrame]:
        path = self.get_training_data_path()
        if path and path.exists():
            return pd.read_csv(path)
        return None
