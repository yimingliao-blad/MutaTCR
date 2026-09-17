"""
Raw Data Preprocessing for TCRP Benchmark V2.

This module processes raw data from three datasets:
- TetTCR-SeqHD: Kevin's annotations + UMI count matrix
- IMMREP23: Direct copy with label preservation
- FingerPrinting: TCR-peptide binding with log2foldchange

Output: Unified CSV files in data/unified/
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import get_paths, get_config, setup_logging, is_valid_sequence, clean_sequence

logger = logging.getLogger(__name__)


class RawDataPreprocessor:
    """
    Preprocessor for converting raw data to unified CSV format.

    All three datasets produce unified CSVs with common columns:
    - ID, Peptide, CDR3a, CDR3b, Va, Ja, Vb, Jb, Label, Epitope_Group
    - Plus dataset-specific columns (UMI, UMI_Fraction, log2foldchange, TCR_Group)
    """

    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self):
        """Initialize preprocessor with paths and config."""
        self.paths = get_paths()
        self.config = get_config()
        self.stats = {}

    def process_all(self) -> Dict[str, Path]:
        """
        Process all three datasets.

        Returns:
            Dictionary mapping dataset names to output paths
        """
        logger.info("=" * 60)
        logger.info("Starting raw data preprocessing")
        logger.info("=" * 60)

        results = {}

        # Process TetTCR-SeqHD
        logger.info("\n[1/3] Processing TetTCR-SeqHD...")
        results['tettcr'] = self.process_tettcr()

        # Process IMMREP23
        logger.info("\n[2/3] Processing IMMREP23...")
        results['immrep23'] = self.process_immrep23()

        # Process FingerPrinting
        logger.info("\n[3/3] Processing FingerPrinting...")
        results['fingerprinting'] = self.process_fingerprinting()

        # Print summary
        self._print_summary()

        return results

    def process_tettcr(self) -> Path:
        """
        Process TetTCR-SeqHD dataset from TCR_antigen_binding_sheet.csv.

        Data sources:
        - TCR_antigen_binding_sheet.csv: Pre-processed binding data with classification
          (strong_positive, strong_negative, possible_cross_reactive)
        - Kevin's Publication TCRs Updated.xlsx: Antigen info (peptide sequences, HLA, groups)

        Classification (from BINDING_CLASSIFICATION_RULES.md):
        - strong_positive=1 → Label=1 (positive binding)
        - strong_negative=1 → Label=0 (negative binding)
        - possible_cross_reactive=1 → Excluded (uncertain)

        Returns:
            Path to output unified CSV
        """
        # Load files
        tettcr_dir = self.paths.RAW_DATA_DIR / 'TetTCR-SeqHD'
        binding_sheet = tettcr_dir / 'TCR_antigen_binding_sheet.csv'
        kevin_file = self.paths.get_tettcr_tcr_file()

        if not binding_sheet.exists():
            raise FileNotFoundError(f"TCR_antigen_binding_sheet.csv not found: {binding_sheet}")

        logger.info(f"  Loading binding sheet: {binding_sheet}")
        logger.info(f"  Loading antigen info: {kevin_file}")

        # Load the binding sheet (pre-processed with classification)
        df_binding = pd.read_csv(binding_sheet)
        logger.info(f"  Binding sheet: {len(df_binding)} rows")

        # Load Antigen sequences from Kevin's file for peptide sequences and HLA
        df_antigens = pd.read_excel(kevin_file, sheet_name='Antigen_sequences', skiprows=[0], engine='openpyxl')
        logger.info(f"  Antigen info: {len(df_antigens)} records")

        # Create antigen info mapping (normalized name -> info)
        # Normalize: replace - with . and uppercase to match binding sheet format
        antigen_info = {}
        for _, row in df_antigens.iterrows():
            antigen_name = str(row['Antigen']).strip()
            peptide_seq = row['AA sequence of antigen']
            if pd.isna(peptide_seq) or peptide_seq == 'Empty':
                continue
            if not self._is_valid_sequence(peptide_seq):
                continue
            # Normalize: replace - with . and uppercase (matching binding sheet format)
            normalized_name = antigen_name.replace('-', '.').upper()
            antigen_info[normalized_name] = {
                'peptide': peptide_seq,
                'group': row['Group'],
                'hla': row['HLA'] if pd.notna(row['HLA']) else 'HLA-A*02:01',
                'original_name': antigen_name
            }

        logger.info(f"  Valid antigens with peptide sequences: {len(antigen_info)}")

        # Filter to strong_positive and strong_negative only (exclude cross-reactive)
        df_filtered = df_binding[
            (df_binding['strong_positive'] == 1) | (df_binding['strong_negative'] == 1)
        ].copy()
        logger.info(f"  After filtering (excluding cross-reactive): {len(df_filtered)} rows")

        # Process rows
        logger.info("  Processing rows...")
        records = []
        record_id = 0
        skipped_no_peptide = 0
        skipped_invalid_cdr3 = 0

        for _, row in df_filtered.iterrows():
            # Get CDR3 sequences
            cdr3a = row['tcr_a_cdr3']
            cdr3b = row['tcr_b_cdr3']

            # Validate CDR3 sequences
            if not self._is_valid_sequence(cdr3a) or not self._is_valid_sequence(cdr3b):
                skipped_invalid_cdr3 += 1
                continue

            # Get antigen info - normalize to match Kevin's file
            antigen_name = str(row['antigen']).upper()
            # Fix typo: binding sheet has RTPRN but Kevin's file has PTPRN
            if antigen_name.startswith('RTPRN'):
                antigen_name = 'P' + antigen_name[1:]  # RTPRN -> PTPRN
            if antigen_name not in antigen_info:
                skipped_no_peptide += 1
                continue

            antigen = antigen_info[antigen_name]
            raw_group = antigen['group']

            # Map group to Epitope_Group (keep all groups including T1D)
            if raw_group == 'Viral':
                epitope_group = 'Viral'
            elif raw_group == 'T1D':
                epitope_group = 'T1D'
            elif raw_group == 'self':
                epitope_group = 'self'
            elif raw_group == 'Empty':
                epitope_group = 'Empty'
            else:
                epitope_group = 'Other'

            # Set label based on strong_positive/strong_negative
            label = 1 if row['strong_positive'] == 1 else 0

            records.append({
                'ID': f"tettcr_{record_id}",
                'Cell_ID': row['TCR'],
                'Peptide': antigen['peptide'],
                'Peptide_Name': antigen['original_name'],
                'HLA': antigen['hla'],
                'TRAV': row['tcr_a_v'],
                'TRAJ': row['tcr_a_j'],
                'CDR3a': cdr3a,
                'TRBV': row['tcr_b_v'],
                'TRBJ': row['tcr_b_j'],
                'CDR3b': cdr3b,
                'Va': row['tcr_a_v'],
                'Ja': row['tcr_a_j'],
                'Vb': row['tcr_b_v'],
                'Jb': row['tcr_b_j'],
                'Label': label,
                'UMI': float(row['avg_umi']),
                'UMI_Fraction': float(row['avg_signal_ratio']),
                'Epitope_Group': epitope_group,
            })
            record_id += 1

        logger.info(f"  Skipped {skipped_no_peptide} rows with unmapped antigen")
        logger.info(f"  Skipped {skipped_invalid_cdr3} rows with invalid CDR3")
        logger.info(f"  Total records: {len(records)}")

        df_unified = pd.DataFrame(records)

        # Round numeric columns
        df_unified['UMI'] = df_unified['UMI'].round(1)
        df_unified['UMI_Fraction'] = df_unified['UMI_Fraction'].round(6)

        # Save
        output_path = self.paths.get_unified_file('tettcr')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(output_path, index=False)

        # Stats
        self.stats['tettcr'] = {
            'total_records': len(df_unified),
            'positives': (df_unified['Label'] == 1).sum(),
            'negatives': (df_unified['Label'] == 0).sum(),
            'unique_tcrs': df_unified['Cell_ID'].nunique(),
            'unique_peptides': df_unified['Peptide'].nunique(),
            'epitope_groups': df_unified['Epitope_Group'].value_counts().to_dict()
        }

        logger.info(f"  Output: {output_path}")
        logger.info(f"  Total records: {len(df_unified)}")
        logger.info(f"  Positives: {self.stats['tettcr']['positives']}, Negatives: {self.stats['tettcr']['negatives']}")
        logger.info(f"  Unique TCRs: {self.stats['tettcr']['unique_tcrs']}")
        logger.info(f"  Unique peptides: {self.stats['tettcr']['unique_peptides']}")
        logger.info(f"  Epitope groups: {self.stats['tettcr']['epitope_groups']}")

        return output_path

    def process_immrep23(self) -> Path:
        """
        Process IMMREP23 dataset.

        This dataset already has labels, so we preserve them.
        Add Epitope_Group based on seen/unseen peptide classification.

        Returns:
            Path to output unified CSV
        """
        input_file = self.paths.get_immrep23_file()
        logger.info(f"  Loading: {input_file}")

        df = pd.read_csv(input_file)
        logger.info(f"  Raw records: {len(df)}")

        # Define seen/unseen peptides (from V1 benchmark - pipeline_utils.py)
        # These are the exact 17 seen and 3 unseen peptides from the original benchmark
        SEEN_PEPTIDES = [
            'EPLPQGQLTAY',   # 1
            'GILGFVFTL',     # 2 - Influenza A M1
            'GLCTLVAML',     # 3 - EBV BMLF1
            'IPSINVHHY',     # 4 - CMV
            'IVTDFSVIK',     # 5 - EBV EBNA3A
            'NLVPMVATV',     # 6 - CMV pp65
            'QIKVRVDMV',     # 7
            'RAKFKQLL',      # 8 - EBV BZLF1
            'RPHERNGFTVL',   # 9
            'RPPIFIRRL',     # 10 - HIV Gag
            'TDLGQNLLY',     # 11
            'TPRVTGGGAM',    # 12 - CMV
            'VLEETSVML',     # 13
            'VSDGGPNLY',     # 14
            'VTEHDTLLY',     # 15
            'YLQPRTFLL',     # 16
            'YVLDHLIVV',     # 17 - EBV BRLF1
        ]
        UNSEEN_PEPTIDES = [
            'FTDALGIDEY',    # 1
            'SALPTNADLY',    # 2
            'TSDACMMTMY',    # 3
        ]

        # Map peptides to groups
        def get_epitope_group(peptide):
            if peptide in SEEN_PEPTIDES:
                return 'seen'
            elif peptide in UNSEEN_PEPTIDES:
                return 'unseen'
            return 'unknown'

        # Add Epitope_Group column to existing data
        df['Epitope_Group'] = df['Peptide'].apply(get_epitope_group)

        # Use existing columns - IMMREP23 already has proper format
        # Just add ID prefix and Epitope_Group
        df_unified = df.copy()
        df_unified['ID'] = [f"immrep23_{idx}" for idx in range(len(df))]

        # Ensure required columns exist
        required_cols = ['ID', 'Peptide', 'HLA', 'CDR3a', 'CDR3b', 'Va', 'Ja', 'Vb', 'Jb', 'Label', 'Epitope_Group']
        for col in required_cols:
            if col not in df_unified.columns:
                df_unified[col] = ''

        # Filter invalid sequences
        df_unified = df_unified[df_unified['CDR3b'].apply(lambda x: self._is_valid_sequence(x) if pd.notna(x) and x else True)]

        # Save
        output_path = self.paths.get_unified_file('immrep23')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(output_path, index=False)

        # Stats
        self.stats['immrep23'] = {
            'total_records': len(df_unified),
            'positives': (df_unified['Label'] == 1).sum(),
            'negatives': (df_unified['Label'] == 0).sum(),
            'unique_peptides': df_unified['Peptide'].nunique(),
            'epitope_groups': df_unified['Epitope_Group'].value_counts().to_dict()
        }

        logger.info(f"  Output: {output_path}")
        logger.info(f"  Total records: {len(df_unified)}")
        logger.info(f"  Epitope groups: {self.stats['immrep23']['epitope_groups']}")

        return output_path

    def process_fingerprinting(self) -> Path:
        """
        Process FingerPrinting dataset.

        Data source: Fingerprinting_TCR_Data_Combined_final.xlsx
        Additional: finger_print.xlsx for log2foldchange reference

        Labeling: log2foldchange > 0 = positive (Label=1), else negative

        Returns:
            Path to output unified CSV
        """
        input_file = self.paths.get_fingerprinting_file()
        log2fc_file = self.paths.get_fingerprinting_log2fc_file()

        logger.info(f"  Loading: {input_file}")

        # Load peptide status mapping from raw finger_print.csv
        peptide_status_mapping = self._load_peptide_status_mapping()

        # Read Excel sheets
        xl = pd.ExcelFile(input_file, engine='openpyxl')

        df_tcr = pd.read_excel(xl, sheet_name='TCR_Info', skiprows=[0])
        df_specificity = pd.read_excel(xl, sheet_name='Specificity_Data', skiprows=[0])
        df_epitope = pd.read_excel(xl, sheet_name='Epitope_Info', skiprows=[0])

        logger.info(f"  TCR_Info: {len(df_tcr)} records")
        logger.info(f"  Specificity_Data: {len(df_specificity)} records")
        logger.info(f"  Epitope_Info: {len(df_epitope)} records")

        # Standardize column names
        df_tcr.columns = df_tcr.columns.str.strip()
        df_specificity.columns = df_specificity.columns.str.strip()
        df_epitope.columns = df_epitope.columns.str.strip()

        # Rename lowercase columns to match
        df_specificity = df_specificity.rename(columns={
            'coord': 'COORD',
            'plate': 'Plate'
        })

        # Store original TCR name (with dots: SVAR.1) for TCR_Group
        df_tcr['TCR_Original'] = df_tcr['TCR'].copy()
        # Convert TCR_Info TCR format (SVAR.1) to Specificity_Data format (SVAR-1)
        df_tcr['TCR'] = df_tcr['TCR'].str.replace('.', '-', regex=False)

        # Merge data
        df_merge = pd.merge(df_tcr, df_specificity, on='TCR', how='inner')
        df_merge = pd.merge(df_merge, df_epitope, on=['COORD', 'Plate'], how='inner')

        logger.info(f"  After merging: {len(df_merge)} records")

        # Reference peptide for epitope group classification
        # Based on config: fingerprinting.index4_reference_peptide
        REFERENCE_PEPTIDE = "YLQPRTFLL"

        def get_epitope_group(peptide):
            """
            Classify peptide into epitope group for fingerprinting dataset.

            Rules (from benchmark_config.yaml):
            - Reference peptide (YLQPRTFLL) → 'non-R-5'
            - Single mutation at index 4 (position 5) compared to reference → 'R-5'
            - All other peptides → 'non-R-5'
            """
            if peptide == REFERENCE_PEPTIDE:
                return 'non-R-5'

            # Check if peptide is a single mutation at index 4 of reference
            if len(peptide) == len(REFERENCE_PEPTIDE) == 9:
                # Count total differences
                diffs = sum(1 for i in range(len(peptide)) if peptide[i] != REFERENCE_PEPTIDE[i])

                # Single mutation at index 4 (position 5) = R-5
                if diffs == 1 and peptide[4] != REFERENCE_PEPTIDE[4]:
                    return 'R-5'

            return 'non-R-5'

        # Create unified format
        records = []
        record_id = 0

        for _, row in df_merge.iterrows():
            peptide = row.get('Peptide', row.get('EPITOPE', ''))
            log2fc = row.get('log2foldchangevalue', row.get('log2foldchange', 0))

            if pd.isna(peptide) or not peptide:
                continue

            # Get TCR group from TCR_Original column (with dots: SVAR.1)
            tcr_group = row.get('TCR_Original', row.get('TCR', ''))

            # Get antigen status from mapping (default to 'valid' if not found)
            antigen_status = peptide_status_mapping.get(peptide, 'valid')

            records.append({
                'ID': f"fp_{record_id}",
                'Peptide': peptide,
                'HLA': row.get('HLA', 'HLA-A*02:01'),
                'CDR3a': row.get('CDR3a', ''),
                'CDR3b': row.get('CDR3b', ''),
                'Va': row.get('TRAV', ''),
                'Ja': row.get('TRAJ', ''),
                'Vb': row.get('TRBV', ''),
                'Jb': row.get('TRBJ', ''),
                'Label': 1 if log2fc > 0 else 0,
                'log2foldchange': log2fc,
                'TCR_Group': tcr_group,
                'Epitope_Group': get_epitope_group(peptide),
                'Antigen_Status': antigen_status,
            })
            record_id += 1

        df_unified = pd.DataFrame(records)

        # Filter invalid sequences
        df_unified = df_unified[df_unified['CDR3b'].apply(lambda x: self._is_valid_sequence(x) if pd.notna(x) and x else True)]

        # Save
        output_path = self.paths.get_unified_file('fingerprinting')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(output_path, index=False)

        # Stats
        self.stats['fingerprinting'] = {
            'total_records': len(df_unified),
            'positives': (df_unified['Label'] == 1).sum(),
            'negatives': (df_unified['Label'] == 0).sum(),
            'unique_tcrs': df_unified['TCR_Group'].nunique(),
            'unique_peptides': df_unified['Peptide'].nunique(),
            'epitope_groups': df_unified['Epitope_Group'].value_counts().to_dict(),
            'antigen_status': df_unified['Antigen_Status'].value_counts().to_dict()
        }

        logger.info(f"  Output: {output_path}")
        logger.info(f"  Total records: {len(df_unified)}")
        logger.info(f"  TCR groups: {self.stats['fingerprinting']['unique_tcrs']}")
        logger.info(f"  Epitope groups: {self.stats['fingerprinting']['epitope_groups']}")
        logger.info(f"  Antigen status: {self.stats['fingerprinting']['antigen_status']}")

        return output_path

    def _is_valid_sequence(self, seq: str) -> bool:
        """Check if sequence contains only valid amino acids."""
        if not seq or not isinstance(seq, str):
            return False
        return set(seq.upper()).issubset(self.VALID_AMINO_ACIDS)

    def _load_peptide_status_mapping(self) -> Dict[str, str]:
        """
        Create peptide sequence -> Antigen_Status mapping from raw finger_print.csv.

        Parses the 'Antigen peptide' column to classify each peptide:
        - 'unstable': peptides ending with '_unstable'
        - 'wild type': peptide 'YLQPRTFLL wild type'
        - 'valid': all other peptides (no suffix)

        Returns:
            Dictionary mapping clean peptide sequence to status
        """
        raw_csv = self.paths.RAW_DATA_DIR / 'FingerPrinting' / 'finger_print.csv'

        if not raw_csv.exists():
            logger.warning(f"  Raw finger_print.csv not found: {raw_csv}")
            return {}

        status_mapping = {}

        try:
            df_raw = pd.read_csv(raw_csv)
            antigen_col = df_raw.columns[0]  # First column is 'Antigen peptide'

            for raw_name in df_raw[antigen_col]:
                if pd.isna(raw_name):
                    continue

                raw_name = str(raw_name).strip()

                # Classify based on suffix
                if raw_name.endswith('_unstable'):
                    # Remove '_unstable' suffix to get clean sequence
                    clean_seq = raw_name[:-9]  # len('_unstable') = 9
                    status = 'unstable'
                elif raw_name.endswith(' wild type'):
                    # Remove ' wild type' suffix to get clean sequence
                    clean_seq = raw_name[:-10]  # len(' wild type') = 10
                    status = 'wild type'
                else:
                    # No suffix - this is a valid peptide
                    clean_seq = raw_name
                    status = 'valid'

                # Only set status if not already mapped, OR if new status is not 'valid'
                # This prevents 'wild type' or 'unstable' from being overwritten by 'valid'
                if clean_seq not in status_mapping or status != 'valid':
                    status_mapping[clean_seq] = status

            logger.info(f"  Loaded peptide status mapping: {len(status_mapping)} peptides")
            status_counts = {}
            for s in status_mapping.values():
                status_counts[s] = status_counts.get(s, 0) + 1
            logger.info(f"  Status distribution: {status_counts}")

        except Exception as e:
            logger.warning(f"  Error loading peptide status mapping: {e}")

        return status_mapping

    def _print_summary(self):
        """Print processing summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("=" * 60)

        for dataset, stats in self.stats.items():
            logger.info(f"\n{dataset.upper()}:")
            logger.info(f"  Total records: {stats['total_records']}")
            logger.info(f"  Positives: {stats['positives']} ({stats['positives']/stats['total_records']*100:.1f}%)")
            logger.info(f"  Negatives: {stats['negatives']} ({stats['negatives']/stats['total_records']*100:.1f}%)")
            logger.info(f"  Epitope groups: {stats['epitope_groups']}")


def process_all_datasets() -> bool:
    """
    Process all datasets (wrapper function for run_benchmark.py).

    Returns:
        True if successful
    """
    try:
        preprocessor = RawDataPreprocessor()
        preprocessor.process_all()
        return True
    except Exception as e:
        logger.error(f"Error processing datasets: {e}")
        return False


def main():
    """Main entry point for preprocessing."""
    # Setup logging
    setup_logging("preprocess_raw", level="INFO")

    # Run preprocessing
    preprocessor = RawDataPreprocessor()
    results = preprocessor.process_all()

    logger.info("\nPreprocessing complete!")
    for dataset, path in results.items():
        logger.info(f"  {dataset}: {path}")


if __name__ == "__main__":
    main()
