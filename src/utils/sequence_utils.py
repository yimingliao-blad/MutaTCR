"""
Sequence utilities for TCRP Benchmark V2.

This module provides utilities for working with amino acid sequences,
including CDR3 and peptide processing.
"""

import re
import logging
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import optimized Levenshtein library (C implementation)
try:
    import Levenshtein as _levenshtein_lib
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logger.warning("python-Levenshtein not available, using slower Python implementation")


# Standard amino acids
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Amino acid mapping for non-standard codes
AA_MAPPING = {
    'X': '',  # Unknown
    'B': 'N',  # Aspartic acid or Asparagine -> Asparagine
    'Z': 'Q',  # Glutamic acid or Glutamine -> Glutamine
    'J': 'L',  # Leucine or Isoleucine -> Leucine
    'U': 'C',  # Selenocysteine -> Cysteine
    'O': 'K',  # Pyrrolysine -> Lysine
}


def is_valid_sequence(sequence: str) -> bool:
    """
    Check if a sequence contains only valid amino acids.

    Args:
        sequence: Amino acid sequence

    Returns:
        True if valid, False otherwise
    """
    if not sequence or not isinstance(sequence, str):
        return False
    return all(aa in AMINO_ACIDS for aa in sequence.upper())


def clean_sequence(sequence: str) -> str:
    """
    Clean an amino acid sequence by removing invalid characters.

    Args:
        sequence: Amino acid sequence

    Returns:
        Cleaned sequence with only standard amino acids
    """
    if not sequence or not isinstance(sequence, str):
        return ""

    sequence = sequence.upper().strip()

    # Map non-standard amino acids
    for old, new in AA_MAPPING.items():
        sequence = sequence.replace(old, new)

    # Remove any remaining non-amino acid characters
    cleaned = ''.join(aa for aa in sequence if aa in AMINO_ACIDS)

    return cleaned


def get_sequence_length(sequence: str) -> int:
    """
    Get length of a sequence.

    Args:
        sequence: Amino acid sequence

    Returns:
        Sequence length (0 if None or empty)
    """
    if not sequence or not isinstance(sequence, str):
        return 0
    return len(sequence.strip())


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Uses the python-Levenshtein C library if available (much faster),
    otherwise falls back to pure Python implementation.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance
    """
    # Use optimized C library if available
    if LEVENSHTEIN_AVAILABLE:
        return _levenshtein_lib.distance(s1, s2)

    # Fallback to pure Python implementation
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def is_fuzzy_match(seq1: str, seq2: str, max_distance: int = 3) -> bool:
    """
    Check if two sequences are within fuzzy match distance.

    Args:
        seq1: First sequence
        seq2: Second sequence
        max_distance: Maximum Levenshtein distance

    Returns:
        True if sequences are within max_distance edits
    """
    if not seq1 or not seq2:
        return False
    return levenshtein_distance(seq1.upper(), seq2.upper()) <= max_distance


def get_position_and_aa(peptide: str, reference: str = "YLQPRTFLL") -> Tuple[int, str]:
    """
    Get the position and amino acid of a single substitution variant.

    Args:
        peptide: Peptide variant sequence
        reference: Reference peptide sequence

    Returns:
        Tuple of (position, substituted_amino_acid) where position is 1-indexed
        Returns (0, '') if no single substitution found
    """
    if len(peptide) != len(reference):
        return (0, '')

    diff_positions = []
    for i, (p, r) in enumerate(zip(peptide.upper(), reference.upper())):
        if p != r:
            diff_positions.append((i + 1, p))

    if len(diff_positions) == 1:
        return diff_positions[0]
    return (0, '')


def get_reference_aa_at_position(position: int, reference: str = "YLQPRTFLL") -> str:
    """
    Get the amino acid at a position in the reference peptide.

    Args:
        position: 1-indexed position
        reference: Reference peptide sequence

    Returns:
        Amino acid at position or empty string if out of range
    """
    if 1 <= position <= len(reference):
        return reference[position - 1].upper()
    return ''


def parse_tcr_info(tcr_string: str) -> dict:
    """
    Parse TCR information from a combined string.

    Args:
        tcr_string: TCR string (may contain multiple TCRs separated by '|')

    Returns:
        Dictionary with TCR fields
    """
    result = {
        'TRAV': '',
        'TRAJ': '',
        'CDR3a': '',
        'TRBV': '',
        'TRBJ': '',
        'CDR3b': ''
    }

    if not tcr_string or not isinstance(tcr_string, str):
        return result

    # Handle pipe-separated values (take first)
    if '|' in tcr_string:
        tcr_string = tcr_string.split('|')[0]

    return result


def normalize_gene_name(gene: str) -> str:
    """
    Normalize V/J gene name format.

    Args:
        gene: Gene name (e.g., "TRBV12-1*01", "TRAV1-2")

    Returns:
        Normalized gene name
    """
    if not gene or not isinstance(gene, str):
        return ""

    gene = gene.strip()

    # Remove allele specification if present
    if '*' in gene:
        gene = gene.split('*')[0]

    return gene


def truncate_cdr3(cdr3: str, max_length: int = 30) -> str:
    """
    Truncate CDR3 sequence if too long.

    Args:
        cdr3: CDR3 sequence
        max_length: Maximum allowed length

    Returns:
        Truncated sequence
    """
    if not cdr3:
        return ""
    return cdr3[:max_length] if len(cdr3) > max_length else cdr3


if __name__ == "__main__":
    # Test sequence utilities
    print("Testing sequence utilities...")

    # Test validation
    print(f"\nis_valid_sequence('CASSQDRG'): {is_valid_sequence('CASSQDRG')}")
    print(f"is_valid_sequence('CASS123'): {is_valid_sequence('CASS123')}")

    # Test cleaning
    print(f"\nclean_sequence('CASSXQDRG'): {clean_sequence('CASSXQDRG')}")

    # Test Levenshtein distance
    print(f"\nlevenshtein_distance('CASSQDRG', 'CASSQDRG'): {levenshtein_distance('CASSQDRG', 'CASSQDRG')}")
    print(f"levenshtein_distance('CASSQDRG', 'CASSQDRA'): {levenshtein_distance('CASSQDRG', 'CASSQDRA')}")
    print(f"levenshtein_distance('CASSQDRG', 'CASAQDRA'): {levenshtein_distance('CASSQDRG', 'CASAQDRA')}")

    # Test fuzzy match
    print(f"\nis_fuzzy_match('CASSQDRG', 'CASSQDRA', max_distance=3): {is_fuzzy_match('CASSQDRG', 'CASSQDRA', max_distance=3)}")

    # Test position finding
    print(f"\nget_position_and_aa('ALQPRTFLL', 'YLQPRTFLL'): {get_position_and_aa('ALQPRTFLL', 'YLQPRTFLL')}")
    print(f"get_position_and_aa('YLQARTFLL', 'YLQPRTFLL'): {get_position_and_aa('YLQARTFLL', 'YLQPRTFLL')}")

    print("\nSequence utilities test completed!")
