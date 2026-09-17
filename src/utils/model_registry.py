"""
Model Registry for TCRP Benchmark V2.

This module serves as the SINGLE SOURCE OF TRUTH for model naming,
folder names, display names, and configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a TCR-peptide binding prediction model."""
    key: str                    # Internal key (e.g., "ERGO")
    display_name: str           # Display name for plots (e.g., "ERGO")
    folder_name: str            # Folder name in output (e.g., "ERGO")
    env_name: str               # Conda environment name
    model_type: str             # "pretrained", "meta_learning", "trained_classifier"
    has_checkpoint: bool        # Whether model uses pretrained checkpoint
    repository: str             # GitHub URL or install command
    input_columns: List[str]    # Required input columns


# Model definitions - SINGLE SOURCE OF TRUTH
MODELS: Dict[str, ModelInfo] = {
    "ERGO": ModelInfo(
        key="ERGO",
        display_name="ERGO",
        folder_name="ERGO",
        env_name="ergo-benchmark",
        model_type="pretrained",
        has_checkpoint=True,
        repository="https://github.com/louzounlab/ERGO",
        input_columns=["CDR3b", "Peptide"]
    ),
    "ERGO2": ModelInfo(
        key="ERGO2",
        display_name="ERGO-II",
        folder_name="ERGO2",
        env_name="ergo2-benchmark",
        model_type="pretrained",
        has_checkpoint=True,
        repository="https://github.com/IdoSpringer/ERGO-II",
        input_columns=["CDR3a", "CDR3b", "Va", "Ja", "Vb", "Jb", "Peptide"]
    ),
    "NetTCR": ModelInfo(
        key="NetTCR",
        display_name="NetTCR-2.0",
        folder_name="NetTCR",
        env_name="nettcr-benchmark",
        model_type="pretrained",
        has_checkpoint=True,
        repository="https://github.com/mnielLab/NetTCR-2.0",
        input_columns=["CDR3a", "CDR3b", "Peptide"]
    ),
    "NetTCR22": ModelInfo(
        key="NetTCR22",
        display_name="NetTCR-2.2",
        folder_name="NetTCR22",
        env_name="nettcr22-benchmark",
        model_type="pretrained",
        has_checkpoint=True,
        repository="https://github.com/mnielLab/NetTCR-2.2",
        input_columns=["CDR1a", "CDR2a", "CDR3a", "CDR1b", "CDR2b", "CDR3b", "Peptide"]
    ),
    "TITAN": ModelInfo(
        key="TITAN",
        display_name="TITAN",
        folder_name="TITAN",
        env_name="titan-benchmark",
        model_type="pretrained",
        has_checkpoint=True,
        repository="https://github.com/PaccMann/TITAN",
        input_columns=["CDR3b", "Peptide"]
    ),
    "EPACT": ModelInfo(
        key="EPACT",
        display_name="EPACT",
        folder_name="EPACT",
        env_name="epact-benchmark",
        model_type="pretrained",
        has_checkpoint=True,
        repository="https://github.com/zhangyumeng1sjtu/EPACT",
        input_columns=["CDR3a", "CDR3b", "Peptide", "MHC"]
    ),
    "PanPep": ModelInfo(
        key="PanPep",
        display_name="PanPep",
        folder_name="PanPep",
        env_name="PanPep_env",
        model_type="meta_learning",
        has_checkpoint=True,
        repository="https://github.com/bm2-lab/PanPep",
        input_columns=["CDR3b", "Peptide"]
    ),
    "SCEPTR": ModelInfo(
        key="SCEPTR",
        display_name="SCEPTR",
        folder_name="SCEPTR",
        env_name="sceptr_env",
        model_type="trained_classifier",
        has_checkpoint=False,
        repository="pip install sceptr",
        input_columns=["CDR3b", "Peptide"]
    ),
}

# Model order for plots (use internal keys for column matching)
MODEL_ORDER: List[str] = [
    "ERGO",
    "ERGO2",
    "NetTCR",
    "NetTCR22",
    "TITAN",
    "EPACT",
    "PanPep",
    "SCEPTR",
]

# Mapping from display name to key
DISPLAY_NAME_TO_KEY: Dict[str, str] = {
    model.display_name: model.key for model in MODELS.values()
}


def get_model(key: str) -> Optional[ModelInfo]:
    """
    Get model info by key.

    Args:
        key: Model key (e.g., "ERGO", "NetTCR22")

    Returns:
        ModelInfo or None if not found
    """
    return MODELS.get(key)


def get_model_by_display_name(display_name: str) -> Optional[ModelInfo]:
    """
    Get model info by display name.

    Args:
        display_name: Display name (e.g., "NetTCR-2.2")

    Returns:
        ModelInfo or None if not found
    """
    key = DISPLAY_NAME_TO_KEY.get(display_name)
    if key:
        return MODELS.get(key)
    return None


def get_all_models() -> List[ModelInfo]:
    """Get list of all model infos."""
    return list(MODELS.values())


def get_all_model_keys() -> List[str]:
    """Get list of all model keys."""
    return list(MODELS.keys())


def get_display_names() -> List[str]:
    """Get list of all display names in order."""
    return MODEL_ORDER


def get_folder_names() -> List[str]:
    """Get list of all folder names."""
    return [model.folder_name for model in MODELS.values()]


def get_model_display_name(key: str) -> str:
    """
    Get display name for a model key.

    Args:
        key: Model key

    Returns:
        Display name or key if not found
    """
    model = MODELS.get(key)
    return model.display_name if model else key


def get_model_folder_name(key: str) -> str:
    """
    Get folder name for a model key.

    Args:
        key: Model key

    Returns:
        Folder name or key if not found
    """
    model = MODELS.get(key)
    return model.folder_name if model else key


def get_model_key_from_display(display_name: str) -> str:
    """
    Get model key from display name.

    Args:
        display_name: Display name

    Returns:
        Model key or display_name if not found
    """
    return DISPLAY_NAME_TO_KEY.get(display_name, display_name)


if __name__ == "__main__":
    # Test model registry
    print("Testing Model Registry...")

    print(f"\nAll models ({len(MODELS)}):")
    for key, model in MODELS.items():
        print(f"  {key}: {model.display_name} ({model.model_type})")

    print(f"\nModel order for plots: {MODEL_ORDER}")

    print(f"\nDisplay name to key mapping:")
    for display, key in DISPLAY_NAME_TO_KEY.items():
        print(f"  {display} -> {key}")

    print(f"\nTest lookups:")
    print(f"  get_model('NetTCR22'): {get_model('NetTCR22')}")
    print(f"  get_model_display_name('NetTCR22'): {get_model_display_name('NetTCR22')}")
    print(f"  get_model_key_from_display('NetTCR-2.2'): {get_model_key_from_display('NetTCR-2.2')}")

    print("\nModel registry test completed!")
