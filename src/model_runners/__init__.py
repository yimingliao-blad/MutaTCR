"""
Model runners for TCRP Benchmark V4.

Each model has its own runner that handles:
- Environment setup
- Model loading
- Prediction generation
- Output formatting
"""

# Lazy imports to avoid circular import issues
__all__ = [
    "BaseModelRunner",
    "DummyModelRunner",
    "create_runner",
    "ERGORunner",
    "ERGO2Runner",
    "NetTCRRunner",
    "NetTCR22Runner",
    "TITANRunner",
    "EPACTRunner",
    "PanPepRunner",
    "SCEPTRRunner",
]
