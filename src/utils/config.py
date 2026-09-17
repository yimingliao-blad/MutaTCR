"""
Configuration management for TCRP Benchmark V4.

Loads configuration from YAML files and environment variables.
Environment variables take precedence over YAML settings.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for TCRP Benchmark V4."""

    _instance: Optional['Config'] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[Path] = None):
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to benchmark_config.yaml. If None, auto-detected.
        """
        if self._initialized and config_path is None:
            return

        self._project_root = self._find_project_root()

        if config_path is None:
            config_path = self._project_root / "config" / "benchmark_config.yaml"

        self.config_path = config_path
        self._config = self._load_config()
        self._display_names = self._load_display_names()
        self._apply_env_overrides()
        self._initialized = True

    def _load_display_names(self) -> Dict[str, Any]:
        """Load display names configuration from JSON file."""
        display_names_path = self._project_root / "config" / "display_names.json"
        if display_names_path.exists():
            with open(display_names_path, 'r') as f:
                return json.load(f)
        return {}

    def _find_project_root(self) -> Path:
        """Find project root by looking for config directory."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "config" / "benchmark_config.yaml").exists():
                return parent
        # Fallback to parent of src
        return Path(__file__).resolve().parent.parent.parent

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_defaults()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {self.config_path}")
        return config

    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'project': {'name': 'TCRP Benchmark V4', 'version': '4.0.0'},
            'paths': {
                'conda': 'tools/miniconda3',
                'models': 'models',
                'data': {
                    'raw': 'data/raw',
                    'unified': 'data/unified',
                    'deduplicated': 'data/deduplicated'
                },
                'output': 'output',
                'logs': 'logs'
            },
            'gpu': {
                'auto_detect': True,
                'default_device': 0,
                'fallback_to_cpu': True
            },
            'models': {},
            'deduplication': {
                'mode': 'exact_match',
                'exact_match': {'columns': ['Peptide', 'CDR3b']},
                'fuzzy_match': {'max_distance': 3, 'columns': ['CDR3b']}
            },
            'evaluation': {'partial_auc_max_fpr': 0.1},
            'visualization': {
                'partial_auc_fpr': 0.1,
                'figure_dpi': 300,
                'figure_format': 'png',
                'heatmap': {'colormap': 'RdBu_r', 'vmin': 0.0, 'vmax': 1.0}
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'datasets': {}
        }

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'TCRP_CONDA_PATH': ('paths', 'conda'),
            'TCRP_MODELS_PATH': ('paths', 'models'),
            'TCRP_OUTPUT_PATH': ('paths', 'output'),
            'TCRP_GPU_DEVICE': ('gpu', 'default_device'),
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config_path, value)
                logger.info(f"Override from env: {env_var} = {value}")

    def _set_nested(self, path: tuple, value: Any):
        """Set a nested dictionary value."""
        d = self._config
        for key in path[:-1]:
            d = d.setdefault(key, {})
        # Convert string to int for numeric settings
        if path[-1] in ('default_device', 'batch_size', 'timeout'):
            value = int(value)
        d[path[-1]] = value

    def _get_nested(self, d: dict, path: tuple, default: Any = None) -> Any:
        """Get a nested dictionary value."""
        for key in path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
        self._apply_env_overrides()

    # ========== PATH RESOLUTION ==========

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root

    def get_path(self, *path) -> Path:
        """Get a path relative to project root."""
        relative_path = self._get_nested(self._config, ('paths',) + path)
        if relative_path:
            return self._project_root / relative_path
        return None

    def get_conda_path(self) -> Path:
        """Get conda installation path."""
        env_path = os.environ.get('TCRP_CONDA_PATH')
        if env_path:
            return Path(env_path)
        return self.get_path('conda')

    def get_models_dir(self) -> Path:
        """Get models directory."""
        return self.get_path('models')

    def get_output_dir(self) -> Path:
        """Get output directory."""
        return self.get_path('output')

    def get_data_dir(self, data_type: str = 'unified') -> Path:
        """Get data directory by type."""
        return self._project_root / self._get_nested(
            self._config, ('paths', 'data', data_type), f'data/{data_type}'
        )

    def get_logs_dir(self) -> Path:
        """Get logs directory."""
        return self.get_path('logs')

    # ========== GPU SETTINGS ==========

    def get_gpu_device(self) -> int:
        """Get GPU device to use."""
        return self._get_nested(self._config, ('gpu', 'default_device'), 0)

    def is_gpu_auto_detect(self) -> bool:
        """Check if GPU auto-detection is enabled."""
        return self._get_nested(self._config, ('gpu', 'auto_detect'), True)

    def fallback_to_cpu(self) -> bool:
        """Check if CPU fallback is enabled."""
        return self._get_nested(self._config, ('gpu', 'fallback_to_cpu'), True)

    # ========== MODEL SETTINGS ==========

    def get_model_config(self, model: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model: Model key (e.g., "ERGO", "ERGO2")

        Returns:
            Model configuration dictionary
        """
        return self._config.get("models", {}).get(model, {})

    def is_model_enabled(self, model: str) -> bool:
        """Check if a model is enabled."""
        return self.get_model_config(model).get('enabled', True)

    def get_model_timeout(self, model: str) -> int:
        """Get timeout for a model in seconds."""
        return self.get_model_config(model).get('timeout', 1800)

    def get_model_env(self, model: str) -> str:
        """Get conda environment name for a model."""
        return self.get_model_config(model).get('env', f"{model.lower()}-benchmark")

    def get_model_batch_size(self, model: str) -> int:
        """Get batch size for a model."""
        return self.get_model_config(model).get('batch_size', 32)

    def get_model_training_data_path(self, model: str) -> Optional[Path]:
        """Get training data path for a model."""
        path = self.get_model_config(model).get('training_data')
        if path:
            return self._project_root / path
        return None

    def get_model_display_name(self, model: str) -> str:
        """Get display name for a model."""
        return self.get_model_config(model).get("display_name", model)

    def get_model_folder_name(self, model: str) -> str:
        """Get folder name for a model."""
        return self.get_model_config(model).get("folder_name", model)

    def get_model_env_name(self, model: str) -> str:
        """Get conda environment name for a model."""
        return self.get_model_config(model).get("env_name", f"{model.lower()}-benchmark")

    def get_model_input_columns(self, model: str) -> List[str]:
        """Get required input columns for a model."""
        return self.get_model_config(model).get("input_columns", [])

    @property
    def model_order(self) -> List[str]:
        """Get model display order for plots."""
        return self._config.get("model_order", [])

    @property
    def all_models(self) -> List[str]:
        """Get list of all model keys."""
        return list(self._config.get("models", {}).keys())

    # ========== PANPEP SETTINGS ==========

    def get_panpep_best_of_modes(self) -> bool:
        """Check if PanPep best-of-modes is enabled."""
        return self.get_model_config('PanPep').get('best_of_modes', False)

    def get_panpep_modes_for_base(self, base_mode: str) -> List[str]:
        """Get modes to try for a given base mode."""
        panpep_config = self.get_model_config('PanPep')
        if base_mode == 'majority':
            return panpep_config.get('modes_for_majority', ['majority', 'few-shot', 'zero-shot'])
        elif base_mode == 'few-shot':
            return panpep_config.get('modes_for_fewshot', ['few-shot', 'zero-shot'])
        else:
            return panpep_config.get('modes_for_zeroshot', ['zero-shot'])

    # ========== DEDUPLICATION SETTINGS ==========

    @property
    def deduplication_mode(self) -> str:
        """Get deduplication mode."""
        return self._config.get("deduplication", {}).get("mode", "exact_match")

    @property
    def exact_match_columns(self) -> List[str]:
        """Columns used for exact match deduplication."""
        return self._config.get("deduplication", {}).get("exact_match", {}).get("columns", ["Peptide", "CDR3b"])

    @property
    def fuzzy_max_distance(self) -> int:
        """Maximum Levenshtein distance for fuzzy matching."""
        return self._config.get("deduplication", {}).get("fuzzy_match", {}).get("max_distance", 3)

    @property
    def fuzzy_columns(self) -> List[str]:
        """Columns to apply fuzzy matching on."""
        return self._config.get("deduplication", {}).get("fuzzy_match", {}).get("columns", ["CDR3b"])

    # ========== DATASET SETTINGS ==========

    def get_dataset_config(self, dataset: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.

        Args:
            dataset: Dataset key ("tettcr", "immrep23", or "fingerprinting")

        Returns:
            Dataset configuration dictionary
        """
        return self._config.get("datasets", {}).get(dataset, {})

    def get_dataset_epitope_groups(self, dataset: str) -> List[str]:
        """Get epitope groups for a dataset."""
        return self.get_dataset_config(dataset).get("epitope_groups", [])

    def get_dataset_evaluation_groups(self, dataset: str) -> List[str]:
        """Get evaluation groups for a dataset."""
        return self.get_dataset_config(dataset).get("evaluation_groups", [])

    def get_dataset_extra_columns(self, dataset: str) -> List[str]:
        """Get extra columns for a dataset."""
        return self.get_dataset_config(dataset).get("extra_columns", [])

    def get_tcr_groups(self) -> List[str]:
        """Get fingerprinting TCR groups."""
        return self._config.get("datasets", {}).get("fingerprinting", {}).get("tcr_groups", [])

    @property
    def reference_peptide(self) -> str:
        """Get fingerprinting reference peptide."""
        return self._config.get("datasets", {}).get("fingerprinting", {}).get("reference_peptide", "YLQPRTFLL")

    # ========== VISUALIZATION SETTINGS ==========

    @property
    def partial_auc_fpr(self) -> float:
        """Get partial AUC false positive rate threshold."""
        return self._config.get("visualization", {}).get("partial_auc_fpr", 0.1)

    @property
    def scatter_alpha(self) -> float:
        """Get scatter plot alpha value."""
        return self._config.get("visualization", {}).get("scatter_alpha", 0.5)

    @property
    def figure_dpi(self) -> int:
        """Get figure DPI setting."""
        return self._config.get("visualization", {}).get("figure_dpi", 300)

    @property
    def figure_format(self) -> str:
        """Get figure format."""
        return self._config.get("visualization", {}).get("figure_format", "png")

    def get_color(self, group: str) -> str:
        """Get color for an epitope group."""
        colors = self._config.get("visualization", {}).get("colors", {})
        return colors.get(group, "#333333")

    @property
    def heatmap_colormap(self) -> str:
        """Get heatmap colormap."""
        return self._config.get("visualization", {}).get("heatmap", {}).get("colormap", "RdBu_r")

    @property
    def heatmap_vmin(self) -> float:
        """Get heatmap minimum value."""
        return self._config.get("visualization", {}).get("heatmap", {}).get("vmin", 0.0)

    @property
    def heatmap_vmax(self) -> float:
        """Get heatmap maximum value."""
        return self._config.get("visualization", {}).get("heatmap", {}).get("vmax", 1.0)

    @property
    def amino_acids(self) -> List[str]:
        """Get ordered list of amino acids for heatmaps."""
        return self._config.get("visualization", {}).get("heatmap", {}).get("amino_acids", list("ARNDCEQGHILKMFPSTWYV"))

    @property
    def peptide_positions(self) -> List[int]:
        """Get peptide positions for heatmaps."""
        return self._config.get("visualization", {}).get("heatmap", {}).get("positions", list(range(1, 10)))

    # ========== DISPLAY NAMES ==========

    def get_dataset_display_name(self, dataset: str) -> str:
        """Get display name for a dataset."""
        return self._display_names.get("datasets", {}).get(dataset, dataset)

    def get_epitope_group_display_name(self, dataset: str, group: str) -> str:
        """Get display name for an epitope group."""
        return self._display_names.get("epitope_groups", {}).get(dataset, {}).get(group, group)

    def get_group_color(self, dataset: str, group: str) -> str:
        """Get color for a dataset group."""
        return self._display_names.get("colors", {}).get(dataset, {}).get(group, "#333333")

    def get_display_names_config(self) -> Dict[str, Any]:
        """Get full display names configuration."""
        return self._display_names

    # ========== EVALUATION SETTINGS ==========

    @property
    def primary_metric(self) -> str:
        """Get primary evaluation metric."""
        return self._config.get("evaluation", {}).get("primary_metric", "partial_auc")

    @property
    def partial_auc_max_fpr(self) -> float:
        """Get partial AUC max FPR."""
        return self._config.get("evaluation", {}).get("partial_auc_max_fpr", 0.1)

    @property
    def metrics_list(self) -> List[str]:
        """Get list of metrics to compute."""
        return self._config.get("evaluation", {}).get("metrics", ["auc", "partial_auc"])

    # ========== LOGGING SETTINGS ==========

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self._config.get("logging", {}).get("level", "INFO")

    @property
    def log_format(self) -> str:
        """Get logging format."""
        return self._config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @property
    def log_dir(self) -> str:
        """Get logging directory."""
        return self._config.get("logging", {}).get("log_dir", "logs")


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config file (optional)

    Returns:
        Config instance
    """
    global _config

    if _config is None or config_path is not None:
        _config = Config(config_path)

    return _config


def reload_config():
    """Reload configuration from file."""
    global _config
    if _config is not None:
        _config.reload()


if __name__ == "__main__":
    # Test configuration loader
    print("Testing Config loader...")

    config = Config()
    print(f"Config loaded from: {config.config_path}")
    print(f"Project root: {config.project_root}")

    print(f"\nPath settings:")
    print(f"  Conda: {config.get_conda_path()}")
    print(f"  Models: {config.get_models_dir()}")
    print(f"  Output: {config.get_output_dir()}")
    print(f"  Data (unified): {config.get_data_dir('unified')}")

    print(f"\nGPU settings:")
    print(f"  Device: {config.get_gpu_device()}")
    print(f"  Auto-detect: {config.is_gpu_auto_detect()}")

    print(f"\nModel settings:")
    for model in config.all_models:
        print(f"  {model}:")
        print(f"    Enabled: {config.is_model_enabled(model)}")
        print(f"    Timeout: {config.get_model_timeout(model)}")
        print(f"    Env: {config.get_model_env(model)}")

    print(f"\nPanPep settings:")
    print(f"  Best-of-modes: {config.get_panpep_best_of_modes()}")
    print(f"  Modes for majority: {config.get_panpep_modes_for_base('majority')}")
    print(f"  Modes for few-shot: {config.get_panpep_modes_for_base('few-shot')}")

    print("\nConfig loader test completed!")
