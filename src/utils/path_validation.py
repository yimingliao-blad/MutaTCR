"""
Path Validation Utilities for TCRP Benchmark V2.

This module provides utilities to ensure all scripts use relative paths
instead of hardcoded absolute paths, making the project portable across
different systems.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory using relative path resolution.

    This function determines the project root by navigating up from the
    current file's location, ensuring portability across systems.

    Returns:
        Path to project root (tcrp-benchmark-v2 directory)
    """
    # Try to find project root from current file location
    current_file = Path(__file__).resolve()

    # Navigate up: path_validation.py -> utils -> src -> tcrp-benchmark-v2
    project_root = current_file.parent.parent.parent

    # Validate we found the correct directory
    if not (project_root / "src").exists():
        raise RuntimeError(
            f"Could not find project root. Expected 'src' directory in {project_root}"
        )

    return project_root


def get_project_root_from_script(script_file: str) -> Path:
    """
    Get project root from a script file path.

    Args:
        script_file: The __file__ variable from the calling script

    Returns:
        Path to project root
    """
    script_path = Path(script_file).resolve()

    # Navigate up to find project root (look for src directory)
    current = script_path.parent
    for _ in range(10):  # Max 10 levels up
        if (current / "src").exists() and (current / "config").exists():
            return current
        current = current.parent

    raise RuntimeError(
        f"Could not find project root from {script_file}. "
        "Ensure script is inside the tcrp-benchmark-v2 directory."
    )


def validate_relative_path(path: Path, base_dir: Path) -> bool:
    """
    Validate that a path is relative to the project directory.

    Args:
        path: Path to validate
        base_dir: Base directory (project root)

    Returns:
        True if path is within base_dir, False otherwise
    """
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def ensure_relative_path(path: Path, base_dir: Path, description: str = "path") -> Path:
    """
    Ensure a path is relative to the project directory.

    Args:
        path: Path to check
        base_dir: Base directory (project root)
        description: Description of the path for error messages

    Returns:
        The validated path

    Raises:
        ValueError: If path is outside project directory
    """
    if not validate_relative_path(path, base_dir):
        raise ValueError(
            f"Invalid {description}: {path} is outside project directory {base_dir}. "
            "All paths must be relative to the project root."
        )
    return path


def setup_project_paths(script_file: str) -> Path:
    """
    Standard setup for script paths.

    Call this at the beginning of each script to ensure proper path setup.

    Args:
        script_file: The __file__ variable from the calling script

    Returns:
        Path to project root

    Example:
        # At the top of each script:
        from src.utils.path_validation import setup_project_paths
        PROJECT_ROOT = setup_project_paths(__file__)
    """
    project_root = get_project_root_from_script(script_file)

    # Add project root to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Validate current working directory is within project
    cwd = Path.cwd()
    if not validate_relative_path(cwd, project_root):
        # Change to project root if outside project
        os.chdir(project_root)

    return project_root


def check_no_absolute_paths(*paths: Path, project_root: Optional[Path] = None) -> None:
    """
    Check that no paths are absolute paths outside the project.

    Args:
        *paths: Paths to check
        project_root: Project root directory (auto-detected if not provided)

    Raises:
        ValueError: If any path is an absolute path outside the project
    """
    if project_root is None:
        project_root = get_project_root()

    for path in paths:
        if path.is_absolute():
            if not validate_relative_path(path, project_root):
                raise ValueError(
                    f"Absolute path detected outside project: {path}. "
                    f"Use relative paths from project root: {project_root}"
                )


def get_relative_path(path: Path, project_root: Optional[Path] = None) -> Path:
    """
    Convert an absolute path to a relative path from project root.

    Args:
        path: Path to convert
        project_root: Project root directory (auto-detected if not provided)

    Returns:
        Relative path from project root
    """
    if project_root is None:
        project_root = get_project_root()

    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        # Path is outside project - return as-is but warn
        return path


# Convenience function for scripts
def init_script(script_file: str, check_cwd: bool = True) -> Path:
    """
    Initialize a script with proper path validation.

    This is the main entry point for scripts to ensure all paths are relative.

    Args:
        script_file: The __file__ variable from the calling script
        check_cwd: Whether to check/change current working directory

    Returns:
        Path to project root

    Example:
        #!/usr/bin/env python
        from pathlib import Path
        import sys

        # Initialize paths
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.utils.path_validation import init_script
        PROJECT_ROOT = init_script(__file__)

        # Now all paths should be relative to PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
    """
    project_root = get_project_root_from_script(script_file)

    # Add to Python path
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    if check_cwd:
        # Ensure we're working from project root or a subdirectory
        cwd = Path.cwd()
        if not validate_relative_path(cwd, project_root):
            os.chdir(project_root)

    return project_root


if __name__ == "__main__":
    # Test the module
    print("Testing path validation utilities...")

    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # Test path validation
    test_paths = [
        project_root / "data",
        project_root / "src" / "scripts",
        Path("/tmp/outside"),
    ]

    for path in test_paths:
        is_valid = validate_relative_path(path, project_root)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {path}: {'valid' if is_valid else 'INVALID (outside project)'}")

    print("\nPath validation test complete!")
