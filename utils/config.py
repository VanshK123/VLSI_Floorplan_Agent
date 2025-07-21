"""Configuration loader for the project."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_configs(config_paths: Dict[str, Path]) -> Dict[str, Any]:
    """Load multiple YAML config files into a single dictionary."""
    configs: Dict[str, Any] = {}
    for name, path in config_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            configs[name] = yaml.safe_load(f)
    return configs
