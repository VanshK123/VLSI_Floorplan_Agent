"""Runs external STA tool."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict


class STARunner:
    """Interface to invoke an STA binary."""

    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path

    def run(self, design_dir: Path) -> Path:
        """Execute STA and return path to report."""
        if not design_dir.exists():
            raise FileNotFoundError(design_dir)
        raise NotImplementedError
