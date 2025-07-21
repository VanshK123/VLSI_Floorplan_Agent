"""Parser for STA reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class STAParser:
    """Parse STA output reports."""

    def parse(self, report_path: Path) -> Dict[str, Any]:
        """Return slack and timing metrics from report."""
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        raise NotImplementedError
