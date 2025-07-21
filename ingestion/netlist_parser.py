"""Netlist parser for extracting connectivity."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class NetlistParser:
    """Parser for gate-level netlists."""

    def parse(self, netlist_path: Path) -> Dict[str, Any]:
        """Parse a netlist file and return a connectivity list.

        Parameters
        ----------
        netlist_path: Path
            Path to the netlist file.
        Returns
        -------
        Dict[str, Any]
            Connectivity information.
        """
        if not netlist_path.exists():
            raise FileNotFoundError(netlist_path)
        raise NotImplementedError
