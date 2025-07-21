"""Pydantic schemas for job API."""
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class JobRequest(BaseModel):
    """Request payload for creating a job."""

    def_path: Path
    lef_path: Path
    netlist_path: Path


class JobResponse(BaseModel):
    """Response model for job status."""

    job_id: int
    status: str
