"""Pydantic schemas for job API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority enumeration."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class JobRequest(BaseModel):
    """Request payload for creating a job."""
    def_path: Path
    lef_path: Path
    netlist_path: Path
    constraints: Dict[str, Any] = {}
    priority: JobPriority = JobPriority.NORMAL


class JobResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    job_id: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class JobCancelRequest(BaseModel):
    """Request payload for cancelling a job."""
    job_id: str


class BatchJobRequest(BaseModel):
    """Request payload for batch job submission."""
    jobs: list[JobRequest]
    priority: JobPriority = JobPriority.NORMAL


class BatchJobResponse(BaseModel):
    """Response model for batch job submission."""
    message: str
    job_ids: list[str]
    total_jobs: int


class QueueStatusResponse(BaseModel):
    """Queue status response."""
    queue_length: int
    active_jobs: int
    completed_jobs: int
    total_jobs_processed: int
    average_processing_time: float
    jobs_per_minute: float


class PerformanceTargets(BaseModel):
    """Performance targets configuration."""
    critical_path_delay_reduction: float = 0.15
    drc_iterations_reduction: float = 0.40
    convergence_efficiency: float = 0.30
    embedding_overhead_reduction: float = 0.20
    max_processing_time: float = 3600.0
    max_memory_usage: int = 8 * 1024 * 1024 * 1024  # 8GB


class OptimizationConstraints(BaseModel):
    """Optimization constraints."""
    max_area: Optional[float] = None
    max_delay: Optional[float] = None
    target_frequency: Optional[float] = None
    power_budget: Optional[float] = None
    temperature_constraint: Optional[float] = None


class GNNConfig(BaseModel):
    """GNN configuration."""
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_cells: int = 1000000
    batch_size: int = 32
    learning_rate: float = 0.001


class EvolutionaryConfig(BaseModel):
    """Evolutionary algorithm configuration."""
    pop_size: int = 50
    num_gens: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    batch_size: int = 16
    sta_timeout: int = 30
    adaptive_mutation: bool = True
    lyapunov_stability: bool = True


class STAConfig(BaseModel):
    """STA configuration."""
    binary_path: str = "/usr/local/bin/sta_tool"
    timeout: int = 30
    batch_size: int = 16
    max_concurrent: int = 4


class SystemConfig(BaseModel):
    """System configuration."""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    redis_url: str = "redis://localhost:6379"
    max_workers: int = 4
    log_level: str = "INFO"


class ConfigurationResponse(BaseModel):
    """Configuration response."""
    gnn: GNNConfig
    evolutionary: EvolutionaryConfig
    sta: STAConfig
    system: SystemConfig
    targets: PerformanceTargets
