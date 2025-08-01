"""Job submission endpoints with async job queue integration."""
from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from ..schemas.job_schema import JobRequest, JobResponse, JobStatus
from ...queue.job_queue import JobQueue, JobPriority
from ...utils.config import get_config
from ...utils.logger import get_logger

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Global job queue instance (will be set by app startup)
job_queue: Optional[JobQueue] = None

logger = get_logger("jobs")


class JobSubmitRequest(BaseModel):
    """Job submission request model."""
    def_path: str
    lef_path: str
    netlist_path: str
    constraints: Dict[str, Any] = {}
    priority: str = "NORMAL"  # LOW, NORMAL, HIGH, URGENT


class JobStatusResponse(BaseModel):
    """Job status response model."""
    job_id: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class JobCancelRequest(BaseModel):
    """Job cancellation request model."""
    job_id: str


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    return job_queue


@router.post("/", response_model=JobResponse)
async def create_job(request: JobSubmitRequest, 
                    background_tasks: BackgroundTasks,
                    queue: JobQueue = Depends(get_job_queue)) -> JobResponse:
    """Submit a new optimization job."""
    try:
        # Validate input files exist
        import os
        for file_path in [request.def_path, request.lef_path, request.netlist_path]:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        
        # Convert priority string to enum
        priority_map = {
            "LOW": JobPriority.LOW,
            "NORMAL": JobPriority.NORMAL,
            "HIGH": JobPriority.HIGH,
            "URGENT": JobPriority.URGENT
        }
        priority = priority_map.get(request.priority.upper(), JobPriority.NORMAL)
        
        # Prepare job input data
        input_data = {
            "def_path": request.def_path,
            "lef_path": request.lef_path,
            "netlist_path": request.netlist_path,
            "constraints": request.constraints,
            "config": get_config().get_section("evolutionary")
        }
        
        # Submit job to queue
        job_id = await queue.submit_job(input_data, priority)
        
        logger.info("job_submitted", job_id=job_id, priority=priority.name)
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Job submitted successfully"
        )
        
    except Exception as e:
        logger.error("job_submission_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, 
                        queue: JobQueue = Depends(get_job_queue)) -> JobStatusResponse:
    """Get job status and progress."""
    try:
        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job.id,
            status=job.status.value,
            progress=job.progress,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error_message=job.error_message,
            metrics=job.metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("job_status_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.delete("/{job_id}")
async def cancel_job(job_id: str, 
                    queue: JobQueue = Depends(get_job_queue)) -> Dict[str, str]:
    """Cancel a running job."""
    try:
        success = await queue.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        logger.info("job_cancelled", job_id=job_id)
        
        return {"message": "Job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("job_cancellation_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/")
async def list_jobs(queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """Get list of all jobs with queue status."""
    try:
        # Get queue status
        queue_status = await queue.get_queue_status()
        
        # Get recent jobs (last 50)
        recent_jobs = []
        for job_id, job in list(queue.completed_jobs.items())[-50:]:
            recent_jobs.append({
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            })
        
        return {
            "queue_status": queue_status,
            "recent_jobs": recent_jobs,
            "active_jobs": len(queue.active_jobs)
        }
        
    except Exception as e:
        logger.error("list_jobs_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/metrics/summary")
async def get_job_metrics(queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """Get job processing metrics."""
    try:
        metrics = await queue.get_metrics()
        
        # Add performance targets
        config = get_config()
        targets = config.get_performance_targets()
        
        return {
            "queue_metrics": metrics,
            "performance_targets": targets,
            "system_info": {
                "max_workers": queue.max_workers,
                "redis_url": queue.redis_url
            }
        }
        
    except Exception as e:
        logger.error("metrics_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/batch")
async def submit_batch_jobs(requests: list[JobSubmitRequest],
                           queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """Submit multiple jobs in batch."""
    try:
        job_ids = []
        for request in requests:
            # Validate input files exist
            import os
            for file_path in [request.def_path, request.lef_path, request.netlist_path]:
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
            
            # Prepare job input data
            input_data = {
                "def_path": request.def_path,
                "lef_path": request.lef_path,
                "netlist_path": request.netlist_path,
                "constraints": request.constraints,
                "config": get_config().get_section("evolutionary")
            }
            
            # Submit job
            job_id = await queue.submit_job(input_data, JobPriority.NORMAL)
            job_ids.append(job_id)
        
        logger.info("batch_jobs_submitted", count=len(job_ids), job_ids=job_ids)
        
        return {
            "message": f"Successfully submitted {len(job_ids)} jobs",
            "job_ids": job_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("batch_submission_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check for jobs service."""
    return {"status": "healthy", "service": "jobs"}
