"""Result retrieval endpoints with async job queue integration."""
from __future__ import annotations

from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...queue.job_queue import JobQueue, Job
from ...utils.logger import get_logger

router = APIRouter(prefix="/results", tags=["results"])

logger = get_logger("results")


class ParetoFrontSolution(BaseModel):
    """Individual solution in Pareto front."""
    area: float
    delay: float
    placement: Dict[str, List[float]]
    fitness: List[float]


class OptimizationResult(BaseModel):
    """Complete optimization result."""
    job_id: str
    status: str
    pareto_front: List[ParetoFrontSolution]
    metrics: Dict[str, float]
    processing_time: float
    num_generations: int
    final_population_size: int
    convergence_metrics: Dict[str, Any]


class ResultSummary(BaseModel):
    """Summary of optimization results."""
    job_id: str
    best_area: float
    best_delay: float
    pareto_front_size: int
    critical_path_reduction: float
    drc_iterations_reduction: float
    convergence_efficiency: float


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    # This will be set by the app startup
    from ..app import job_queue
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    return job_queue


@router.get("/{job_id}", response_model=OptimizationResult)
async def get_result(job_id: str, 
                    queue: JobQueue = Depends(get_job_queue)) -> OptimizationResult:
    """Get detailed optimization results for a job."""
    try:
        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status.value not in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Job not yet completed")
        
        if job.status.value == "failed":
            raise HTTPException(status_code=500, detail=f"Job failed: {job.error_message}")
        
        # Extract results from job output
        output_data = job.output_data or {}
        pareto_front = output_data.get("pareto_front", [])
        metrics = output_data.get("metrics", {})
        
        # Convert to response format
        pareto_solutions = []
        for solution in pareto_front:
            pareto_solutions.append(ParetoFrontSolution(
                area=solution.get("area", 0.0),
                delay=solution.get("delay", 0.0),
                placement=solution.get("placement", {}),
                fitness=solution.get("fitness", [])
            ))
        
        # Calculate processing time
        processing_time = 0.0
        if job.completed_at and job.started_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
        
        return OptimizationResult(
            job_id=job.id,
            status=job.status.value,
            pareto_front=pareto_solutions,
            metrics=metrics,
            processing_time=processing_time,
            num_generations=metrics.get("num_generations", 0),
            final_population_size=metrics.get("final_population_size", 0),
            convergence_metrics=metrics.get("convergence_metrics", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_result_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")


@router.get("/{job_id}/summary", response_model=ResultSummary)
async def get_result_summary(job_id: str, 
                           queue: JobQueue = Depends(get_job_queue)) -> ResultSummary:
    """Get summary of optimization results."""
    try:
        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status.value != "completed":
            raise HTTPException(status_code=400, detail="Job not yet completed")
        
        # Extract summary from job output
        output_data = job.output_data or {}
        pareto_front = output_data.get("pareto_front", [])
        metrics = output_data.get("metrics", {})
        
        if not pareto_front:
            raise HTTPException(status_code=404, detail="No results available")
        
        # Calculate summary metrics
        areas = [sol.get("area", 0.0) for sol in pareto_front]
        delays = [sol.get("delay", 0.0) for sol in pareto_front]
        
        best_area = min(areas) if areas else 0.0
        best_delay = min(delays) if delays else 0.0
        
        return ResultSummary(
            job_id=job.id,
            best_area=best_area,
            best_delay=best_delay,
            pareto_front_size=len(pareto_front),
            critical_path_reduction=metrics.get("critical_path_delay_reduction", 0.0),
            drc_iterations_reduction=metrics.get("drc_iterations_reduction", 0.0),
            convergence_efficiency=metrics.get("convergence_efficiency", 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_summary_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get("/{job_id}/pareto")
async def get_pareto_front(job_id: str, 
                          queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """Get Pareto front solutions."""
    try:
        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status.value != "completed":
            raise HTTPException(status_code=400, detail="Job not yet completed")
        
        output_data = job.output_data or {}
        pareto_front = output_data.get("pareto_front", [])
        
        return {
            "job_id": job_id,
            "pareto_front": pareto_front,
            "num_solutions": len(pareto_front)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_pareto_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get Pareto front: {str(e)}")


@router.get("/{job_id}/metrics")
async def get_job_metrics(job_id: str, 
                         queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """Get detailed metrics for a job."""
    try:
        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        output_data = job.output_data or {}
        metrics = output_data.get("metrics", {})
        
        # Add job-level metrics
        job_metrics = {
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time": 0.0
        }
        
        if job.completed_at and job.started_at:
            job_metrics["processing_time"] = (job.completed_at - job.started_at).total_seconds()
        
        return {
            "job_id": job_id,
            "job_metrics": job_metrics,
            "optimization_metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_metrics_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/")
async def list_completed_results(queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """List all completed jobs with result summaries."""
    try:
        completed_jobs = []
        
        for job_id, job in queue.completed_jobs.items():
            if job.status.value == "completed":
                output_data = job.output_data or {}
                pareto_front = output_data.get("pareto_front", [])
                metrics = output_data.get("metrics", {})
                
                # Calculate summary
                areas = [sol.get("area", 0.0) for sol in pareto_front]
                delays = [sol.get("delay", 0.0) for sol in pareto_front]
                
                summary = {
                    "job_id": job_id,
                    "best_area": min(areas) if areas else 0.0,
                    "best_delay": min(delays) if delays else 0.0,
                    "pareto_front_size": len(pareto_front),
                    "critical_path_reduction": metrics.get("critical_path_delay_reduction", 0.0),
                    "drc_iterations_reduction": metrics.get("drc_iterations_reduction", 0.0),
                    "convergence_efficiency": metrics.get("convergence_efficiency", 0.0),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None
                }
                completed_jobs.append(summary)
        
        # Sort by completion time (most recent first)
        completed_jobs.sort(key=lambda x: x["completed_at"] or "", reverse=True)
        
        return {
            "total_completed": len(completed_jobs),
            "results": completed_jobs[:50]  # Return last 50 results
        }
        
    except Exception as e:
        logger.error("list_results_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")


@router.get("/performance/summary")
async def get_performance_summary(queue: JobQueue = Depends(get_job_queue)) -> Dict[str, Any]:
    """Get performance summary across all completed jobs."""
    try:
        completed_jobs = [job for job in queue.completed_jobs.values() 
                         if job.status.value == "completed"]
        
        if not completed_jobs:
            return {"message": "No completed jobs found"}
        
        # Aggregate metrics
        critical_path_reductions = []
        drc_reductions = []
        convergence_efficiencies = []
        processing_times = []
        
        for job in completed_jobs:
            output_data = job.output_data or {}
            metrics = output_data.get("metrics", {})
            
            critical_path_reductions.append(metrics.get("critical_path_delay_reduction", 0.0))
            drc_reductions.append(metrics.get("drc_iterations_reduction", 0.0))
            convergence_efficiencies.append(metrics.get("convergence_efficiency", 0.0))
            
            if job.completed_at and job.started_at:
                processing_times.append((job.completed_at - job.started_at).total_seconds())
        
        # Calculate averages
        avg_critical_path_reduction = sum(critical_path_reductions) / len(critical_path_reductions)
        avg_drc_reduction = sum(drc_reductions) / len(drc_reductions)
        avg_convergence_efficiency = sum(convergence_efficiencies) / len(convergence_efficiencies)
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            "total_jobs": len(completed_jobs),
            "average_metrics": {
                "critical_path_reduction": avg_critical_path_reduction,
                "drc_iterations_reduction": avg_drc_reduction,
                "convergence_efficiency": avg_convergence_efficiency,
                "processing_time": avg_processing_time
            },
            "targets_achieved": {
                "critical_path_reduction": avg_critical_path_reduction >= 0.15,
                "drc_iterations_reduction": avg_drc_reduction >= 0.40,
                "convergence_efficiency": avg_convergence_efficiency >= 0.30
            }
        }
        
    except Exception as e:
        logger.error("performance_summary_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check for results service."""
    return {"status": "healthy", "service": "results"}
