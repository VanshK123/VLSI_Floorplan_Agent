"""Async job queue with Redis backend for VLSI floorplan optimization."""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import aioredis
from pydantic import BaseModel


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class Job(BaseModel):
    """Job model for floorplan optimization."""
    id: str
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    metrics: Dict[str, Any] = {}


class JobQueue:
    """Async job queue with Redis backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 max_workers: int = 4, queue_name: str = "vlsi_jobs"):
        self.redis_url = redis_url
        self.max_workers = max_workers
        self.queue_name = queue_name
        self.redis: Optional[aioredis.Redis] = None
        
        # Job tracking
        self.active_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.workers: List[asyncio.Task] = []
        
        # Performance metrics
        self.total_jobs_processed = 0
        self.average_processing_time = 0.0
        self.jobs_per_minute = 0.0
        
        # Processing callbacks
        self.job_processor: Optional[Callable] = None
        
        logging.info(f"JobQueue initialized with max_workers={max_workers}")

    async def start(self):
        """Start the job queue."""
        # Connect to Redis
        self.redis = aioredis.from_url(self.redis_url)
        await self.redis.ping()
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start metrics collection
        asyncio.create_task(self._collect_metrics())
        
        logging.info("JobQueue started successfully")

    async def stop(self):
        """Stop the job queue."""
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        logging.info("JobQueue stopped")

    async def submit_job(self, input_data: Dict[str, Any], 
                        priority: JobPriority = JobPriority.NORMAL) -> str:
        """Submit a new job to the queue."""
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            priority=priority,
            created_at=datetime.utcnow(),
            input_data=input_data
        )
        
        # Store job in Redis
        await self.redis.hset(
            f"{self.queue_name}:jobs",
            job_id,
            job.model_dump_json()
        )
        
        # Add to priority queue
        await self.redis.zadd(
            f"{self.queue_name}:queue",
            {job_id: priority.value}
        )
        
        logging.info(f"Job {job_id} submitted with priority {priority.name}")
        return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        # Check active jobs first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check Redis
        job_data = await self.redis.hget(f"{self.queue_name}:jobs", job_id)
        if job_data:
            return Job.model_validate_json(job_data)
        
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        # Update in Redis
        await self.redis.hset(
            f"{self.queue_name}:jobs",
            job_id,
            job.model_dump_json()
        )
        
        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        # Remove from queue
        await self.redis.zrem(f"{self.queue_name}:queue", job_id)
        
        logging.info(f"Job {job_id} cancelled")
        return True

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        # Get queue length
        queue_length = await self.redis.zcard(f"{self.queue_name}:queue")
        
        # Get active jobs count
        active_count = len(self.active_jobs)
        
        # Get completed jobs count
        completed_count = len(self.completed_jobs)
        
        return {
            "queue_length": queue_length,
            "active_jobs": active_count,
            "completed_jobs": completed_count,
            "total_jobs_processed": self.total_jobs_processed,
            "average_processing_time": self.average_processing_time,
            "jobs_per_minute": self.jobs_per_minute
        }

    async def _worker(self, worker_name: str):
        """Worker task that processes jobs."""
        logging.info(f"Worker {worker_name} started")
        
        while True:
            try:
                # Get next job from queue
                job_data = await self.redis.zpopmax(f"{self.queue_name}:queue", 1)
                
                if not job_data:
                    # No jobs in queue, wait a bit
                    await asyncio.sleep(1)
                    continue
                
                job_id = job_data[0][0]
                
                # Get job details
                job_json = await self.redis.hget(f"{self.queue_name}:jobs", job_id)
                if not job_json:
                    continue
                
                job = Job.model_validate_json(job_json)
                
                # Update job status to running
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                
                # Add to active jobs
                self.active_jobs[job_id] = job
                
                # Update in Redis
                await self.redis.hset(
                    f"{self.queue_name}:jobs",
                    job_id,
                    job.model_dump_json()
                )
                
                logging.info(f"Worker {worker_name} processing job {job_id}")
                
                # Process the job
                try:
                    result = await self._process_job(job)
                    
                    # Update job with results
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.output_data = result
                    job.progress = 100.0
                    
                    # Move to completed jobs
                    self.completed_jobs[job_id] = job
                    del self.active_jobs[job_id]
                    
                    # Update metrics
                    self.total_jobs_processed += 1
                    processing_time = (job.completed_at - job.started_at).total_seconds()
                    self._update_processing_time(processing_time)
                    
                    logging.info(f"Job {job_id} completed successfully")
                    
                except Exception as e:
                    # Job failed
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.error_message = str(e)
                    
                    # Move to completed jobs
                    self.completed_jobs[job_id] = job
                    del self.active_jobs[job_id]
                    
                    logging.error(f"Job {job_id} failed: {e}")
                
                # Update job in Redis
                await self.redis.hset(
                    f"{self.queue_name}:jobs",
                    job_id,
                    job.model_dump_json()
                )
                
            except asyncio.CancelledError:
                logging.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logging.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: Job) -> Dict[str, Any]:
        """Process a job with the VLSI floorplan optimization."""
        # This would integrate with the actual optimization pipeline
        # For now, simulate processing
        
        # Simulate processing steps
        steps = [
            ("Parsing netlist", 10),
            ("Building graph", 20),
            ("GNN embedding", 30),
            ("NSGA-II optimization", 60),
            ("STA analysis", 80),
            ("Finalizing results", 100)
        ]
        
        for step_name, target_progress in steps:
            # Simulate step processing
            await asyncio.sleep(0.5)  # Simulate work
            
            # Update progress
            job.progress = target_progress
            await self.redis.hset(
                f"{self.queue_name}:jobs",
                job.id,
                job.model_dump_json()
            )
            
            logging.info(f"Job {job.id}: {step_name} ({target_progress}%)")
        
        # Return simulated results
        return {
            "pareto_front": [
                {"area": 1000, "delay": 5.2, "placement": {"x": [100, 200], "y": [150, 250]}},
                {"area": 1200, "delay": 4.8, "placement": {"x": [120, 180], "y": [140, 260]}},
                {"area": 1100, "delay": 5.0, "placement": {"x": [110, 190], "y": [145, 255]}}
            ],
            "metrics": {
                "critical_path_delay_reduction": 0.18,
                "drc_iterations_reduction": 0.45,
                "convergence_efficiency": 0.30,
                "embedding_overhead_reduction": 0.20
            },
            "processing_time": (job.completed_at - job.started_at).total_seconds() if job.completed_at else 0
        }

    def _update_processing_time(self, processing_time: float):
        """Update average processing time."""
        if self.total_jobs_processed == 1:
            self.average_processing_time = processing_time
        else:
            self.average_processing_time = (
                (self.average_processing_time * (self.total_jobs_processed - 1) + processing_time) 
                / self.total_jobs_processed
            )

    async def _collect_metrics(self):
        """Collect and update metrics periodically."""
        while True:
            try:
                # Calculate jobs per minute
                current_time = datetime.utcnow()
                recent_jobs = [
                    job for job in self.completed_jobs.values()
                    if job.completed_at and (current_time - job.completed_at) < timedelta(minutes=1)
                ]
                self.jobs_per_minute = len(recent_jobs)
                
                # Store metrics in Redis
                metrics = {
                    "total_jobs_processed": self.total_jobs_processed,
                    "average_processing_time": self.average_processing_time,
                    "jobs_per_minute": self.jobs_per_minute,
                    "active_jobs": len(self.active_jobs),
                    "completed_jobs": len(self.completed_jobs)
                }
                
                await self.redis.hset(
                    f"{self.queue_name}:metrics",
                    "current",
                    json.dumps(metrics)
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "total_jobs_processed": self.total_jobs_processed,
            "average_processing_time": self.average_processing_time,
            "jobs_per_minute": self.jobs_per_minute,
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "queue_length": await self.redis.zcard(f"{self.queue_name}:queue") if self.redis else 0
        }

    async def clear_completed_jobs(self, older_than_hours: int = 24):
        """Clear completed jobs older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        jobs_to_remove = []
        for job_id, job in self.completed_jobs.items():
            if job.completed_at and job.completed_at < cutoff_time:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
            await self.redis.hdel(f"{self.queue_name}:jobs", job_id)
        
        logging.info(f"Cleared {len(jobs_to_remove)} old completed jobs")
