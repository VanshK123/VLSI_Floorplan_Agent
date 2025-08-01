"""FastAPI application entry point with async job processing."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routes import jobs, results
from ..queue.job_queue import JobQueue
from ..utils.config import get_config
from ..utils.logger import setup_logging


# Global job queue
job_queue: JobQueue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global job_queue
    
    # Startup
    logging.info("Starting VLSI Floorplan Agent API...")
    
    # Initialize job queue
    config = get_config()
    job_queue = JobQueue(
        redis_url=config.get('redis_url', 'redis://localhost:6379'),
        max_workers=config.get('max_workers', 4)
    )
    await job_queue.start()
    
    logging.info("API startup complete")
    
    yield
    
    # Shutdown
    logging.info("Shutting down VLSI Floorplan Agent API...")
    if job_queue:
        await job_queue.stop()
    logging.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    # Setup logging
    setup_logging()
    
    app = FastAPI(
        title="VLSI Floorplan Agent",
        description="Self-Optimizing VLSI Floorplan & Timing Agent with GNN and NSGA-II",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(jobs.router, prefix="/api/v1")
    app.include_router(results.router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "VLSI Floorplan Agent",
            "version": "1.0.0"
        }
    
    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics."""
        if job_queue:
            queue_metrics = await job_queue.get_metrics()
        else:
            queue_metrics = {}
        
        return {
            "queue_metrics": queue_metrics,
            "system_metrics": {
                "active_jobs": len(job_queue.active_jobs) if job_queue else 0,
                "completed_jobs": len(job_queue.completed_jobs) if job_queue else 0
            }
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logging.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
