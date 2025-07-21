"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import jobs, results


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(title="VLSI Floorplan Agent")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(jobs.router)
    app.include_router(results.router)
    return app


app = create_app()
