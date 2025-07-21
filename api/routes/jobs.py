"""Job submission endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..schemas.job_schema import JobRequest, JobResponse
from ...persistence.database import SessionLocal
from ...persistence import models
from ...queue.job_queue import process_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=JobResponse)
def create_job(request: JobRequest, db: Session = Depends(get_db)) -> JobResponse:
    job = models.Job(
        def_path=str(request.def_path),
        lef_path=str(request.lef_path),
        netlist_path=str(request.netlist_path),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    process_job.delay(job.id)
    return JobResponse(job_id=job.id, status=job.status)
