"""Result retrieval endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..schemas.job_schema import JobResponse
from ...persistence.database import SessionLocal
from ...persistence import models

router = APIRouter(prefix="/results", tags=["results"])


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{job_id}")
def get_result(job_id: int, db: Session = Depends(get_db)):
    result = db.query(models.Result).filter(models.Result.job_id == job_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
