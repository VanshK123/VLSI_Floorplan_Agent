"""SQLAlchemy model mirror for API usage."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ...persistence import models


def get_job(db: Session, job_id: int) -> models.Job:
    """Retrieve a Job from the database."""
    return db.query(models.Job).filter(models.Job.id == job_id).first()
