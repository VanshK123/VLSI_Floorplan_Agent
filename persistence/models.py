"""Database ORM models."""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    def_path = Column(String, nullable=False)
    lef_path = Column(String, nullable=False)
    netlist_path = Column(String, nullable=False)
    status = Column(String, default="pending")

    result = relationship("Result", back_populates="job", uselist=False)


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    pareto_front = Column(JSON)
    metrics = Column(JSON)

    job = relationship("Job", back_populates="result")
