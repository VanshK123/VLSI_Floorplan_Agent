"""Celery task queue for background processing."""
from __future__ import annotations

from celery import Celery
from pathlib import Path
from sqlalchemy.orm import Session

from ..persistence.database import SessionLocal
from ..persistence import models
from ..ingestion.def_parser import DEFParser
from ..ingestion.lef_parser import LEFParser
from ..ingestion.netlist_parser import NetlistParser
from ..graph.graph_builder import GraphBuilder
from ..gnn.inference import run_inference
from ..evolutionary.ga import GeneticAlgorithm
from ..sta_integration.sta_runner import STARunner
from ..storage.s3_client import S3Client

app = Celery("job_queue")


@app.task(name="process_job")
def process_job(job_id: int) -> None:
    """Background task for processing jobs."""
    db: Session = SessionLocal()
    try:
        job = db.query(models.Job).get(job_id)
        if not job:
            return
        def_data = DEFParser().parse(Path(job.def_path))
        lef_data = LEFParser().parse(Path(job.lef_path))
        netlist_data = NetlistParser().parse(Path(job.netlist_path))
        graph = GraphBuilder().build(def_data, lef_data, netlist_data)
        coords = run_inference(Path("model.pt"), graph)
        ga = GeneticAlgorithm({}, STARunner(Path("sta")))
        pareto = ga.run(graph, [coords])
        # Persist result
        result = models.Result(job_id=job.id, pareto_front=pareto, metrics={})
        db.add(result)
        job.status = "completed"
        db.commit()
        S3Client().upload("pareto.json", "bucket", f"{job.id}/pareto.json")
    finally:
        db.close()
