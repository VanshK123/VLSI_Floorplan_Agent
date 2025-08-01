"""Logging configuration for VLSI floorplan agent."""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import structlog


class PerformanceLogger:
    """Performance tracking logger for optimization metrics."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = structlog.get_logger(logger_name)
        self.metrics: Dict[str, Any] = {}
    
    def log_optimization_start(self, job_id: str, num_cells: int, constraints: Dict[str, Any]):
        """Log optimization start."""
        self.logger.info(
            "optimization_started",
            job_id=job_id,
            num_cells=num_cells,
            constraints=constraints,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_gnn_embedding(self, job_id: str, embedding_time: float, memory_usage: float):
        """Log GNN embedding performance."""
        self.logger.info(
            "gnn_embedding_completed",
            job_id=job_id,
            embedding_time=embedding_time,
            memory_usage=memory_usage,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_evolutionary_generation(self, job_id: str, generation: int, 
                                  pareto_size: int, evaluations_per_second: float):
        """Log evolutionary algorithm generation."""
        self.logger.info(
            "evolutionary_generation",
            job_id=job_id,
            generation=generation,
            pareto_size=pareto_size,
            evaluations_per_second=evaluations_per_second,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_sta_evaluation(self, job_id: str, sta_time: float, critical_delay: float,
                          violations: int):
        """Log STA evaluation results."""
        self.logger.info(
            "sta_evaluation",
            job_id=job_id,
            sta_time=sta_time,
            critical_delay=critical_delay,
            violations=violations,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_optimization_complete(self, job_id: str, total_time: float, 
                                final_metrics: Dict[str, float]):
        """Log optimization completion."""
        self.logger.info(
            "optimization_completed",
            job_id=job_id,
            total_time=total_time,
            final_metrics=final_metrics,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_performance_target(self, target_name: str, achieved_value: float, 
                             target_value: float, success: bool):
        """Log performance target achievement."""
        self.logger.info(
            "performance_target",
            target_name=target_name,
            achieved_value=achieved_value,
            target_value=target_value,
            success=success,
            timestamp=datetime.utcnow().isoformat()
        )


class MetricsCollector:
    """Collect and aggregate performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or {}
        }
        self.metrics[name].append(metric_entry)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        values = [entry["value"] for entry in self.metrics[name]]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all metrics."""
        return {name: self.get_metric_summary(name) for name in self.metrics.keys()}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                 structured: bool = True) -> None:
    """Setup logging configuration."""
    
    # Determine log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    if structured:
        # Setup structured logging with structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=numeric_level
        )
    else:
        # Traditional logging setup
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        handlers = [logging.StreamHandler(sys.stdout)]
        
        if log_file:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add file handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        
        # Configure logging
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=handlers
        )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_performance_metrics(metrics: Dict[str, Any], job_id: str):
    """Log performance metrics for a job."""
    logger = get_logger("performance")
    logger.info(
        "job_performance_metrics",
        job_id=job_id,
        metrics=metrics,
        timestamp=datetime.utcnow().isoformat()
    )


def log_optimization_progress(job_id: str, step: str, progress: float, 
                            additional_info: Optional[Dict[str, Any]] = None):
    """Log optimization progress."""
    logger = get_logger("optimization")
    log_data = {
        "job_id": job_id,
        "step": step,
        "progress": progress,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if additional_info:
        log_data.update(additional_info)
    
    logger.info("optimization_progress", **log_data)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an error with context."""
    logger = get_logger("error")
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if context:
        log_data.update(context)
    
    logger.error("application_error", **log_data)


def log_system_metrics(metrics: Dict[str, Any]):
    """Log system-level metrics."""
    logger = get_logger("system")
    logger.info(
        "system_metrics",
        metrics=metrics,
        timestamp=datetime.utcnow().isoformat()
    )


class OptimizationLogger:
    """Specialized logger for optimization processes."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logger = get_logger("optimization")
        self.start_time = datetime.utcnow()
    
    def log_step_start(self, step_name: str, step_data: Optional[Dict[str, Any]] = None):
        """Log the start of an optimization step."""
        log_data = {
            "job_id": self.job_id,
            "step": step_name,
            "action": "start",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if step_data:
            log_data.update(step_data)
        
        self.logger.info("optimization_step", **log_data)
    
    def log_step_complete(self, step_name: str, duration: float, 
                         step_data: Optional[Dict[str, Any]] = None):
        """Log the completion of an optimization step."""
        log_data = {
            "job_id": self.job_id,
            "step": step_name,
            "action": "complete",
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if step_data:
            log_data.update(step_data)
        
        self.logger.info("optimization_step", **log_data)
    
    def log_gnn_performance(self, embedding_time: float, memory_usage: float,
                           num_nodes: int, num_edges: int):
        """Log GNN-specific performance metrics."""
        self.logger.info(
            "gnn_performance",
            job_id=self.job_id,
            embedding_time=embedding_time,
            memory_usage=memory_usage,
            num_nodes=num_nodes,
            num_edges=num_edges,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_evolutionary_performance(self, generation: int, population_size: int,
                                   pareto_front_size: int, evaluations_per_second: float):
        """Log evolutionary algorithm performance metrics."""
        self.logger.info(
            "evolutionary_performance",
            job_id=self.job_id,
            generation=generation,
            population_size=population_size,
            pareto_front_size=pareto_front_size,
            evaluations_per_second=evaluations_per_second,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_sta_performance(self, sta_time: float, critical_delay: float,
                           slack_violations: int, total_nets: int):
        """Log STA performance metrics."""
        self.logger.info(
            "sta_performance",
            job_id=self.job_id,
            sta_time=sta_time,
            critical_delay=critical_delay,
            slack_violations=slack_violations,
            total_nets=total_nets,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_final_results(self, total_time: float, pareto_front_size: int,
                         best_area: float, best_delay: float):
        """Log final optimization results."""
        self.logger.info(
            "optimization_complete",
            job_id=self.job_id,
            total_time=total_time,
            pareto_front_size=pareto_front_size,
            best_area=best_area,
            best_delay=best_delay,
            timestamp=datetime.utcnow().isoformat()
        )


# Initialize logging on module import
setup_logging()
