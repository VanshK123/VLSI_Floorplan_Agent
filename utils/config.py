"""Configuration management for VLSI floorplan agent."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the VLSI floorplan agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Default configuration
        self._config = {
            # API Configuration
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "workers": 4
            },
            
            # Database Configuration
            "database": {
                "url": "postgresql://localhost/vlsi_agent",
                "pool_size": 10,
                "max_overflow": 20
            },
            
            # Redis Configuration
            "redis": {
                "url": "redis://localhost:6379",
                "max_connections": 20
            },
            
            # GNN Configuration
            "gnn": {
                "model_path": "models/floorplan_gnn.pt",
                "hidden_dim": 128,
                "num_layers": 4,
                "num_heads": 8,
                "dropout": 0.1,
                "max_cells": 1000000,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            
            # Evolutionary Algorithm Configuration
            "evolutionary": {
                "pop_size": 50,
                "num_gens": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "batch_size": 16,
                "sta_timeout": 30,
                "adaptive_mutation": True,
                "lyapunov_stability": True
            },
            
            # STA Configuration
            "sta": {
                "binary_path": "/usr/local/bin/sta_tool",
                "timeout": 30,
                "batch_size": 16,
                "max_concurrent": 4
            },
            
            # Storage Configuration
            "storage": {
                "s3_bucket": "vlsi-results",
                "s3_region": "us-east-1",
                "local_storage": "/tmp/vlsi_results"
            },
            
            # Kubernetes Configuration
            "kubernetes": {
                "namespace": "vlsi-agent",
                "replicas": 3,
                "resources": {
                    "cpu": "2",
                    "memory": "4Gi"
                }
            },
            
            # Monitoring Configuration
            "monitoring": {
                "prometheus_enabled": True,
                "grafana_enabled": True,
                "metrics_port": 9090
            },
            
            # Performance Targets
            "targets": {
                "critical_path_delay_reduction": 0.15,  # 15-25%
                "drc_iterations_reduction": 0.40,       # 40-60%
                "convergence_efficiency": 0.30,         # 30%
                "embedding_overhead_reduction": 0.20,   # 20%
                "max_processing_time": 3600,            # 1 hour
                "max_memory_usage": 8 * 1024 * 1024 * 1024  # 8GB
            }
        }
        
        # Load from file if exists
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._merge_config(file_config)
        
        # Override with environment variables
        self._load_from_env()
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing config."""
        def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
            result = d1.copy()
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        self._config = merge_dicts(self._config, new_config)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # API
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port"),
            "API_DEBUG": ("api", "debug"),
            "API_WORKERS": ("api", "workers"),
            
            # Database
            "DATABASE_URL": ("database", "url"),
            "DATABASE_POOL_SIZE": ("database", "pool_size"),
            "DATABASE_MAX_OVERFLOW": ("database", "max_overflow"),
            
            # Redis
            "REDIS_URL": ("redis", "url"),
            "REDIS_MAX_CONNECTIONS": ("redis", "max_connections"),
            
            # GNN
            "GNN_MODEL_PATH": ("gnn", "model_path"),
            "GNN_HIDDEN_DIM": ("gnn", "hidden_dim"),
            "GNN_NUM_LAYERS": ("gnn", "num_layers"),
            "GNN_NUM_HEADS": ("gnn", "num_heads"),
            "GNN_DROPOUT": ("gnn", "dropout"),
            "GNN_MAX_CELLS": ("gnn", "max_cells"),
            "GNN_BATCH_SIZE": ("gnn", "batch_size"),
            "GNN_LEARNING_RATE": ("gnn", "learning_rate"),
            
            # Evolutionary
            "EVO_POP_SIZE": ("evolutionary", "pop_size"),
            "EVO_NUM_GENS": ("evolutionary", "num_gens"),
            "EVO_MUTATION_RATE": ("evolutionary", "mutation_rate"),
            "EVO_CROSSOVER_RATE": ("evolutionary", "crossover_rate"),
            "EVO_BATCH_SIZE": ("evolutionary", "batch_size"),
            "EVO_STA_TIMEOUT": ("evolutionary", "sta_timeout"),
            "EVO_ADAPTIVE_MUTATION": ("evolutionary", "adaptive_mutation"),
            "EVO_LYAPUNOV_STABILITY": ("evolutionary", "lyapunov_stability"),
            
            # STA
            "STA_BINARY_PATH": ("sta", "binary_path"),
            "STA_TIMEOUT": ("sta", "timeout"),
            "STA_BATCH_SIZE": ("sta", "batch_size"),
            "STA_MAX_CONCURRENT": ("sta", "max_concurrent"),
            
            # Storage
            "STORAGE_S3_BUCKET": ("storage", "s3_bucket"),
            "STORAGE_S3_REGION": ("storage", "s3_region"),
            "STORAGE_LOCAL_PATH": ("storage", "local_storage"),
            
            # Kubernetes
            "K8S_NAMESPACE": ("kubernetes", "namespace"),
            "K8S_REPLICAS": ("kubernetes", "replicas"),
            "K8S_CPU": ("kubernetes", "resources", "cpu"),
            "K8S_MEMORY": ("kubernetes", "resources", "memory"),
            
            # Monitoring
            "MONITORING_PROMETHEUS": ("monitoring", "prometheus_enabled"),
            "MONITORING_GRAFANA": ("monitoring", "grafana_enabled"),
            "MONITORING_METRICS_PORT": ("monitoring", "metrics_port"),
            
            # Targets
            "TARGET_CRITICAL_PATH_REDUCTION": ("targets", "critical_path_delay_reduction"),
            "TARGET_DRC_REDUCTION": ("targets", "drc_iterations_reduction"),
            "TARGET_CONVERGENCE_EFFICIENCY": ("targets", "convergence_efficiency"),
            "TARGET_EMBEDDING_REDUCTION": ("targets", "embedding_overhead_reduction"),
            "TARGET_MAX_PROCESSING_TIME": ("targets", "max_processing_time"),
            "TARGET_MAX_MEMORY": ("targets", "max_memory_usage")
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
    
    def _set_nested_value(self, path: tuple, value: str):
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value to appropriate type
        key = path[-1]
        if isinstance(current.get(key), bool):
            current[key] = value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(current.get(key), int):
            current[key] = int(value)
        elif isinstance(current.get(key), float):
            current[key] = float(value)
        else:
            current[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_sections = ['api', 'database', 'redis', 'gnn', 'evolutionary', 'sta']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific values
        if self.get('api.port') < 1 or self.get('api.port') > 65535:
            raise ValueError("Invalid API port")
        
        if self.get('evolutionary.pop_size') < 1:
            raise ValueError("Population size must be positive")
        
        if self.get('gnn.hidden_dim') < 1:
            raise ValueError("Hidden dimension must be positive")
        
        return True
    
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets for optimization."""
        return self.get_section('targets')
    
    def get_gnn_config(self) -> Dict[str, Any]:
        """Get GNN-specific configuration."""
        return self.get_section('gnn')
    
    def get_evolutionary_config(self) -> Dict[str, Any]:
        """Get evolutionary algorithm configuration."""
        return self.get_section('evolutionary')
    
    def get_sta_config(self) -> Dict[str, Any]:
        """Get STA configuration."""
        return self.get_section('sta')


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        _config_instance.validate()
    return _config_instance


def set_config(config: Config):
    """Set global configuration instance."""
    global _config_instance
    _config_instance = config


def reload_config():
    """Reload configuration from file."""
    global _config_instance
    _config_instance = None
    return get_config()
