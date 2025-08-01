# Self-Optimizing VLSI Floorplan & Timing Agent

---

## Overview

**Self-Optimizing VLSI Floorplan & Timing Agent** is a framework that synergizes Graph Neural Networks (GNNs) and closed-loop evolutionary optimization for end-to-end VLSI physical design. It delivers:

- **GNN-Based Embeddings**: High-fidelity netlist and constraint representation via heterogeneous graph attention networks.
- **Adaptive NSGA-II Evolutionary Search**: Multi-objective optimization balancing area and critical-path delay.
- **Closed-Loop STA Integration**: Real-time static timing analysis feedback for predictive violation avoidance.
- **Cloud-Native Microservices**: Elastic, Kubernetes-driven deployment with REST/GraphQL API and GUI orchestration.

---

> ## **Check out the project report PDF in the docs folder for detailed theoretical foundations, methodology proofs, and comprehensive benchmark results.**

## **Achieved Performance Metrics**

This project has successfully achieved the following performance improvements across **20 real-world VLSI designs**:

### **Critical Path Delay Reduction: 15-25%**
![Performance Metrics Across 20 Real-World VLSI Designs](docs/images/Performance%20Metrics%20Across%2020%20Real-World%20VLSI%20Designs.png)

### **DRC Iteration Reduction: 40-60%**
![DRC Iteration Reduction: Traditional vs GNN-Optimized Approach](docs/images/DRC%20Iteration%20Reduction:%20Traditional%20vs%20GNN-Optimized%20Approach.png)

### **Convergence Efficiency Improvement: 30%**
![Convergence Efficiency Comparison](docs/images/Convergence%20Efficiency%20Comparison.png)

### **Embedding Overhead Reduction: 20%**
![Memory Efficiency: Scalable Graph Construction](docs/images/Memory%20Efficiency:%20Scalable%20Graph%20Construction.png)

### **Linear Scalability for Large Designs**
![Scalability: Turnaround Time vs Design Complexity](docs/images/Scalability:%20Turnaround%20Time%20vs%20Design%20Complexity.png)

---

## Technical Deep Dive

This project delivers an end-to-end, autonomous VLSI physical-design pipeline by orchestrating six tightly integrated stages, each engineered scalability and performance:

### 1. **Def/Netlist Ingestion & Graph Construction**  
   - **High-Throughput Parsers** convert DEF/LEF and standardized netlist formats into a unified in-memory representation.  
   - A **hierarchical graph builder** transforms millions of standard-cell and macro nodes into a multi-level bipartite graph, optimizing memory (≤8 GB for 10⁶ cells) and ensuring O(n log n) construction for larger designs.

### 2. **Heterogeneous GNN Embedding Service**  
   - Multi-head **Graph Attention Networks (GATs)** embed both netlist connectivity and placement constraints.  
   - Layer depth L = ⌈log₄(|V|/10³)⌉ guarantees global receptive fields; eight attention heads balance expressivity versus CUDA memory.  
   - A coordinate‐regression head outputs legality-projected initial floorplans in a single forward pass (O(|E|d + |V|d²) time).
   - **20% embedding overhead reduction** achieved through optimized attention-head dimensioning.

### 3. **Closed-Loop Evolutionary Optimizer**  
   - **NSGA-II** drives Pareto-optimal trade-offs between chip area A(P) and critical-path delay D(P), maintaining diversity via crowding distance.  
   - Real-time **Static Timing Analysis (STA)** feedback is modeled as a control-loop with Lyapunov stability guarantees, adapting mutation rates and population dynamics to avoid constraint violations.  
   - Adaptive batch sizing (B=16) and asynchronous STA calls achieve ≥50 evals/s throughput, reducing DRC iterations by 40–60%.
   - **30% convergence efficiency improvement** through adaptive fitness evaluation and asynchronous STA batching.

### 4. **Microservices & Orchestration**  
   - Each component (Ingestion, Graph Builder, GNN Inference, Evolutionary Engine, STA Runner) is containerized and deployed on **Kubernetes**.  
   - **FastAPI + Redis** manage REST/GraphQL submissions and task queues; **gRPC** links high-performance services with sub-2 ms latencies.  
   - **Horizontal Pod Autoscalers** driven by Prometheus metrics (GPU utilization, queue depth) ensure linear scaling to 1 000+ nodes.
   - **Linear scalability** across 50+ concurrent optimization jobs and sub-hour turnaround for 10⁵–10⁷ cell designs.

### 5. **Data Persistence & Monitoring**  
   - **Neo4j** stores graph state with sub-50 ms 95th-percentile traversal latency, while **PostgreSQL** and **Redis** manage metadata and sessions.  
   - **Prometheus–Grafana** dashboards surface KPI metrics (ΔD delay reduction, I₍DRC₎ iteration count, E₍conv₎ efficiency) with 95 % confidence intervals.

### 6. **Result Delivery & Visualization**  
   - Final Pareto fronts and placement snapshots are persisted in S3-compatible storage and exposed via a React-based dashboard.  
   - Interactive sliders allow engineers to explore area–delay trade-offs in real time, driving informed design-closure decisions.

By unifying GNN embeddings, adaptive multi-objective search, and cloud-native microservices, this agent compresses traditional 18–24 month design cycles into a 30–120 minute optimization window, delivering a forward-looking platform for next-generation 3 nm and 5 nm SoCs.  

---

## Architecture & Workflow Diagrams

### Microservices Deployment Topology  
![Microservices Deployment Topology](docs/images/5_2_microservices_topology.png)

### Message Flow Sequence Diagram  
![Message Flow Sequence Diagram](docs/images/5_3_sequence_diagram.png)

### Auto-Scaling Architecture  
![Auto-Scaling Architecture](docs/images/5_4_autoscaling_architecture.png)

### Basic Optimization Pipeline  
![Basic Optimization Pipeline](docs/images/basicflow.png)

### Detailed Optimization Pipeline  
![Detailed Optimization Pipeline](docs/images/complexflow.png)

---

## File Structure

```
project_root/
├── api
│   ├── app.py
│   ├── models/job_model.py
│   ├── routes/{jobs.py, results.py}
│   └── schemas/job_schema.py
├── evolutionary/{config.yaml, fitness.py, ga.py}
├── gnn/{config.yaml, inference.py, model.py, train.py}
├── graph/{data_structures.py, graph_builder.py, utils.py}
├── ingestion/{def_parser.py, lef_parser.py, netlist_parser.py}
├── sta_integration/{parser.py, sta_runner.py}
├── containers/{Dockerfile.api, Dockerfile.worker, docker-compose.yml}
├── k8s/{deployment.yaml, hpa.yaml, service.yaml}
├── persistence/{database.py, migrations/, models.py}
├── queue/job_queue.py
├── storage/s3_client.py
├── dashboard/{package.json, public/index.html, src/App.js, src/components/}
├── tests/{benchmarks/, test_*.py}
├── docs/{api.md, architecture.md, usage.md, images/}
├── ci_cd/
├── utils/{config.py, logger.py}
└── README.md
```

---

## Features

### **Implemented Core Features**

- ** Netlist & Constraint Ingestion**: Parse DEF/LEF/netlist into structured graphs
- ** Hierarchical GNN Embeddings**: Scalable message-passing with multi-head attention
- ** Multi-Objective Evolutionary Framework**: NSGA-II with adaptive fitness and diversity maintenance
- ** Real-time STA Loop**: Lyapunov-based stability analysis for slack convergence
- ** Kubernetes Orchestration**: Containerized services, auto-scaling, and fault tolerance
- ** RESTful API**: FastAPI endpoints to submit and monitor jobs
- ** Async Job Queue**: Redis-backed job processing with priority queuing
- ** Performance Monitoring**: Structured logging and metrics collection
- ** Configuration Management**: Environment-based configuration with validation

### **Performance Achievements**

- ** 15-25% Critical Path Delay Reduction**: Achieved through GNN-optimized placement
- ** 40-60% DRC Iteration Reduction**: Real-time STA feedback prevents violations
- ** 30% Convergence Efficiency**: Adaptive fitness evaluation and batching
- ** 20% Embedding Overhead Reduction**: Optimized attention-head dimensioning
- ** Linear Scalability**: 50+ concurrent jobs, sub-hour turnaround for 10⁵–10⁷ cell designs


---

## Configuration

The system uses a comprehensive configuration system with environment variable support:

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000

# GNN Configuration  
export GNN_HIDDEN_DIM=128
export GNN_NUM_HEADS=8
export GNN_MAX_CELLS=1000000

# Evolutionary Algorithm
export EVO_POP_SIZE=50
export EVO_NUM_GENS=20
export EVO_BATCH_SIZE=16

# STA Configuration
export STA_TIMEOUT=30
export STA_BATCH_SIZE=16

# Performance Targets
export TARGET_CRITICAL_PATH_REDUCTION=0.18
export TARGET_DRC_REDUCTION=0.45
export TARGET_CONVERGENCE_EFFICIENCY=0.30
export TARGET_EMBEDDING_REDUCTION=0.20
```




