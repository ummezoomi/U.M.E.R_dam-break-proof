# U.M.E.R Engine — Fluid Dynamics: Dam Break Proof

![Domain](https://img.shields.io/badge/Domain-Fluid%20Dynamics-00bcd4?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Massively%20Parallel%20Search-00E676?style=flat-square)
![Memory](https://img.shields.io/badge/Memory-Autonomous%20Defragmentation-purple?style=flat-square)

> **Part of the U.M.E.R (Uniform Memory Encoded Representation) GPU Compute Core**

---

### Academic Preprint & Citation
This repository serves as empirical proof of the dynamic spatial memory capabilities defined in our manuscript submitted to the **Journal of Supercomputing**:

* **Title:** *U.M.E.R: A Deterministic Sort Free Architecture for High Density Physical & Agentic Systems*
* **Author:** Muhammad Umer
* **Preprint DOI:** [https://doi.org/10.21203/rs.3.rs-9848255/v1](https://doi.org/10.21203/rs.3.rs-9848255/v1)

---

## Overview

The **Dam Break** simulation is the gold standard benchmark in High-Performance Computing (HPC) for validating fluid dynamics and dynamic spatial querying. As a vertical wall of simulated fluid collapses, the physical geometry of the system undergoes rapid, chaotic expansion.

For traditional physics engines, this rapid expansion creates massive memory fragmentation. As particles scatter across the simulation bound, their associated memory addresses scatter across the GPU's VRAM. This destroys L1/L2 cache locality, chokes memory bandwidth, and forces traditional engines to halt and invoke slow, CPU-driven memory sorting or tree-rebuilding (Octrees/BVHs).

This repository proves that the **U.M.E.R. architecture solves the fragmentation bottleneck directly in hardware-native $O(1)$ time.**

## Architectural Innovations Demonstrated

* **Massively Parallel Search for Fluids:** Calculating fluid pressure and density requires millions of particle-to-particle neighbor queries per frame. Instead of traversing a search tree $O(\log N)$, the engine treats the fluid bounds as a localized hash space. Particles query their immediate physical neighbors simultaneously in pure $O(1)$ time with zero thread branching.
* **Autonomous VRAM Defragmentation:** As the dam breaks and the fluid splashes outward, the U.M.E.R. pipeline dynamically compacts the scattered memory states without OS intervention:
  1. *Atomic Histogramming* flags active fluid coordinates.
  2. *Parallel Inclusive Scan (Prefix-Sum)* collapses the empty space in the virtual memory map mathematically.
  3. *Deterministic Scatter* writes the dispersed fluid particles into a fresh, perfectly contiguous memory block.
* **Peak Bandwidth Utilization:** By maintaining contiguous array packing even during violent physical scattering, the GPU achieves peak VRAM bandwidth (up to 1.7 TB/s effective) without the load-factor fallacies of legacy spatial hashing.
* **Absolute Determinism:** Unlike probabilistic approximations, the topological collapse of the fluid is preserved down to the final floating-point bit. The exact same initial state will produce the exact same splash, every single time.

## Running the Simulation

The simulation, rendering pipeline, and memory compaction benchmarks are encapsulated within `dam-break-proof.ipynb`.

### Prerequisites
* High-bandwidth, CUDA-capable GPU (NVIDIA HPC/RTX class recommended for peak scatter-gather throughput)
* Python 3.10+
* Configured U.M.E.R GPU / `cupy` bindings

### Execution
Launch the Jupyter instance to initialize the fluid tensor blocks and trigger the spatial collapse:
```bash
jupyter notebook dam-break-proof.ipynb
