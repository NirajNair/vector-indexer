# vector_indexer

A high-performance vector indexing engine written in Rust. It implements an **Inverted File (IVF)** index with **parallel K-Means clustering** and **SIMD-optimized distance computation**.

The library is designed as a lightweight, standalone component for building and experimenting with **approximate nearest neighbor (ANN)** search algorithms. It can serve as the computational core for larger retrieval systems or as a research tool for understanding and benchmarking vector indexing techniques.

---

## Key Features

- **Parallel K-Means clustering** using [Rayon] for multicore scalability.
- **SIMD-accelerated distance calculations** using Rust’s `wide` crate.
- **Early convergence detection** for faster and stable training.
- **Modular architecture** to support future extensions (e.g., PQ, WAL, or memory-mapped storage).
- **Benchmark-ready design** for evaluating IVF performance across datasets.

---
