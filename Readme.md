# vector_indexer

A vector indexing engine written in Rust. Implements an **Inverted File (IVF)** index with **two-level clustering**, **async I/O with tokio-uring**, and **SIMD-optimized distance computation**.

Designed as a lightweight, standalone component for building and experimenting with **approximate nearest neighbor (ANN)** search algorithms which can serve as the computational core for retrieval systems.

---

## Key Features

- **Two-level clustering architecture** — IVF clusters organized into shards via super-centroids for scalable distribution
- **Async I/O with tokio-uring** — asynchronous disk I/O for efficient shard loading and search operations
- **Mini-batch K-Means** with early convergence detection, parallel execution using Rayon, and deterministic initialization
- **Hierarchical assignment** — Optimized point-to-centroid assignment for large cluster counts (k > 100) using meta-clustering
- **Custom binary storage format** with structured headers, indices, and cluster blocks
- **SIMD-accelerated distance calculations** using the `wide` crate for vectorized Euclidean distance computation

## Commands

### Run all tests

```bash
docker-compose run --rm app cargo test --features internal_tests
```

### Run benchmarks (requires bench.yaml config file)

```bash
docker-compose run --rm app cargo run --release --bin bench --features internal_tests -- bench.yaml
```

### Interactive shell

```bash
docker-compose run --rm app bash
```

### Run specific test

```bash
docker-compose run --rm app cargo test --features internal_tests test_name
```

### Build the project

```bash
docker-compose run --rm app cargo build --release
```
