# vector_indexer

A high-performance vector indexing engine written in Rust. Implements an **Inverted File (IVF)** index with **two-level clustering**, **memory-mapped storage**, and **SIMD-optimized distance computation**.

Designed as a lightweight, standalone component for building and experimenting with **approximate nearest neighbor (ANN)** search algorithms which can serve as the computational core for retrieval systems.

---

## Key Features

- **Two-level clustering architecture** — IVF clusters organized into shards via super-centroids for scalable distribution
- **Mini-batch K-Means** with early convergence detection and parallel execution using Rayon
- **Memory-mapped I/O** for efficient disk access with zero-copy deserialization
- **Custom binary storage format** with structured headers, indices, and cluster blocks
- **SIMD-accelerated distance calculations** using the `wide` crate

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
