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

### Build and run the benchmark

```bash
docker compose -f docker-compose.bench.yml build
docker compose -f docker-compose.bench.yml run --rm bench
```

#### Compare with Faiss

```bash
BACKEND=both docker compose -f docker-compose.bench.yml run --rm bench
```

#### Custom parameters

```bash
N=50000 D=64 docker compose -f docker-compose.bench.yml run --rm bench
```

### Benchmark with real datasets (e.g., SIFT1M)

The benchmark supports using real datasets from local files. By default it generates synthetic data, but you can also point it at:\n+- **NumPy `.npy` arrays**\n+- **Faiss dataset binaries**: `.fvecs` (vectors) + `.ivecs` (ground truth)

#### Dataset layout

You can use either of these layouts:

```
data/
  sift1m/
    xb.npy    # Database vectors (float32, shape [N, D])
    xq.npy    # Query vectors (float32, shape [NQ, D])
    gt.npy    # Ground truth indices (optional, int64, shape [NQ, K])
```

Or, if you downloaded the standard SIFT1M files (as commonly distributed for Faiss benchmarks), you can place them directly under `data/`:\n+\n+`\n+data/\n+  sift_base.fvecs\n+  sift_query.fvecs\n+  sift_groundtruth.ivecs\n+  sift_learn.fvecs\n+`\n+\n The `data/` directory is gitignored so large datasets won't be committed.
The `data/` directory is gitignored so large datasets won't be committed.

#### Using a subset

The `--n` parameter (or `N` env var) limits how many database vectors to use. This is useful for faster benchmarks:

- Full SIFT1M has 1M vectors. Use `N=100000` to only use the first 100k.
- Ground truth will be recomputed if the provided ground truth contains out-of-bounds indices for the sliced dataset.

#### Docker

```bash
# SIFT1M with 100k vectors (NumPy .npy files)
DATASET=local_npy \
XB_PATH=/workspace/data/sift1m/xb.npy \
XQ_PATH=/workspace/data/sift1m/xq.npy \
N=100000 \
docker compose -f docker-compose.bench.yml run --rm bench

# With pre-computed ground truth (NumPy .npy)
DATASET=local_npy \
XB_PATH=/workspace/data/sift1m/xb.npy \
XQ_PATH=/workspace/data/sift1m/xq.npy \
GT_PATH=/workspace/data/sift1m/gt.npy \
N=100000 \
docker compose -f docker-compose.bench.yml run --rm bench

# SIFT1M with 100k vectors (Faiss fvecs/ivecs files in ./data)
DATASET=local_npy \
XB_PATH=/workspace/data/sift_base.fvecs \
XQ_PATH=/workspace/data/sift_query.fvecs \
GT_PATH=/workspace/data/sift_groundtruth.ivecs \
N=100000 \
docker compose -f docker-compose.bench.yml run --rm bench
```