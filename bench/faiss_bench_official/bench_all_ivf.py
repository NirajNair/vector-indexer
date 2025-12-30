#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from: https://github.com/facebookresearch/faiss/blob/main/benchs/bench_all_ivf/bench_all_ivf.py
# Changes:
#   - Replaced datasets_fb/datasets_oss with synthetic data generation
#   - Added vector_indexer backend support via VectorIndexerFaissAdapter
#   - Simplified to focus on IVF benchmark comparison

"""
Official Faiss-style benchmark script adapted for vector_indexer comparison.

Uses the official Faiss eval_setting() methodology:
- Runs search until min_test_duration is reached
- Averages timing across multiple runs
- Reports recalls at rank 1, 10, 100

Usage (synthetic dataset - default):
    python bench_all_ivf.py --backend vector_indexer --n 100000 --d 128 --k 100
    python bench_all_ivf.py --backend faiss --n 100000 --d 128 --k 100
    python bench_all_ivf.py --backend both --n 100000 --d 128 --k 100

Usage (local dataset files):
    # NumPy .npy arrays
    python bench_all_ivf.py --dataset local_npy --xb-path data/sift1m/xb.npy --xq-path data/sift1m/xq.npy --n 100000 --nq 1000 --k 100
    python bench_all_ivf.py --dataset local_npy --xb-path data/sift1m/xb.npy --xq-path data/sift1m/xq.npy --gt-path data/sift1m/gt.npy --n 100000

    # Faiss fvecs/ivecs binaries (SIFT1M files)
    python bench_all_ivf.py --dataset local_npy --xb-path data/sift_base.fvecs --xq-path data/sift_query.fvecs --gt-path data/sift_groundtruth.ivecs --n 100000 --nq 1000 --k 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss not installed. Faiss backend unavailable.")


# ==============================================================================
# Synthetic Dataset Generation (replaces datasets_oss/datasets_fb)
# ==============================================================================


def generate_synthetic_data(
    n: int, d: int, nq: int, k: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate deterministic synthetic dataset.

    Returns:
        xb: Database vectors, shape (n, d)
        xq: Query vectors, shape (nq, d)
        gt: Ground truth indices, shape (nq, k)
    """
    print(f"Generating synthetic dataset: n={n}, d={d}, nq={nq}, k={k}, seed={seed}")
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d)).astype(np.float32)
    xq = rng.standard_normal((nq, d)).astype(np.float32)

    # Compute ground truth using Faiss brute force
    if faiss is None:
        raise RuntimeError("Faiss required for ground truth computation")

    print("Computing ground truth with Faiss Flat L2...")
    gt_index = faiss.IndexFlatL2(d)
    gt_index.add(xb)
    _, gt = gt_index.search(xq, k)

    return xb, xq, gt


# ==============================================================================
# Local Dataset Loading (.npy or .fvecs/.ivecs files)
# ==============================================================================


def _read_fvecs(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """
    Read a Faiss .fvecs file.

    Format: repeated records of [int32 dim][float32 x dim].

    Returns float32 array of shape (n, d).
    """
    # Read dim from the first record
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=1)
        if header.size != 1:
            raise ValueError(f"Empty or invalid fvecs file: {path}")
        d = int(header[0])

    # Each record is (d + 1) int32 values when viewed as int32
    count = None
    if max_rows is not None:
        count = int(max_rows) * (d + 1)

    a = np.fromfile(path, dtype=np.int32, count=count)
    if a.size % (d + 1) != 0:
        raise ValueError(
            f"Invalid fvecs file (size not multiple of d+1): {path} (d={d}, ints={a.size})"
        )
    a = a.reshape(-1, d + 1)
    vectors = a[:, 1:].view(np.float32)
    return np.ascontiguousarray(vectors)


def _read_ivecs(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """
    Read a Faiss .ivecs file.

    Format: repeated records of [int32 k][int32 x k].

    Returns int64 array of shape (n, k).
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=1)
        if header.size != 1:
            raise ValueError(f"Empty or invalid ivecs file: {path}")
        k = int(header[0])

    count = None
    if max_rows is not None:
        count = int(max_rows) * (k + 1)

    a = np.fromfile(path, dtype=np.int32, count=count)
    if a.size % (k + 1) != 0:
        raise ValueError(
            f"Invalid ivecs file (size not multiple of k+1): {path} (k={k}, ints={a.size})"
        )
    a = a.reshape(-1, k + 1)
    ids = a[:, 1:]
    return np.ascontiguousarray(ids.astype(np.int64, copy=False))


def _load_vectors(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """Load vectors from .npy or .fvecs."""
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        if max_rows is not None:
            arr = arr[:max_rows]
        arr = np.asarray(arr, dtype=np.float32)
        return np.ascontiguousarray(arr)
    if path.endswith(".fvecs"):
        return _read_fvecs(path, max_rows=max_rows)
    raise ValueError(f"Unsupported vector file type (expected .npy or .fvecs): {path}")


def _load_groundtruth(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    """Load ground truth from .npy or .ivecs."""
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        if max_rows is not None:
            arr = arr[:max_rows]
        arr = np.asarray(arr, dtype=np.int64)
        return np.ascontiguousarray(arr)
    if path.endswith(".ivecs"):
        return _read_ivecs(path, max_rows=max_rows)
    raise ValueError(
        f"Unsupported ground truth file type (expected .npy or .ivecs): {path}"
    )


def load_local_npy_data(
    xb_path: str,
    xq_path: str,
    gt_path: Optional[str],
    n: int,
    nq: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from local files (.npy or Faiss .fvecs/.ivecs).

    Arrays are sliced to (n, nq, k) to allow using a subset of larger datasets.
    If gt_path is not provided, ground truth is computed using Faiss FlatL2.

    Args:
        xb_path: Path to database vectors (.npy or .fvecs)
        xq_path: Path to query vectors (.npy or .fvecs)
        gt_path: Optional path to ground truth indices (.npy or .ivecs)
        n: Max number of database vectors to use (slice xb[:n])
        nq: Max number of query vectors to use (slice xq[:nq])
        k: Number of neighbors (used to slice gt and/or compute it)

    Returns:
        xb: Database vectors, shape (min(n, N), d)
        xq: Query vectors, shape (min(nq, NQ), d)
        gt: Ground truth indices, shape (nq_actual, k)
    """
    print("Loading local dataset files:")
    print(f"  xb: {xb_path}")
    print(f"  xq: {xq_path}")
    print(f"  gt: {gt_path if gt_path else '(will compute with FlatL2)'}")

    # Load database vectors
    xb = _load_vectors(xb_path, max_rows=n)

    # Load query vectors
    xq = _load_vectors(xq_path, max_rows=nq)

    # Validate dimensions match
    if xb.shape[1] != xq.shape[1]:
        raise ValueError(
            f"Dimension mismatch: xb has d={xb.shape[1]}, xq has d={xq.shape[1]}"
        )

    d = xb.shape[1]
    print(f"  Loaded xb: {xb.shape}, xq: {xq.shape}, d={d}")

    # Actual sizes (load already capped by max_rows, but keep consistent)
    n_actual = xb.shape[0]
    nq_actual = xq.shape[0]

    # Load or compute ground truth
    gt = None
    if gt_path:
        gt = _load_groundtruth(gt_path, max_rows=nq_actual)
        print(f"  Loaded gt: {gt.shape} (from {gt_path})")
        sys.stdout.flush()

        # Slice gt to match nq and k
        if gt.shape[1] < k:
            raise ValueError(
                f"Ground truth has only {gt.shape[1]} neighbors per query, but k={k} was requested"
            )
        gt = gt[:nq_actual, :k]
        print(f"  Sliced gt to: {gt.shape}")

        # Validate gt indices are within bounds of sliced xb
        max_idx = int(gt.max()) if gt.size else -1
        invalid_count = int((gt >= n_actual).sum())
        invalid_pct = 100.0 * invalid_count / gt.size if gt.size else 0

        print(
            f"  GT index range: [{gt.min()}, {max_idx}], invalid (>={n_actual}): {invalid_count}/{gt.size} ({invalid_pct:.1f}%)"
        )
        sys.stdout.flush()

        if max_idx >= n_actual:
            print(
                f"  WARNING: gt contains indices up to {max_idx} but xb has only {n_actual} vectors."
            )
            print("  Recomputing ground truth for sliced dataset...")
            sys.stdout.flush()
            gt = None

    if gt is None:
        # Compute ground truth using Faiss brute force
        if faiss is None:
            raise RuntimeError("Faiss required for ground truth computation")

        print(
            f"  Computing ground truth with Faiss FlatL2 (n={n_actual}, nq={nq_actual}, k={k})..."
        )
        sys.stdout.flush()
        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        _, gt = gt_index.search(xq, k)
        print(f"  Ground truth computed. GT index range: [{gt.min()}, {gt.max()}]")
        sys.stdout.flush()

    print(f"  Dataset ready: xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")
    sys.stdout.flush()
    return xb, xq, gt


# ==============================================================================
# Official Faiss eval_setting() - kept as close to upstream as possible
# ==============================================================================


def eval_setting(index, xq, gt, k, inter, min_time):
    """
    Evaluate searching in terms of precision vs. speed.

    This is the official Faiss evaluation methodology:
    - Runs search repeatedly until min_time seconds have elapsed
    - Averages timing across runs
    - Reports recalls at rank 1, 10, 100

    Args:
        index: Index to evaluate (Faiss index or VectorIndexerFaissAdapter)
        xq: Query vectors
        gt: Ground truth indices
        k: Number of neighbors to retrieve
        inter: If True, use intersection measure instead of recall
        min_time: Minimum time to run the benchmark

    Returns:
        dict with timing and recall metrics
    """
    nq = xq.shape[0]

    # Get IVF stats if available (Faiss only)
    ivf_stats = None
    if faiss is not None:
        try:
            ivf_stats = faiss.cvar.indexIVF_stats
            ivf_stats.reset()
        except:
            pass

    nrun = 0
    t0 = time.time()
    while True:
        D, I = index.search(xq, k)
        nrun += 1
        t1 = time.time()
        if t1 - t0 > min_time:
            break

    ms_per_query = (t1 - t0) * 1000.0 / nq / nrun
    qps = 1000.0 / ms_per_query

    res = {"ms_per_query": ms_per_query, "qps": qps, "nrun": nrun}

    if inter:
        # Intersection measure (Faiss default for some benchmarks)
        rank = k
        inter_measure = faiss.eval_intersection(gt[:, :rank], I[:, :rank]) / (nq * rank)
        print("%.4f" % inter_measure, end=" ")
        res["inter_measure"] = inter_measure
    else:
        # Recall at ranks 1, 10, 100
        # Recall@k = fraction of queries where true NN (gt[:,0]) appears in top-k results
        res["recalls"] = {}
        true_nn = gt[
            :, 0:1
        ]  # Shape: (nq, 1) - the true nearest neighbor for each query
        for rank in [1, 10, 100]:
            if rank > k:
                continue
            # Check if true NN appears anywhere in top-rank results for each query
            # Broadcasting: I[:, :rank] is (nq, rank), true_nn is (nq, 1)
            # Result is (nq, rank), then .any(axis=1) gives (nq,) bool array
            matches = (I[:, :rank] == true_nn).any(axis=1)
            recall = matches.sum() / float(nq)
            print("R@%-3d=%.4f" % (rank, recall), end="  ")
            res["recalls"][rank] = recall

    print("  %9.3f ms/q  %9.1f QPS" % (ms_per_query, qps), end="  ")

    if ivf_stats is not None:
        try:
            print("ndis=%d" % (ivf_stats.ndis / nrun), end="")
            res["ndis"] = ivf_stats.ndis / nrun
        except:
            pass

    print("  (nrun=%d)" % nrun)

    return res


# ==============================================================================
# Benchmark Runners
# ==============================================================================


def benchmark_faiss_ivf(
    xb: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    k: int,
    nlist: int,
    nprobes: List[int],
    min_time: float,
    inter: bool = False,
) -> dict:
    """Benchmark Faiss IVF Flat index."""
    if faiss is None:
        raise RuntimeError("Faiss required for faiss backend")

    n, d = xb.shape

    print(f"\n{'='*60}")
    print(f"Benchmarking Faiss IVF (n={n}, d={d}, nlist={nlist})")
    print(f"{'='*60}")

    # Build index
    print("Building Faiss IVF index...")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    t0 = time.time()
    index.train(xb)
    index.add(xb)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")

    results = {
        "backend": "faiss",
        "n": n,
        "d": d,
        "nlist": nlist,
        "k": k,
        "build_time_s": build_time,
        "search_results": {},
    }

    # Benchmark each nprobe
    print(f"\n{'nprobe':<10} {'Recalls':<40} {'Timing':<30}")
    print("-" * 80)

    for nprobe in nprobes:
        index.nprobe = nprobe
        print(f"nprobe={nprobe:<4}", end="  ")
        sys.stdout.flush()

        res = eval_setting(index, xq, gt, k, inter, min_time)
        results["search_results"][f"nprobe={nprobe}"] = res

    return results


def benchmark_vector_indexer(
    xb: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    k: int,
    nprobes: List[int],
    min_time: float,
    work_dir: Optional[str] = None,
    inter: bool = False,
) -> dict:
    """Benchmark vector_indexer via adapter."""
    from vector_indexer_adapter import VectorIndexerFaissAdapter
    import vector_indexer_py

    n, d = xb.shape
    nlist = vector_indexer_py.suggest_nlist(n)

    print(f"\n{'='*60}")
    print(f"Benchmarking vector_indexer (n={n}, d={d}, nlist={nlist})")
    print(f"{'='*60}")

    # Build index
    print("Building vector_indexer index...")
    t0 = time.time()
    idx = vector_indexer_py.build(xb, work_dir)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")

    # Create adapter
    adapter = VectorIndexerFaissAdapter(idx, k)

    results = {
        "backend": "vector_indexer",
        "n": n,
        "d": d,
        "nlist": nlist,
        "k": k,
        "build_time_s": build_time,
        "search_results": {},
    }

    # Benchmark each nprobe
    print(f"\n{'nprobe':<10} {'Recalls':<40} {'Timing':<30}")
    print("-" * 80)

    for nprobe in nprobes:
        adapter.nprobe = nprobe
        print(f"nprobe={nprobe:<4}", end="  ")
        sys.stdout.flush()

        res = eval_setting(adapter, xq, gt, k, inter, min_time)
        results["search_results"][f"nprobe={nprobe}"] = res

    return results


# ==============================================================================
# Results Output
# ==============================================================================


def save_results(all_results: List[dict], output_dir: Path):
    """Save benchmark results to JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / "faiss_bench_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Markdown summary
    md_path = output_dir / "faiss_bench_results.md"
    with open(md_path, "w") as f:
        f.write("# IVF Benchmark Results (Official Faiss Methodology)\n\n")
        f.write("Uses official Faiss eval_setting():\n")
        f.write("- Runs until min_test_duration, averages timing\n")
        f.write("- R@1 = fraction of queries where true NN is rank 1\n")
        f.write("- R@10 = fraction of queries where true NN is in top 10\n\n")

        for result in all_results:
            f.write(f"## {result['backend']}\n\n")
            f.write(
                f"- n={result['n']}, d={result['d']}, nlist={result['nlist']}, k={result['k']}\n"
            )
            f.write(f"- Build time: {result['build_time_s']:.2f}s\n\n")

            f.write("| nprobe | R@1 | R@10 | R@100 | ms/query | QPS |\n")
            f.write("|--------|-----|------|-------|----------|-----|\n")

            for key, res in result["search_results"].items():
                nprobe = key.replace("nprobe=", "")
                recalls = res.get("recalls", {})
                r1 = recalls.get(1, "-")
                r10 = recalls.get(10, "-")
                r100 = recalls.get(100, "-")
                if isinstance(r1, float):
                    r1 = f"{r1:.4f}"
                if isinstance(r10, float):
                    r10 = f"{r10:.4f}"
                if isinstance(r100, float):
                    r100 = f"{r100:.4f}"

                f.write(
                    f"| {nprobe} | {r1} | {r10} | {r100} | {res['ms_per_query']:.3f} | {res['qps']:.1f} |\n"
                )
            f.write("\n")

    print(f"Results saved to {md_path}")


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Official Faiss-style benchmark for IVF indexes"
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["vector_indexer", "faiss", "both"],
        default="both",
        help="Backend to benchmark",
    )

    # Dataset source selection
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "local_npy"],
        default="synthetic",
        help="Dataset source: 'synthetic' (generate random data) or 'local_npy' (load from .npy files)",
    )
    parser.add_argument(
        "--xb-path",
        type=str,
        default=None,
        help="Path to database vectors .npy file (required for local_npy dataset)",
    )
    parser.add_argument(
        "--xq-path",
        type=str,
        default=None,
        help="Path to query vectors .npy file (required for local_npy dataset)",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Path to ground truth indices .npy file (optional; if omitted, computed with FlatL2)",
    )

    # Dataset parameters
    parser.add_argument(
        "--n",
        type=int,
        default=100000,
        help="Number of database vectors (or max to use from local dataset)",
    )
    parser.add_argument("--d", type=int, default=128, help="Vector dimension")
    parser.add_argument("--nq", type=int, default=1000, help="Number of query vectors")
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of neighbors (default 100 for R@1,10,100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Search parameters
    parser.add_argument(
        "--nprobes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated nprobe values",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="Number of clusters for Faiss (default: computed from n)",
    )

    # Benchmark parameters (official Faiss defaults)
    parser.add_argument(
        "--min_test_duration",
        type=float,
        default=3.0,
        help="Minimum test duration in seconds (official Faiss default)",
    )
    parser.add_argument(
        "--inter",
        action="store_true",
        default=False,
        help="Use intersection measure instead of recall",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./faiss_bench_official_results",
        help="Output directory",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None, help="Work directory for vector_indexer"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output JSON results to stdout",
    )

    args = parser.parse_args()

    # Validate local_npy arguments
    if args.dataset == "local_npy":
        if not args.xb_path:
            parser.error("--xb-path is required when --dataset=local_npy")
        if not args.xq_path:
            parser.error("--xq-path is required when --dataset=local_npy")

    nprobes = [int(x.strip()) for x in args.nprobes.split(",")]
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("IVF Benchmark (Official Faiss Methodology)")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Dataset source: {args.dataset}")
    if args.dataset == "synthetic":
        print(f"Dataset params: n={args.n}, d={args.d}, nq={args.nq}, k={args.k}")
    else:
        print(f"Dataset params: n<={args.n}, nq<={args.nq}, k={args.k}")
        print(f"  xb: {args.xb_path}")
        print(f"  xq: {args.xq_path}")
        if args.gt_path:
            print(f"  gt: {args.gt_path}")
    print(f"nprobes: {nprobes}")
    print(f"min_test_duration: {args.min_test_duration}s")
    print("=" * 60)

    # Load or generate dataset
    if args.dataset == "synthetic":
        xb, xq, gt = generate_synthetic_data(args.n, args.d, args.nq, args.k, args.seed)
    else:  # local_npy
        xb, xq, gt = load_local_npy_data(
            args.xb_path, args.xq_path, args.gt_path, args.n, args.nq, args.k
        )

    print(f"Dataset ready: xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")

    all_results = []

    # Use actual n from loaded data (may be smaller than args.n for local datasets)
    n_actual = xb.shape[0]

    # Compute nlist if not provided
    if args.nlist is None:
        try:
            import vector_indexer_py

            nlist = vector_indexer_py.suggest_nlist(n_actual)
        except ImportError:
            # Fallback formula
            if n_actual < 10_000:
                nlist = int(n_actual**0.5)
            elif n_actual < 100_000:
                nlist = int(2 * (n_actual**0.5))
            else:
                nlist = int(4 * (n_actual**0.5))
    else:
        nlist = args.nlist

    # Benchmark vector_indexer
    if args.backend in ("vector_indexer", "both"):
        try:
            result = benchmark_vector_indexer(
                xb,
                xq,
                gt,
                args.k,
                nprobes,
                args.min_test_duration,
                args.work_dir,
                args.inter,
            )
            all_results.append(result)
        except Exception as e:
            print(f"vector_indexer benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Benchmark Faiss
    if args.backend in ("faiss", "both"):
        try:
            result = benchmark_faiss_ivf(
                xb, xq, gt, args.k, nlist, nprobes, args.min_test_duration, args.inter
            )
            all_results.append(result)
        except Exception as e:
            print(f"Faiss benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    if all_results:
        save_results(all_results, output_dir)

        if args.json:
            print("\nJSON results:")
            print(json.dumps(all_results, indent=2))

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
