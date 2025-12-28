use crate::config::BenchmarkRun;
use crate::dataset::{ensure_dataset, get_file_size, load_dataset};
use crate::metrics::{measure_rss, LatencyHistogram};
use vector_indexer::ivf_index::{load_index_from, IvfIndex};
use vector_indexer::utils::{calculate_num_clusters, euclidean_distance_squared};
use vector_indexer::vector_store::VectorStore;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Result from build benchmark
#[derive(Debug, Clone)]
pub struct BuildResult {
    pub build_time_sec: f64,
    pub memory_bytes: u64,
    pub index_size_bytes: u64,
    pub vector_file_size_bytes: u64,
    pub actual_nlist: usize,
}

/// Result from latency benchmark
#[derive(Debug, Clone)]
pub struct LatencyResult {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
}

/// Result from throughput benchmark
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    pub total_queries: usize,
    pub duration_sec: f64,
    pub qps: f64,
    pub p95_latency_under_load_ms: f64,
}

/// Result from recall benchmark
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub recall_at_k: f64,
}

/// Get a unique run ID for this benchmark configuration
fn run_id(run: &BenchmarkRun) -> String {
    format!(
        "{}_{}_{}_{}_{}",
        run.vector_count, run.dimension, run.nprobe, run.concurrency, run.seed
    )
}

/// Get index directory path for a run
fn index_dir(run: &BenchmarkRun) -> PathBuf {
    PathBuf::from("bench_index").join(run_id(run))
}

/// Get shards directory path for a run
fn shards_dir(run: &BenchmarkRun) -> PathBuf {
    index_dir(run).join("shards")
}

/// Build the IVF index and measure build time, memory, and sizes
pub fn build_benchmark(run: &BenchmarkRun) -> Result<BuildResult, Box<dyn std::error::Error>> {
    eprintln!("Building index for run: {:?}", run_id(run));

    // Ensure dataset exists
    let dataset_path = ensure_dataset(run.vector_count, run.dimension, run.seed)?;
    let vector_file_size = get_file_size(&dataset_path)?;

    // Load vectors
    let vectors = load_dataset(&dataset_path)?;
    let store = VectorStore::new(vectors);

    // Calculate expected nlist (for recording)
    let expected_nlist = calculate_num_clusters(run.vector_count);

    // Measure memory before
    let memory_before = measure_rss().unwrap_or(0);

    // Build index
    let index_dir_path = index_dir(run);
    let shards_dir_path = shards_dir(run);
    std::fs::create_dir_all(&shards_dir_path)?;

    let start = Instant::now();
    let mut index = IvfIndex::new(run.dimension);
    index.fit_with_paths(&store, &shards_dir_path);
    let build_time = start.elapsed();

    // Measure memory after
    let memory_after = measure_rss().unwrap_or(0);
    let memory_delta = memory_after.saturating_sub(memory_before);

    // Save index
    index.save_to(&index_dir_path)?;

    // Calculate index size (index.bin + all shard files)
    let index_file_size = get_file_size(&index_dir_path.join("index.bin")).unwrap_or(0);
    let mut shard_size_total = 0u64;
    if shards_dir_path.exists() {
        for entry in std::fs::read_dir(&shards_dir_path)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with("shard_") {
                shard_size_total += get_file_size(&entry.path()).unwrap_or(0);
            }
        }
    }
    let total_index_size = index_file_size + shard_size_total;

    // Get actual nlist (number of centroids in the index)
    // We can't access centroids directly, so we use the calculated value
    // In practice, some centroids might be empty and filtered out
    let actual_nlist = expected_nlist; // This is approximate, actual might be less due to filtering

    Ok(BuildResult {
        build_time_sec: build_time.as_secs_f64(),
        memory_bytes: memory_delta,
        index_size_bytes: total_index_size,
        vector_file_size_bytes: vector_file_size,
        actual_nlist,
    })
}

/// Run latency benchmark: sequential queries with per-query timing
pub fn latency_benchmark(
    run: &BenchmarkRun,
    query_vectors: &[Vec<f32>],
) -> Result<LatencyResult, Box<dyn std::error::Error>> {
    eprintln!("Running latency benchmark with {} queries", query_vectors.len());

    // Load index
    let index_dir_path = index_dir(run);
    let shards_dir_path = shards_dir(run);
    let index = load_index_from(&index_dir_path)?;

    // Warmup: run a few queries to warm OS cache
    let warmup_count = 10.min(query_vectors.len());
    for query in query_vectors.iter().take(warmup_count) {
        let _ = index.search_with_paths(query, run.k, run.nprobe, &shards_dir_path);
    }

    // Measure latency for each query
    let mut histogram = LatencyHistogram::new();
    for query in query_vectors {
        let start = Instant::now();
        let _ = index.search_with_paths(query, run.k, run.nprobe, &shards_dir_path)?;
        let elapsed = start.elapsed();
        histogram.record_duration(elapsed);
    }

    Ok(LatencyResult {
        p50_ms: histogram.p50().unwrap_or(0.0),
        p95_ms: histogram.p95().unwrap_or(0.0),
        p99_ms: histogram.p99().unwrap_or(0.0),
        mean_ms: histogram.mean().unwrap_or(0.0),
    })
}

/// Run throughput benchmark: time-boxed execution with thread pool
pub fn throughput_benchmark(
    run: &BenchmarkRun,
    query_vectors: &[Vec<f32>],
) -> Result<ThroughputResult, Box<dyn std::error::Error>> {
    eprintln!(
        "Running throughput benchmark: {} threads, {} seconds",
        run.concurrency, run.duration_sec
    );

    // Load index
    let index_dir_path = index_dir(run);
    let shards_dir_path = shards_dir(run);
    let index = Arc::new(load_index_from(&index_dir_path)?);

    // Prepare query vectors (cycle through them)
    let queries = Arc::new(query_vectors.to_vec());

    // Barrier for synchronized start
    let barrier = Arc::new(Barrier::new(run.concurrency));

    // Thread handles and result channels
    let mut handles = Vec::new();
    let (tx, rx) = std::sync::mpsc::channel();

    // Spawn worker threads
    let duration_sec = run.duration_sec;
    for _ in 0..run.concurrency {
        let index = Arc::clone(&index);
        let queries = Arc::clone(&queries);
        let barrier = Arc::clone(&barrier);
        let tx = tx.clone();
        let shards_dir = shards_dir_path.clone();
        let k = run.k;
        let nprobe = run.nprobe;
        let duration_sec = duration_sec; // Copy value

        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            barrier.wait();

            let start_time = Instant::now();
            let deadline = start_time + Duration::from_secs(duration_sec);
            let mut query_count = 0usize;
            let mut latencies = Vec::new();
            let mut query_idx = 0usize;

            // Run queries until deadline
            while Instant::now() < deadline {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;

                let query_start = Instant::now();
                let _ = index.search_with_paths(query, k, nprobe, &shards_dir);
                let query_elapsed = query_start.elapsed();

                query_count += 1;
                latencies.push(query_elapsed.as_secs_f64() * 1000.0); // Convert to ms
            }

            let actual_duration = start_time.elapsed();

            // Send results
            tx.send((query_count, actual_duration, latencies)).unwrap();
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    drop(tx);

    // Collect results from all threads
    let mut total_queries = 0usize;
    let mut all_latencies = Vec::new();
    let mut max_duration = Duration::ZERO;

    while let Ok((count, duration, latencies)) = rx.try_recv() {
        total_queries += count;
        all_latencies.extend(latencies);
        if duration > max_duration {
            max_duration = duration;
        }
    }

    // Calculate QPS
    let duration_sec = max_duration.as_secs_f64();
    let qps = total_queries as f64 / duration_sec;

    // Calculate p95 latency under load
    all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = (all_latencies.len() as f64 * 0.95).ceil() as usize - 1;
    let p95_idx = p95_idx.min(all_latencies.len().saturating_sub(1));
    let p95_latency = all_latencies.get(p95_idx).copied().unwrap_or(0.0);

    Ok(ThroughputResult {
        total_queries,
        duration_sec,
        qps,
        p95_latency_under_load_ms: p95_latency,
    })
}

/// Run recall benchmark: compare IVF results with brute-force ground truth
pub fn recall_benchmark(
    run: &BenchmarkRun,
    query_vectors: &[Vec<f32>],
    dataset_vectors: &[(u64, Vec<f32>, u64)],
) -> Result<RecallResult, Box<dyn std::error::Error>> {
    eprintln!("Running recall benchmark with {} queries", query_vectors.len());

    // Only run for small datasets
    if run.vector_count > 100_000 {
        eprintln!("Skipping recall benchmark for large dataset ({} vectors)", run.vector_count);
        return Ok(RecallResult { recall_at_k: 0.0 });
    }

    // Load index
    let index_dir_path = index_dir(run);
    let shards_dir_path = shards_dir(run);
    let index = load_index_from(&index_dir_path)?;

    // Sample queries (use first 1000 or all if less)
    let sample_size = 1000.min(query_vectors.len());
    let sample_queries = &query_vectors[..sample_size];

    let mut total_recall = 0.0;
    let mut query_count = 0;

    for query in sample_queries {
        // Get IVF results
        let ivf_results = index.search_with_paths(query, run.k, run.nprobe, &shards_dir_path)?;
        let ivf_ids: std::collections::HashSet<usize> =
            ivf_results.iter().map(|(id, _, _)| *id).collect();

        // Compute brute-force ground truth
        let mut distances: Vec<(usize, f32)> = dataset_vectors
            .iter()
            .enumerate()
            .map(|(idx, (_, vec, _))| {
                let dist = euclidean_distance_squared(query, vec);
                (idx, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth_ids: std::collections::HashSet<usize> = distances
            .iter()
            .take(run.k)
            .map(|(idx, _)| *idx)
            .collect();

        // Calculate recall@k: intersection size / k
        let intersection = ivf_ids.intersection(&ground_truth_ids).count();
        let recall = intersection as f64 / run.k as f64;
        total_recall += recall;
        query_count += 1;
    }

    let avg_recall = if query_count > 0 {
        total_recall / query_count as f64
    } else {
        0.0
    };

    Ok(RecallResult {
        recall_at_k: avg_recall,
    })
}

/// Generate query vectors deterministically from the dataset
pub fn generate_query_vectors(
    dataset_vectors: &[(u64, Vec<f32>, u64)],
    count: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut queries = Vec::with_capacity(count);

    for _ in 0..count {
        let idx = rng.gen_range(0..dataset_vectors.len());
        queries.push(dataset_vectors[idx].1.clone());
    }

    queries
}

