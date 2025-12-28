mod config;
mod dataset;
mod metrics;
mod output;
mod workloads;

use config::BenchmarkConfig;
use dataset::load_dataset;
use output::{write_csv, write_json, write_markdown, BenchmarkResult};
use std::env;
use std::path::PathBuf;
use workloads::{
    build_benchmark, generate_query_vectors, latency_benchmark, recall_benchmark,
    throughput_benchmark,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <config.yaml>", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[1];
    eprintln!("Loading config from: {}", config_path);

    // Load and parse config
    let config = BenchmarkConfig::from_file(config_path)?;
    eprintln!("Loaded config: {} parameter combinations", {
        config.dataset.dimensions.len()
            * config.dataset.vector_counts.len()
            * config.ivf.nprobe.len()
            * config.concurrency.len()
    });

    // Expand parameter combinations
    let runs = config.expand_combinations();
    eprintln!("Expanded to {} benchmark runs", runs.len());

    // Create output directory structure
    std::fs::create_dir_all("bench_data")?;
    std::fs::create_dir_all("bench_index")?;

    // Run benchmarks
    let mut results = Vec::new();

    for (idx, run) in runs.iter().enumerate() {
        eprintln!(
            "\n=== Run {}/{}: {}k vectors, dim={}, nprobe={}, concurrency={} ===",
            idx + 1,
            runs.len(),
            run.vector_count / 1000,
            run.dimension,
            run.nprobe,
            run.concurrency
        );

        // Build benchmark
        eprintln!("[1/4] Building index...");
        let build_result = match build_benchmark(run) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("ERROR: Build benchmark failed: {}", e);
                continue;
            }
        };

        eprintln!(
            "  Build time: {:.2}s, Memory: {:.2}MB, nlist: {}",
            build_result.build_time_sec,
            build_result.memory_bytes as f64 / 1_000_000.0,
            build_result.actual_nlist
        );

        // Load dataset for queries
        let dataset_path = dataset::dataset_path(run.vector_count, run.dimension, run.seed);
        let dataset_vectors = load_dataset(&dataset_path)?;

        // Generate query vectors
        let query_vectors = generate_query_vectors(&dataset_vectors, run.queries, run.seed);

        // Latency benchmark
        eprintln!("[2/4] Running latency benchmark...");
        let latency_result = match latency_benchmark(run, &query_vectors) {
            Ok(result) => {
                eprintln!(
                    "  p50: {:.2}ms, p95: {:.2}ms, p99: {:.2}ms",
                    result.p50_ms, result.p95_ms, result.p99_ms
                );
                Some(result)
            }
            Err(e) => {
                eprintln!("ERROR: Latency benchmark failed: {}", e);
                None
            }
        };

        // Throughput benchmark
        eprintln!("[3/4] Running throughput benchmark...");
        let throughput_result = match throughput_benchmark(run, &query_vectors) {
            Ok(result) => {
                eprintln!(
                    "  QPS: {:.0}, p95 latency: {:.2}ms",
                    result.qps, result.p95_latency_under_load_ms
                );
                Some(result)
            }
            Err(e) => {
                eprintln!("ERROR: Throughput benchmark failed: {}", e);
                None
            }
        };

        // Recall benchmark (only for small datasets)
        eprintln!("[4/4] Running recall benchmark...");
        let recall_result = if run.vector_count <= 100_000 {
            match recall_benchmark(run, &query_vectors, &dataset_vectors) {
                Ok(result) => {
                    eprintln!("  Recall@{}: {:.3}", run.k, result.recall_at_k);
                    Some(result)
                }
                Err(e) => {
                    eprintln!("ERROR: Recall benchmark failed: {}", e);
                    None
                }
            }
        } else {
            eprintln!("  Skipped (dataset too large)");
            None
        };

        // Create result entry
        let result = BenchmarkResult::from_run_and_results(
            run,
            &build_result,
            latency_result.as_ref(),
            throughput_result.as_ref(),
            recall_result.as_ref(),
        );

        results.push(result);
    }

    eprintln!("\n=== Writing Results ===");

    // Write outputs
    let output_prefix = &config.output_prefix;
    let json_path = PathBuf::from(format!("{}.json", output_prefix));
    let csv_path = PathBuf::from(format!("{}.csv", output_prefix));
    let md_path = PathBuf::from(format!("{}.md", output_prefix));

    write_json(&results, &json_path)?;
    write_csv(&results, &csv_path)?;
    write_markdown(&results, &md_path)?;

    eprintln!("\n=== Benchmark Complete ===");
    eprintln!("Results written to:");
    eprintln!("  - {}", json_path.display());
    eprintln!("  - {}", csv_path.display());
    eprintln!("  - {}", md_path.display());

    Ok(())
}

