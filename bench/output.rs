use crate::config::BenchmarkRun;
use crate::workloads::{
    BuildResult, LatencyResult, RecallResult, ThroughputResult,
};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Complete benchmark result for a single run
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub vector_count: usize,
    pub dimension: u32,
    pub actual_nlist: usize,
    pub nprobe: usize,
    pub concurrency: usize,
    pub k: usize,
    pub queries: usize,
    pub seed: u64,
    // Build metrics
    pub build_time_sec: f64,
    pub memory_bytes: u64,
    pub index_size_bytes: u64,
    pub vector_file_size_bytes: u64,
    // Latency metrics
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub mean_ms: Option<f64>,
    // Throughput metrics
    pub total_queries: Option<usize>,
    pub duration_sec: Option<f64>,
    pub qps: Option<f64>,
    pub p95_latency_under_load_ms: Option<f64>,
    // Recall metrics
    pub recall_at_k: Option<f64>,
}

impl BenchmarkResult {
    pub fn from_run_and_results(
        run: &BenchmarkRun,
        build: &BuildResult,
        latency: Option<&LatencyResult>,
        throughput: Option<&ThroughputResult>,
        recall: Option<&RecallResult>,
    ) -> Self {
        BenchmarkResult {
            vector_count: run.vector_count,
            dimension: run.dimension,
            actual_nlist: build.actual_nlist,
            nprobe: run.nprobe,
            concurrency: run.concurrency,
            k: run.k,
            queries: run.queries,
            seed: run.seed,
            build_time_sec: build.build_time_sec,
            memory_bytes: build.memory_bytes,
            index_size_bytes: build.index_size_bytes,
            vector_file_size_bytes: build.vector_file_size_bytes,
            p50_ms: latency.map(|l| l.p50_ms),
            p95_ms: latency.map(|l| l.p95_ms),
            p99_ms: latency.map(|l| l.p99_ms),
            mean_ms: latency.map(|l| l.mean_ms),
            total_queries: throughput.map(|t| t.total_queries),
            duration_sec: throughput.map(|t| t.duration_sec),
            qps: throughput.map(|t| t.qps),
            p95_latency_under_load_ms: throughput.map(|t| t.p95_latency_under_load_ms),
            recall_at_k: recall.map(|r| r.recall_at_k),
        }
    }
}

/// Write results to JSON file
pub fn write_json(
    results: &[BenchmarkResult],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Writing JSON results to {:?}", output_path);

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(results)?;
    let mut file = File::create(output_path)?;
    file.write_all(json.as_bytes())?;

    eprintln!("Wrote {} results to JSON", results.len());
    Ok(())
}

/// Write results to CSV file
pub fn write_csv(
    results: &[BenchmarkResult],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Writing CSV results to {:?}", output_path);

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(output_path)?;

    // Write header
    writeln!(
        file,
        "vector_count,dimension,actual_nlist,nprobe,concurrency,k,queries,seed,build_time_sec,memory_bytes,index_size_bytes,vector_file_size_bytes,p50_ms,p95_ms,p99_ms,mean_ms,total_queries,duration_sec,qps,p95_latency_under_load_ms,recall_at_k"
    )?;

    // Write data rows
    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            result.vector_count,
            result.dimension,
            result.actual_nlist,
            result.nprobe,
            result.concurrency,
            result.k,
            result.queries,
            result.seed,
            result.build_time_sec,
            result.memory_bytes,
            result.index_size_bytes,
            result.vector_file_size_bytes,
            result.p50_ms.map(|v| v.to_string()).unwrap_or_default(),
            result.p95_ms.map(|v| v.to_string()).unwrap_or_default(),
            result.p99_ms.map(|v| v.to_string()).unwrap_or_default(),
            result.mean_ms.map(|v| v.to_string()).unwrap_or_default(),
            result.total_queries.map(|v| v.to_string()).unwrap_or_default(),
            result.duration_sec.map(|v| v.to_string()).unwrap_or_default(),
            result.qps.map(|v| v.to_string()).unwrap_or_default(),
            result.p95_latency_under_load_ms
                .map(|v| v.to_string())
                .unwrap_or_default(),
            result.recall_at_k.map(|v| v.to_string()).unwrap_or_default(),
        )?;
    }

    eprintln!("Wrote {} results to CSV", results.len());
    Ok(())
}

/// Write results to Markdown file
pub fn write_markdown(
    results: &[BenchmarkResult],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Writing Markdown results to {:?}", output_path);

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(output_path)?;

    writeln!(file, "# Benchmark Results\n")?;

    // Group results by (vector_count, dimension)
    let mut grouped: HashMap<(usize, u32), Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        let key = (result.vector_count, result.dimension);
        grouped.entry(key).or_insert_with(Vec::new).push(result);
    }

    // Sort groups by vector_count, then dimension
    let mut groups: Vec<_> = grouped.into_iter().collect();
    groups.sort_by_key(|((vc, dim), _)| (*vc, *dim));

    for ((vector_count, dimension), group_results) in groups {
        writeln!(
            file,
            "## Dataset: {}k vectors, dim={}\n",
            vector_count / 1000,
            dimension
        )?;

        // Latency table
        writeln!(file, "### Latency Metrics\n")?;
        writeln!(
            file,
            "| nprobe | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |"
        )?;
        writeln!(file, "|--------|----------|----------|----------|-----------|")?;

        // Group by nprobe for latency (use first concurrency level)
        let mut latency_by_nprobe: HashMap<usize, &BenchmarkResult> = HashMap::new();
        for result in &group_results {
            if result.p50_ms.is_some() {
                latency_by_nprobe
                    .entry(result.nprobe)
                    .or_insert_with(|| *result);
            }
        }

        let mut nprobes: Vec<_> = latency_by_nprobe.keys().copied().collect();
        nprobes.sort();
        for nprobe in nprobes {
            let result = latency_by_nprobe[&nprobe];
            writeln!(
                file,
                "| {} | {:.2} | {:.2} | {:.2} | {:.2} |",
                nprobe,
                result.p50_ms.unwrap_or(0.0),
                result.p95_ms.unwrap_or(0.0),
                result.p99_ms.unwrap_or(0.0),
                result.mean_ms.unwrap_or(0.0),
            )?;
        }

        writeln!(file)?;

        // Throughput table
        writeln!(file, "### Throughput Metrics\n")?;
        writeln!(file, "| Concurrency | QPS | p95 Latency (ms) |")?;
        writeln!(file, "|-------------|-----|------------------|")?;

        // Group by concurrency for throughput (use first nprobe)
        let mut throughput_by_concurrency: HashMap<usize, &BenchmarkResult> = HashMap::new();
        for result in &group_results {
            if result.qps.is_some() {
                throughput_by_concurrency
                    .entry(result.concurrency)
                    .or_insert_with(|| *result);
            }
        }

        let mut concurrencies: Vec<_> = throughput_by_concurrency.keys().copied().collect();
        concurrencies.sort();
        for concurrency in concurrencies {
            let result = throughput_by_concurrency[&concurrency];
            writeln!(
                file,
                "| {} | {:.0} | {:.2} |",
                concurrency,
                result.qps.unwrap_or(0.0),
                result.p95_latency_under_load_ms.unwrap_or(0.0),
            )?;
        }

        writeln!(file)?;

        // Recall table
        writeln!(file, "### Recall Metrics\n")?;
        writeln!(file, "| nprobe | Recall@{} |", group_results[0].k)?;
        writeln!(file, "|--------|------------|")?;

        // Group by nprobe for recall
        let mut recall_by_nprobe: HashMap<usize, &BenchmarkResult> = HashMap::new();
        for result in &group_results {
            if result.recall_at_k.is_some() && result.recall_at_k.unwrap() > 0.0 {
                recall_by_nprobe.entry(result.nprobe).or_insert_with(|| *result);
            }
        }

        let mut nprobes: Vec<_> = recall_by_nprobe.keys().copied().collect();
        nprobes.sort();
        for nprobe in nprobes {
            let result = recall_by_nprobe[&nprobe];
            writeln!(
                file,
                "| {} | {:.3} |",
                nprobe,
                result.recall_at_k.unwrap_or(0.0),
            )?;
        }

        writeln!(file)?;

        // Build metrics summary
        if let Some(first_result) = group_results.first() {
            writeln!(file, "### Build Metrics\n")?;
            writeln!(
                file,
                "| Metric | Value |"
            )?;
            writeln!(file, "|--------|-------|")?;
            writeln!(
                file,
                "| Build Time | {:.2} s |",
                first_result.build_time_sec
            )?;
            writeln!(
                file,
                "| Memory Usage | {:.2} MB |",
                first_result.memory_bytes as f64 / 1_000_000.0
            )?;
            writeln!(
                file,
                "| Index Size | {:.2} MB |",
                first_result.index_size_bytes as f64 / 1_000_000.0
            )?;
            writeln!(
                file,
                "| Actual nlist | {} |",
                first_result.actual_nlist
            )?;
            writeln!(file)?;
        }
    }

    eprintln!("Wrote {} results to Markdown", results.len());
    Ok(())
}

