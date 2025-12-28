use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub dataset: DatasetConfig,
    pub ivf: IvfConfig,
    pub queries: usize,
    pub k: usize,
    pub concurrency: Vec<usize>,
    pub duration_sec: u64,
    pub seed: u64,
    pub output_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub dimensions: Vec<u32>,
    pub vector_counts: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    pub nprobe: Vec<usize>,
}

/// A single benchmark run configuration (one parameter combination)
#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    pub dimension: u32,
    pub vector_count: usize,
    pub nprobe: usize,
    pub concurrency: usize,
    pub k: usize,
    pub queries: usize,
    pub seed: u64,
    pub duration_sec: u64,
    pub output_prefix: String,
}

impl BenchmarkConfig {
    /// Load config from YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: BenchmarkConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Expand all parameter combinations into individual BenchmarkRun structs
    pub fn expand_combinations(&self) -> Vec<BenchmarkRun> {
        let mut runs = Vec::new();

        for &dimension in &self.dataset.dimensions {
            for &vector_count in &self.dataset.vector_counts {
                for &nprobe in &self.ivf.nprobe {
                    for &concurrency in &self.concurrency {
                        runs.push(BenchmarkRun {
                            dimension,
                            vector_count,
                            nprobe,
                            concurrency,
                            k: self.k,
                            queries: self.queries,
                            seed: self.seed,
                            duration_sec: self.duration_sec,
                            output_prefix: self.output_prefix.clone(),
                        });
                    }
                }
            }
        }

        runs
    }
}

