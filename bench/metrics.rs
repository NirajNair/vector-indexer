use std::time::Duration;

/// Simple latency histogram for computing percentiles
pub struct LatencyHistogram {
    latencies: Vec<f64>, // Latencies in milliseconds
}

impl LatencyHistogram {
    pub fn new() -> Self {
        LatencyHistogram {
            latencies: Vec::new(),
        }
    }

    /// Record a latency measurement (in milliseconds)
    pub fn record(&mut self, latency_ms: f64) {
        self.latencies.push(latency_ms);
    }

    /// Record a duration measurement
    pub fn record_duration(&mut self, duration: Duration) {
        let ms = duration.as_secs_f64() * 1000.0;
        self.record(ms);
    }

    /// Get the number of samples
    pub fn count(&self) -> usize {
        self.latencies.len()
    }

    /// Compute percentile (0.0 to 1.0, e.g., 0.5 for p50, 0.95 for p95)
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.latencies.is_empty() {
            return None;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (sorted.len() as f64 * p).ceil() as usize - 1;
        let index = index.min(sorted.len() - 1);
        Some(sorted[index])
    }

    /// Get p50 (median) latency in milliseconds
    pub fn p50(&self) -> Option<f64> {
        self.percentile(0.5)
    }

    /// Get p95 latency in milliseconds
    pub fn p95(&self) -> Option<f64> {
        self.percentile(0.95)
    }

    /// Get p99 latency in milliseconds
    pub fn p99(&self) -> Option<f64> {
        self.percentile(0.99)
    }

    /// Get mean latency in milliseconds
    pub fn mean(&self) -> Option<f64> {
        if self.latencies.is_empty() {
            return None;
        }
        let sum: f64 = self.latencies.iter().sum();
        Some(sum / self.latencies.len() as f64)
    }

    /// Get all latencies (for external analysis)
    pub fn latencies(&self) -> &[f64] {
        &self.latencies
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Measure RSS (Resident Set Size) memory usage in bytes
pub fn measure_rss() -> Result<u64, Box<dyn std::error::Error>> {
    use std::fs;
    let status = fs::read_to_string("/proc/self/status")?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1].parse()?;
                return Ok(kb * 1024); // Convert KB to bytes
            }
        }
    }
    Err("VmRSS not found in /proc/self/status".into())
}

/// Compute basic statistics from a slice of values
pub fn compute_stats(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;

    let variance: f64 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    (mean, std_dev, max - min) // mean, std_dev, range
}
