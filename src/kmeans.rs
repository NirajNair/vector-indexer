use ndarray::Array1;
use ndarray::Array2;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;
use std::io::{Error, ErrorKind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::vec;
use wide::{f32x4, f32x8};

/// Runs parallel + SIMD optimized K-Means++ clustering with early stopping.
pub fn run_kmeans_parallel(
    data: &Array2<f32>,                // shape: (n, dim)
    k: usize,                          // number of clusters
    max_iters: usize,                  // maximum iterations
    early_stop_threshold: Option<f32>, // early stop threshold
) -> Result<(Array2<f32>, Array1<usize>), Error> {
    let early_stop_threshold = early_stop_threshold.unwrap_or(1e-4);
    if data.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Input vectors cannot be empty",
        ));
    }
    let n = data.nrows();
    let dim = data.ncols();
    let mut rng = thread_rng();

    let mut curr_centroids = kmeans_plus_plus_init(&data, k);
    let mut labels = vec![0usize; n];

    for iter in 0..max_iters {
        println!("Iteration {iter}...");

        // Assignment (parallel + SIMD)
        assign_points_simd_parallel(&data, &curr_centroids, &mut labels);

        // Update (parallel reduction)
        let (mut new_centroids, counts) = update_centroids_parallel(&data, &labels, k, dim);

        // Handle empty clusters
        handle_empty_clusters(&mut new_centroids, &counts, &data, &mut rng);

        // Compute centroid movement (for early stopping)
        let delta = compute_centroid_delta(&new_centroids, &curr_centroids);
        println!("→ Centroid delta: {:.6}", delta);

        curr_centroids = new_centroids.clone();
        if delta < early_stop_threshold {
            println!("Converged early at iteration {}", iter + 1);
            break;
        }
    }

    Ok((curr_centroids, Array1::from_vec(labels)))
}

/// Runs mini-batch K-means clustering with SIMD optimizations and per-cluster learning rates.
/// Uses random batch sampling for faster convergence on large datasets.
pub fn run_kmeans_mini_batch(
    data: &Array2<f32>,                // shape: (n, dim)
    k: usize,                          // number of clusters
    max_iters: usize,                  // maximum iterations
    early_stop_threshold: Option<f32>, // early stop threshold
) -> Result<(Array2<f32>, Array1<usize>), Error> {
    let early_stop_threshold = early_stop_threshold.unwrap_or(1e-4);
    if data.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Input vectors cannot be empty",
        ));
    }
    let n = data.nrows();
    let dim = data.ncols();
    let mut rng = thread_rng();

    // Auto-calculate batch size
    let batch_size = std::cmp::min(256, std::cmp::max(10, (n as f32).sqrt() as usize));
    println!(
        "Mini-batch K-means: n={}, k={}, batch_size={}",
        n, k, batch_size
    );

    // Initialize centroids using K-means++
    let mut curr_centroids = kmeans_plus_plus_init(&data, k);
    let mut per_cluster_counts = vec![0usize; k];

    // Track previous centroids for convergence check
    let mut prev_centroids = curr_centroids.clone();

    for iter in 0..max_iters {
        println!("Mini-batch iteration {iter}...");

        // Sample random batch
        let batch_indices = sample_batch(n, batch_size, &mut rng);

        // Assign batch points to nearest centroids IN PARALLEL (across batch points)
        let batch_assignments: Vec<(usize, usize)> = batch_indices
            .par_iter()
            .map(|&idx| {
                let vector = data.row(idx);
                let (best_c, _) = find_nearest_centroid(&vector, &curr_centroids, dim);
                (idx, best_c)
            })
            .collect();

        // Create labels array and populate it
        let mut batch_labels = vec![0usize; n];
        for (idx, label) in batch_assignments {
            batch_labels[idx] = label;
        }

        // Update centroids using mini-batch with per-cluster learning rates
        update_centroids_mini_batch(
            &data,
            &mut curr_centroids,
            &batch_indices,
            &batch_labels,
            &mut per_cluster_counts,
            k,
            dim,
        );

        // Handle empty clusters (reinitialize from random data point)
        handle_empty_clusters(&mut curr_centroids, &per_cluster_counts, &data, &mut rng);

        // Compute centroid movement for convergence check (parallel reduction)
        let delta = compute_centroid_delta(&curr_centroids, &prev_centroids);
        println!("→ Centroid delta: {:.6}", delta);

        prev_centroids = curr_centroids.clone();

        if delta < early_stop_threshold {
            println!("Converged early at iteration {}", iter + 1);
            break;
        }
    }

    // Final assignment: assign all points to final centroids
    println!("Final assignment of all points...");
    let mut final_labels = vec![0usize; n];
    assign_points_simd_parallel(&data, &curr_centroids, &mut final_labels);

    Ok((curr_centroids, Array1::from_vec(final_labels)))
}

/// Fast approximate K-means++ using sampling for large datasets
/// Uses exact K-means++ for small datasets, switches to sampling for large ones
fn kmeans_plus_plus_init(data: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = data.nrows();

    // Use sampling-based approach for large datasets
    let sample_threshold = 50000;
    if n > sample_threshold {
        kmeans_plus_plus_init_sampled(data, k, sample_threshold)
    } else {
        kmeans_plus_plus_init_exact(data, k)
    }
}

/// Exact K-means++ initialization (for smaller datasets)
fn kmeans_plus_plus_init_exact(data: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = data.nrows();
    let dim = data.ncols();
    let mut rng = thread_rng();
    
    // If k >= n, use all data points as centroids and duplicate some if needed
    let actual_k = k.min(n);
    let mut centroids = Array2::<f32>::zeros((k, dim));

    // Choose first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&data.row(first_idx));
    println!("K-means++: Selected centroid 0 (point {})", first_idx);

    // Track minimum distance from each point to any centroid
    let mut min_distances = vec![f32::INFINITY; n];

    // Choose remaining centroids up to actual_k
    for i in 1..actual_k {
        // Update distances: for each point, find distance to nearest centroid
        update_min_distances_parallel(data, &centroids, i, &mut min_distances);

        // Choose next centroid with probability proportional to distance²
        let weights: Vec<f32> = min_distances.iter().map(|&d| d * d).collect();
        
        // Check if all weights are zero (all points already selected)
        let total_weight: f32 = weights.iter().sum();
        if total_weight == 0.0 {
            // All points have been selected, duplicate a random existing centroid
            let dup_idx = rng.gen_range(0..i);
            let dup_row = centroids.row(dup_idx).to_owned();
            centroids.row_mut(i).assign(&dup_row);
            println!("K-means++: Selected centroid {} (duplicate of centroid {})", i, dup_idx);
        } else {
            let dist = WeightedIndex::new(&weights).expect("Failed to create weighted distribution");
            let chosen_idx = dist.sample(&mut rng);

            centroids.row_mut(i).assign(&data.row(chosen_idx));
            println!(
                "K-means++: Selected centroid {} (point {}, dist²={:.2})",
                i, chosen_idx, min_distances[chosen_idx]
            );
        }
    }
    
    // If k > n, fill remaining centroids by duplicating existing ones
    for i in actual_k..k {
        let dup_idx = rng.gen_range(0..actual_k);
        let dup_row = centroids.row(dup_idx).to_owned();
        centroids.row_mut(i).assign(&dup_row);
        println!("K-means++: Selected centroid {} (duplicate of centroid {})", i, dup_idx);
    }

    centroids
}

/// Approximate K-means++ using sampling (for large datasets)
/// Much faster while maintaining good initialization quality
fn kmeans_plus_plus_init_sampled(data: &Array2<f32>, k: usize, sample_size: usize) -> Array2<f32> {
    let n = data.nrows();
    let dim = data.ncols();
    let mut rng = thread_rng();
    
    // If k >= n, use exact method instead
    let actual_k = k.min(n);
    let mut centroids = Array2::<f32>::zeros((k, dim));

    println!(
        "K-means++ (sampled): Using sample size {} for n={}",
        sample_size, n
    );

    // Choose first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&data.row(first_idx));
    println!("K-means++: Selected centroid 0 (point {})", first_idx);

    // Sample a random subset of points to use for distance calculations
    let mut sample_indices: Vec<usize> = (0..n).collect();
    sample_indices.shuffle(&mut rng);
    sample_indices.truncate(sample_size.min(n));
    let actual_sample_size = sample_indices.len();

    // Track minimum distances for sampled points only
    let mut min_distances = vec![f32::INFINITY; actual_sample_size];

    // Choose remaining centroids up to actual_k
    for i in 1..actual_k {
        // Update distances for sampled points in parallel
        update_min_distances_parallel(data, &centroids, i, &mut min_distances);

        // Choose next centroid from sampled points
        let weights: Vec<f32> = min_distances.iter().map(|&d| d * d).collect();
        
        // Check if all weights are zero
        let total_weight: f32 = weights.iter().sum();
        if total_weight == 0.0 {
            // All sampled points have been selected, duplicate a random existing centroid
            let dup_idx = rng.gen_range(0..i);
            let dup_row = centroids.row(dup_idx).to_owned();
            centroids.row_mut(i).assign(&dup_row);
            println!("K-means++: Selected centroid {} (duplicate of centroid {})", i, dup_idx);
        } else {
            let dist = WeightedIndex::new(&weights).expect("Failed to create weighted distribution");
            let sample_idx = dist.sample(&mut rng);
            let chosen_idx = sample_indices[sample_idx];

            centroids.row_mut(i).assign(&data.row(chosen_idx));
            println!(
                "K-means++: Selected centroid {} (point {}, dist²={:.2})",
                i, chosen_idx, min_distances[sample_idx]
            );
        }
    }
    
    // If k > n, fill remaining centroids by duplicating existing ones
    for i in actual_k..k {
        let dup_idx = rng.gen_range(0..actual_k);
        let dup_row = centroids.row(dup_idx).to_owned();
        centroids.row_mut(i).assign(&dup_row);
        println!("K-means++: Selected centroid {} (duplicate of centroid {})", i, dup_idx);
    }

    centroids
}

/// Reinitialize empty clusters with random data points
fn handle_empty_clusters(
    centroids: &mut Array2<f32>,
    counts: &[usize],
    data: &Array2<f32>,
    rng: &mut ThreadRng,
) {
    let k = centroids.nrows();
    let dim = centroids.ncols();
    let n = data.nrows();

    for c in 0..k {
        if counts[c] == 0 {
            let ri = rng.gen_range(0..n);
            for d in 0..dim {
                centroids[(c, d)] = data[(ri, d)];
            }
        }
    }
}

/// Compute the RMS change in centroids between iterations
fn compute_centroid_delta(curr: &Array2<f32>, prev: &Array2<f32>) -> f32 {
    let k = curr.nrows();
    let dim = curr.ncols();

    let delta_squared: f32 = (0..k)
        .into_par_iter()
        .map(|c| {
            let mut local_delta = 0.0;
            for d in 0..dim {
                let diff = curr[(c, d)] - prev[(c, d)];
                local_delta += diff * diff;
            }
            local_delta
        })
        .sum();

    (delta_squared / (k * dim) as f32).sqrt()
}

/// Find the nearest centroid to a point from a subset of centroids
#[inline]
fn find_nearest_centroid(
    point: &ndarray::ArrayView1<f32>,
    centroids: &Array2<f32>,
    dim: usize,
) -> (usize, f32) {
    let mut best_c = 0;
    let mut best_dist = f32::INFINITY;
    let centroids_len = centroids.nrows();

    for i in 0..centroids_len {
        let dist = compute_distance_simd(point, &centroids.row(i), dim);
        if dist < best_dist {
            best_dist = dist;
            best_c = i;
        }
    }

    (best_c, best_dist)
}

/// Compute distance between two points using SIMD (helper function)
#[inline]
fn compute_distance_simd(
    point: &ndarray::ArrayView1<f32>,
    centroid: &ndarray::ArrayView1<f32>,
    dim: usize,
) -> f32 {
    let p_slice = point.as_slice().unwrap();
    let c_slice = centroid.as_slice().unwrap();
    let mut j = 0;

    // SIMD 8-element chunks
    let mut acc8 = f32x8::splat(0.0);
    while j + 8 <= dim {
        let p_arr: [f32; 8] = p_slice[j..j + 8].try_into().unwrap();
        let c_arr: [f32; 8] = c_slice[j..j + 8].try_into().unwrap();
        let a = f32x8::from(p_arr);
        let b = f32x8::from(c_arr);
        let diff = a - b;
        acc8 += diff * diff;
        j += 8;
    }

    // SIMD 4-element chunks
    let mut acc4 = f32x4::splat(0.0);
    while j + 4 <= dim {
        let p_arr: [f32; 4] = p_slice[j..j + 4].try_into().unwrap();
        let c_arr: [f32; 4] = c_slice[j..j + 4].try_into().unwrap();
        let a = f32x4::from(p_arr);
        let b = f32x4::from(c_arr);
        let diff = a - b;
        acc4 += diff * diff;
        j += 4;
    }

    // Tail elements
    let mut tail = 0.0;
    while j < dim {
        let diff = p_slice[j] - c_slice[j];
        tail += diff * diff;
        j += 1;
    }

    acc8.reduce_add() + acc4.reduce_add() + tail
}

/// Update minimum distances to nearest centroid (parallelized with f32x8 SIMD)
fn update_min_distances_parallel(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
    num_centroids: usize,
    min_distances: &mut [f32],
) {
    let dim = data.ncols();
    let latest_centroid = centroids.row(num_centroids - 1);

    min_distances
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, min_dist)| {
            let point = data.row(i);
            let dist = compute_distance_simd(&point, &latest_centroid, dim);

            // Update minimum distance
            if dist < *min_dist {
                *min_dist = dist;
            }
        });
}

fn assign_points_simd_parallel(data: &Array2<f32>, centroids: &Array2<f32>, labels: &mut [usize]) {
    let k = centroids.nrows();

    // Use hierarchical assignment for large k (better performance)
    if k > 100 {
        assign_points_hierarchical(data, centroids, labels);
    } else {
        assign_points_brute_force(data, centroids, labels);
    }
}

/// Brute force assignment with SIMD (good for small k < 100)
fn assign_points_brute_force(data: &Array2<f32>, centroids: &Array2<f32>, labels: &mut [usize]) {
    let dim = data.ncols();

    labels.par_iter_mut().enumerate().for_each(|(i, label)| {
        let point = data.row(i);
        let (best_c, _) = find_nearest_centroid(&point, centroids, dim);
        *label = best_c;
    });
}

/// Hierarchical assignment for large k (much faster for k > 100)
/// Reduces O(k) comparisons to O(sqrt(k)) per point
fn assign_points_hierarchical(data: &Array2<f32>, centroids: &Array2<f32>, labels: &mut [usize]) {
    let k = centroids.nrows();
    let dim = centroids.ncols();
    let n = data.nrows();
    let meta_k = ((k as f32).sqrt() as usize).max(2).min(k / 2);

    println!(
        "→ Using hierarchical assignment: n={}, k={}, meta_k={}",
        n, k, meta_k
    );
    println!("  Building centroid hierarchy...");

    // Build hierarchy: cluster the centroids themselves
    let start = std::time::Instant::now();
    let (meta_centroids, centroid_to_meta) = build_centroid_hierarchy(centroids, meta_k);
    println!(
        "  Hierarchy built in {:?} ({} meta-centroids)",
        start.elapsed(),
        meta_k
    );

    // Count points per meta-cluster for stats
    let mut meta_cluster_sizes = vec![0usize; meta_k];
    for &meta_idx in &centroid_to_meta {
        meta_cluster_sizes[meta_idx] += 1;
    }
    let avg_cluster_size = k as f32 / meta_k as f32;
    println!(
        "  Avg centroids per meta-cluster: {:.1} (range: {}-{})",
        avg_cluster_size,
        meta_cluster_sizes.iter().min().unwrap(),
        meta_cluster_sizes.iter().max().unwrap()
    );

    // Pre-build inverted index: meta_cluster -> [centroid_indices]
    println!("  Building meta-cluster index...");
    let mut meta_to_centroids: Vec<Vec<usize>> = vec![Vec::new(); meta_k];
    for (c, &meta_idx) in centroid_to_meta.iter().enumerate() {
        meta_to_centroids[meta_idx].push(c);
    }
    println!("  Index built.");

    println!("  Assigning {} points in parallel...", n);
    let assign_start = std::time::Instant::now();

    // Create atomic counter for progress tracking
    let progress_counter = Arc::new(AtomicUsize::new(0));
    let progress_interval = (n / 10).max(1000); // Report every 10% or 1000 points

    // Assign each point using the hierarchy
    labels.par_iter_mut().enumerate().for_each(|(i, label)| {
        let point = data.row(i);

        // Find top-3 nearest meta-centroids (for accuracy)
        let top_k_check = 3.min(meta_k);
        let top_meta = find_top_k_meta_centroids(&point, &meta_centroids, top_k_check, dim);

        // Collect candidate centroid indices
        let candidate_indices: Vec<usize> = top_meta
            .iter()
            .flat_map(|&meta_idx| meta_to_centroids[meta_idx].iter().copied())
            .collect();

        let num_candidates = candidate_indices.len();

        // Build Array2 of candidate centroids
        let mut candidate_centroids = Array2::<f32>::zeros((num_candidates, dim));
        for (i, &c_idx) in candidate_indices.iter().enumerate() {
            candidate_centroids.row_mut(i).assign(&centroids.row(c_idx));
        }

        let (best_c, _) = find_nearest_centroid(&point, &candidate_centroids, dim);
        *label = candidate_indices[best_c];

        // Update progress counter and print occasionally
        let count = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;

        // Only print from one thread at progress intervals
        if count % progress_interval == 0 || count == n {
            let progress_pct = (count as f64 / n as f64) * 100.0;
            let elapsed = assign_start.elapsed();
            let rate = count as f64 / elapsed.as_secs_f64();
            let eta_secs = (n - count) as f64 / rate;

            println!(
                "    Progress: {}/{} ({:.1}%) | {:.0} pts/sec | avg {} centroids checked | ETA: {:.1}s",
                count, n, progress_pct, rate, candidate_centroids.nrows(), eta_secs
            );
        }
    });

    let assign_time = assign_start.elapsed();
    let throughput = n as f64 / assign_time.as_secs_f64();
    println!(
        "  Assignment completed in {:?} ({:.0} points/sec, {:.0}k points/sec)",
        assign_time,
        throughput,
        throughput / 1000.0
    );
}

/// Build a 2-level hierarchy by clustering the centroids
fn build_centroid_hierarchy(centroids: &Array2<f32>, meta_k: usize) -> (Array2<f32>, Vec<usize>) {
    let k = centroids.nrows();
    let dim = centroids.ncols();
    let mut rng = thread_rng();

    // Initialize meta-centroids randomly from centroids
    let mut meta_centroids = Array2::<f32>::zeros((meta_k, dim));
    let chosen: Vec<usize> = (0..k).choose_multiple(&mut rng, meta_k);
    for (i, &idx) in chosen.iter().enumerate() {
        meta_centroids.row_mut(i).assign(&centroids.row(idx));
    }

    // Run a few iterations of k-means on centroids to get meta-centroids
    let mut centroid_to_meta = vec![0usize; k];
    let num_iters = 5;

    for iter in 0..num_iters {
        // Assign centroids to meta-centroids
        for c in 0..k {
            let mut best_meta = 0;
            let mut best_dist = f32::INFINITY;

            for m in 0..meta_k {
                let dist = compute_distance_simd(&centroids.row(c), &meta_centroids.row(m), dim);
                if dist < best_dist {
                    best_dist = dist;
                    best_meta = m;
                }
            }
            centroid_to_meta[c] = best_meta;
        }

        // Update meta-centroids
        for m in 0..meta_k {
            let mut count = 0;
            let mut sum = vec![0.0f32; dim];

            for c in 0..k {
                if centroid_to_meta[c] == m {
                    count += 1;
                    for d in 0..dim {
                        sum[d] += centroids[(c, d)];
                    }
                }
            }

            if count > 0 {
                for d in 0..dim {
                    meta_centroids[(m, d)] = sum[d] / count as f32;
                }
            }
        }

        // Show progress for large k (only print every iteration for k > 1000)
        if k > 1000 {
            println!("    Hierarchy iteration {}/{}", iter + 1, num_iters);
        }
    }

    (meta_centroids, centroid_to_meta)
}

/// Find top-k nearest meta-centroids to a point
fn find_top_k_meta_centroids(
    point: &ndarray::ArrayView1<f32>,
    meta_centroids: &Array2<f32>,
    top_k: usize,
    dim: usize,
) -> Vec<usize> {
    let meta_k = meta_centroids.nrows();
    let mut distances: Vec<(usize, f32)> = (0..meta_k)
        .map(|m| {
            let dist = compute_distance_simd(point, &meta_centroids.row(m), dim);
            (m, dist)
        })
        .collect();

    // Partial sort to get top-k
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances
        .into_iter()
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

fn update_centroids_parallel(
    data: &Array2<f32>,
    labels: &[usize],
    k: usize,
    dim: usize,
) -> (Array2<f32>, Vec<usize>) {
    // Group points by cluster first (outside parallel section)
    let mut cluster_points: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &label) in labels.iter().enumerate() {
        cluster_points[label].push(i);
    }

    // Parallel compute per cluster (not per point!)
    let results: Vec<_> = (0..k)
        .into_par_iter()
        .map(|c| {
            let mut sum = vec![0.0f32; dim];
            let count = cluster_points[c].len();

            for &point_idx in &cluster_points[c] {
                for d in 0..dim {
                    sum[d] += data[(point_idx, d)];
                }
            }

            if count > 0 {
                for d in 0..dim {
                    sum[d] /= count as f32;
                }
            }
            (sum, count)
        })
        .collect();

    // Assemble results
    let mut new_centroids = Array2::<f32>::zeros((k, dim));
    let mut counts = vec![0usize; k];
    for (c, (centroid_vec, count)) in results.into_iter().enumerate() {
        for d in 0..dim {
            new_centroids[(c, d)] = centroid_vec[d];
        }
        counts[c] = count;
    }

    (new_centroids, counts)
}

/// Sample a random batch of indices without replacement
fn sample_batch(n: usize, batch_size: usize, rng: &mut ThreadRng) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices.into_iter().take(batch_size).collect()
}

/// Update centroids using mini-batch with per-cluster learning rates (parallelized)
fn update_centroids_mini_batch(
    data: &Array2<f32>,
    centroids: &mut Array2<f32>,
    batch_indices: &[usize],
    labels: &[usize],
    per_cluster_counts: &mut Vec<usize>,
    k: usize,
    dim: usize,
) {
    // Group batch points by cluster
    let mut cluster_points: Vec<Vec<usize>> = vec![Vec::new(); k];
    for &idx in batch_indices {
        cluster_points[labels[idx]].push(idx);
    }

    // Clone centroids for thread-safe reading
    let current_centroids = centroids.clone();
    let current_counts = per_cluster_counts.clone();

    // Parallel compute per cluster
    let centroid_updates: Vec<_> = (0..k)
        .into_par_iter()
        .map(|c| {
            let points = &cluster_points[c];
            if points.is_empty() {
                return (c, None);
            }

            let new_count = current_counts[c] + 1;
            let eta = 1.0 / new_count as f32;

            // Compute batch mean for this cluster
            let mut batch_sum = vec![0.0f32; dim];
            for &idx in points {
                for d in 0..dim {
                    batch_sum[d] += data[(idx, d)];
                }
            }

            let mut updated_centroid = vec![0.0f32; dim];
            for d in 0..dim {
                let batch_mean = batch_sum[d] / points.len() as f32;
                updated_centroid[d] = (1.0 - eta) * current_centroids[(c, d)] + eta * batch_mean;
            }

            (c, Some((updated_centroid, new_count)))
        })
        .collect();

    // Apply updates
    for (c, update) in centroid_updates {
        if let Some((updated_centroid, new_count)) = update {
            for d in 0..dim {
                centroids[(c, d)] = updated_centroid[d];
            }
            per_cluster_counts[c] = new_count;
        }
    }
}
