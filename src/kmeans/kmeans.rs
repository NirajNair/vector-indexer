use ndarray::Array2;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;
use wide::f32x8;

/// Runs parallel + SIMD optimized K-Means++ clustering with early stopping.
pub fn run_kmeans_parallel(
    data: &Array2<f32>,                // shape: (n, dim)
    k: usize,                          // number of clusters
    max_iters: usize,                  // maximum iterations
    early_stop_threshold: Option<f32>, // early stop threshold
) -> (Array2<f32>, Vec<usize>) {
    let early_stop_threshold = early_stop_threshold.unwrap_or(1e-4);
    let n = data.nrows();
    let dim = data.ncols();
    let mut rng = thread_rng();

    let mut curr_centroids = kmeans_plus_plus_init(data, k);
    let mut labels = vec![0usize; n];

    for iter in 0..max_iters {
        println!("Iteration {iter}...");

        // Assignment (parallel + SIMD)
        assign_points_simd_parallel(data, &curr_centroids, &mut labels);

        // Update (parallel reduction)
        let (mut new_centroids, counts) = update_centroids_parallel(data, &labels, k, dim);

        // Handle empty clusters
        for c in 0..k {
            if counts[c] == 0 {
                let ri = rng.gen_range(0..n);
                for d in 0..dim {
                    new_centroids[(c, d)] = data[(ri, d)];
                }
            }
        }

        // Compute centroid movement (for early stopping)
        let mut delta = 0.0;
        for c in 0..k {
            for d in 0..dim {
                let diff = new_centroids[(c, d)] - curr_centroids[(c, d)];
                delta += diff * diff;
            }
        }

        delta = (delta / (k * dim) as f32).sqrt();
        println!("→ Centroid delta: {:.6}", delta);

        curr_centroids = new_centroids.clone();
        if delta < early_stop_threshold {
            println!("Converged early at iteration {}", iter + 1);
            break;
        }
    }

    (curr_centroids, labels)
}

use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// K-means++ initialization: smart centroid selection
/// Centroids are chosen with probability proportional to distance^2 from existing centroids
fn kmeans_plus_plus_init(data: &Array2<f32>, k: usize) -> Array2<f32> {
    let n = data.nrows();
    let dim = data.ncols();
    let mut rng = thread_rng();
    let mut centroids = Array2::<f32>::zeros((k, dim));

    // Choose first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&data.row(first_idx));
    println!("K-means++: Selected centroid 0 (point {})", first_idx);

    // Track minimum distance from each point to any centroid
    let mut min_distances = vec![f32::INFINITY; n];

    // Choose remaining k-1 centroids
    for i in 1..k {
        // Update distances: for each point, find distance to nearest centroid
        update_min_distances_parallel(data, &centroids, i, &mut min_distances);

        // Choose next centroid with probability proportional to distance²
        let weights: Vec<f32> = min_distances.iter().map(|&d| d * d).collect();
        let dist = WeightedIndex::new(&weights).expect("All weights are zero or invalid");
        let chosen_idx = dist.sample(&mut rng);

        centroids.row_mut(i).assign(&data.row(chosen_idx));
        println!(
            "K-means++: Selected centroid {} (point {}, dist²={:.2})",
            i, chosen_idx, min_distances[chosen_idx]
        );
    }

    centroids
}

/// Update minimum distances to nearest centroid (parallelized with SIMD)
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

            // Compute distance to latest centroid using SIMD
            let mut acc = f32x8::splat(0.0);
            let mut j = 0;

            while j + 8 <= dim {
                let a = f32x8::new(point.as_slice().unwrap()[j..j + 8].try_into().unwrap());
                let b = f32x8::new(
                    latest_centroid.as_slice().unwrap()[j..j + 8]
                        .try_into()
                        .unwrap(),
                );
                let diff = a - b;
                acc += diff * diff;
                j += 8;
            }

            let mut tail = 0.0;
            while j < dim {
                let diff = point[j] - latest_centroid[j];
                tail += diff * diff;
                j += 1;
            }

            let dist = acc.reduce_add() + tail;

            // Update minimum distance
            if dist < *min_dist {
                *min_dist = dist;
            }
        });
}

fn assign_points_simd_parallel(data: &Array2<f32>, centroids: &Array2<f32>, labels: &mut [usize]) {
    let dim = data.ncols();
    let k = centroids.nrows();

    labels.par_iter_mut().enumerate().for_each(|(i, label)| {
        let p = data.row(i);
        let mut best_c = 0usize;
        let mut best_dist = f32::INFINITY;

        for c in 0..k {
            let cent = centroids.row(c);
            let mut acc = f32x8::splat(0.0);
            let mut j = 0;

            // SIMD 8-element chunks
            while j + 8 <= dim {
                let a = f32x8::new(p.as_slice().unwrap()[j..j + 8].try_into().unwrap());
                let b = f32x8::new(cent.as_slice().unwrap()[j..j + 8].try_into().unwrap());
                let diff = a - b;
                acc += diff * diff;
                j += 8;
            }

            // Handle tail elements
            let mut tail = 0.0;
            for d in j..dim {
                let diff = p[d] - cent[d];
                tail += diff * diff;
            }

            let dist = acc.reduce_add() + tail;
            if dist < best_dist {
                best_dist = dist;
                best_c = c;
            }
        }
        *label = best_c;
    });
}

fn update_centroids_parallel(
    data: &Array2<f32>,
    labels: &[usize],
    k: usize,
    dim: usize,
) -> (Array2<f32>, Vec<usize>) {
    let (sum_acc, count_acc) = (0..data.nrows())
        .into_par_iter()
        .map(|i| {
            // Each thread: local partial sums
            let mut sums = Array2::<f32>::zeros((k, dim));
            let mut counts = vec![0usize; k];
            let c = labels[i];
            for d in 0..dim {
                sums[(c, d)] += data[(i, d)];
            }
            counts[c] += 1;
            (sums, counts)
        })
        .reduce(
            || (Array2::<f32>::zeros((k, dim)), vec![0usize; k]),
            |(mut a_sums, mut a_counts), (b_sums, b_counts)| {
                // Merge partials
                for c in 0..k {
                    for d in 0..dim {
                        a_sums[(c, d)] += b_sums[(c, d)];
                    }
                    a_counts[c] += b_counts[c];
                }
                (a_sums, a_counts)
            },
        );

    // Divide sums by counts to form new centroids
    let mut new_centroids = sum_acc.clone();
    for c in 0..k {
        if count_acc[c] > 0 {
            for d in 0..dim {
                new_centroids[(c, d)] /= count_acc[c] as f32;
            }
        }
    }

    (new_centroids, count_acc)
}
