use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::fs;
use vector_indexer::ivf_index::Centroid;
use vector_indexer::shards::Shard;
use vector_indexer::vector_store::{Vector, VectorStore};

/// Generate synthetic test vectors with random data
#[allow(dead_code)]
pub fn create_test_vectors(n: usize, dim: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (n, dim),
        (0..n * dim).map(|x| (x as f32 * 0.1) % 50.0).collect(),
    )
    .unwrap()
}

/// Create a VectorStore from test data
#[allow(dead_code)]
pub fn create_test_vector_store(n: usize, dim: usize) -> VectorStore {
    let data = create_test_vectors(n, dim);
    let vectors_data: Vec<(u64, Vec<f32>, u64)> = (0..n)
        .map(|i| {
            let vector = data.row(i).to_vec();
            (i as u64, vector, i as u64)
        })
        .collect();
    VectorStore::new(vectors_data)
}

/// Create synthetic data with well-separated Gaussian clusters
/// Returns (data, true_labels)
#[allow(dead_code)]
pub fn create_gaussian_clusters(
    num_clusters: usize,
    points_per_cluster: usize,
    dim: usize,
    separation: f32,
) -> (Array2<f32>, Vec<usize>) {
    let mut rng = rand::thread_rng();

    let total_points = num_clusters * points_per_cluster;
    let mut data = Array2::<f32>::zeros((total_points, dim));
    let mut true_labels = Vec::with_capacity(total_points);

    for cluster_id in 0..num_clusters {
        // Create a center for this cluster
        let center: Vec<f32> = (0..dim)
            .map(|d| (cluster_id as f32) * separation + (d as f32) * 0.1)
            .collect();

        // Generate points around this center
        for point_id in 0..points_per_cluster {
            let idx = cluster_id * points_per_cluster + point_id;
            true_labels.push(cluster_id);

            for d in 0..dim {
                // Add Gaussian noise around the center
                let noise: f32 = rng.gen_range(-0.5..0.5);
                data[(idx, d)] = center[d] + noise;
            }
        }
    }

    (data, true_labels)
}

/// Create VectorStore from Gaussian clusters
#[allow(dead_code)]
pub fn create_gaussian_vector_store(
    num_clusters: usize,
    points_per_cluster: usize,
    dim: usize,
    separation: f32,
) -> (VectorStore, Vec<usize>) {
    let (data, labels) =
        create_gaussian_clusters(num_clusters, points_per_cluster, dim, separation);
    let n = data.nrows();

    let vectors_data: Vec<(u64, Vec<f32>, u64)> = (0..n)
        .map(|i| {
            let vector = data.row(i).to_vec();
            (i as u64, vector, i as u64)
        })
        .collect();

    (VectorStore::new(vectors_data), labels)
}

/// Calculate the Euclidean distance between two points
#[allow(dead_code)]
pub fn euclidean_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate squared Euclidean distance
#[allow(dead_code)]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Calculate the within-cluster sum of squares (inertia)
/// This is a standard metric to evaluate clustering quality
#[allow(dead_code)]
pub fn calculate_inertia(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
    labels: &Array1<usize>,
) -> f32 {
    let mut inertia = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        let point = data.row(i);
        let centroid = centroids.row(label);
        inertia += euclidean_distance(point, centroid).powi(2);
    }
    inertia
}

/// Verify that each point is assigned to its nearest centroid
#[allow(dead_code)]
pub fn verify_optimal_assignment(
    data: &Array2<f32>,
    centroids: &Array2<f32>,
    labels: &Array1<usize>,
) -> bool {
    for (i, &assigned_label) in labels.iter().enumerate() {
        let point = data.row(i);
        let assigned_dist = euclidean_distance(point, centroids.row(assigned_label));

        // Check if any other centroid is closer
        for c in 0..centroids.nrows() {
            let dist = euclidean_distance(point, centroids.row(c));
            if dist < assigned_dist - 1e-5 {
                // Using small epsilon for floating point comparison
                return false;
            }
        }
    }
    true
}

/// Clean up test files (shards and index)
#[allow(dead_code)]
pub fn cleanup_test_files() {
    // Remove shards directory
    if let Ok(entries) = fs::read_dir("shards") {
        for entry in entries.flatten() {
            let _ = fs::remove_file(entry.path());
        }
    }
    let _ = fs::remove_dir("shards");

    // Remove index directory
    if let Ok(entries) = fs::read_dir("index") {
        for entry in entries.flatten() {
            let _ = fs::remove_file(entry.path());
        }
    }
    let _ = fs::remove_dir("index");
}

/// Clean up specific test shard files by ID
#[allow(dead_code)]
pub fn cleanup_test_shards(shard_ids: &[u64]) {
    for &shard_id in shard_ids {
        let path = format!("shards/shard_{}.bin", shard_id);
        let _ = fs::remove_file(path);
    }
}

/// Verify that a shard has valid structure
#[allow(dead_code)]
pub fn verify_shard_structure(shard: &Shard) -> bool {
    // Check that shard has valid ID
    if shard.id == u64::MAX {
        return false;
    }

    // Check that dimensions are consistent
    if shard.dimension == 0 {
        return false;
    }

    // Check that centroids and IVF lists have same length
    if shard.centroids.len() != shard.ivf_lists.len() {
        return false;
    }

    // Check that each centroid has correct dimension
    for centroid in &shard.centroids {
        if centroid.vector.len() != shard.dimension as usize {
            return false;
        }
    }

    // Check that each IVF list has vectors with correct dimension
    for ivf_list in &shard.ivf_lists {
        for vector in &ivf_list.vectors {
            if vector.data.len() != shard.dimension as usize {
                return false;
            }
        }
    }

    true
}

/// Calculate recall@k for search results
#[allow(dead_code)]
pub fn calculate_recall(true_neighbors: &[usize], found_neighbors: &[usize]) -> f32 {
    let true_set: std::collections::HashSet<_> = true_neighbors.iter().collect();
    let found_count = found_neighbors
        .iter()
        .filter(|id| true_set.contains(id))
        .count();
    found_count as f32 / true_neighbors.len() as f32
}

/// Find true k nearest neighbors (brute force)
#[allow(dead_code)]
pub fn find_true_nearest_neighbors(query: &[f32], data: &Array2<f32>, k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = (0..data.nrows())
        .map(|i| {
            let dist = euclidean_distance_squared(query, data.row(i).as_slice().unwrap());
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(idx, _)| idx).collect()
}

/// Create a centroid with given ID and vector
#[allow(dead_code)]
pub fn create_test_centroid(id: usize, vector: Vec<f32>) -> Centroid {
    Centroid::new(id, vector)
}

/// Generate deterministic test vectors (for reproducibility)
#[allow(dead_code)]
pub fn create_deterministic_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-10.0..10.0)).collect();

    Array2::from_shape_vec((n, dim), data).unwrap()
}

/// Create a test Vector object
#[allow(dead_code)]
pub fn create_test_vector(id: u64, external_id: u64, data: Vec<f32>, timestamp: u64) -> Vector {
    Vector::new(id, external_id, Array1::from_vec(data), timestamp)
}

/// Create multiple test Vector objects from Vec<Vec<f32>>
#[allow(dead_code)]
pub fn create_test_vectors_from_data(vectors_data: Vec<Vec<f32>>) -> Vec<Vector> {
    vectors_data
        .into_iter()
        .enumerate()
        .map(|(i, data)| Vector::new(i as u64, i as u64, Array1::from_vec(data), 0))
        .collect()
}
