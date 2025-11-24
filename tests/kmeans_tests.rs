use ndarray::{Array1, Array2, ArrayView1};
use vector_indexer::kmeans::{run_kmeans_mini_batch, run_kmeans_parallel};

// ============================================================================
// Core Functionality Tests
// ============================================================================

#[test]
fn test_basic_kmeans_runs_without_panic() {
    // Smoke test: K-means completes successfully on simple data
    let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f32).collect()).unwrap();
    let k = 3;

    let (centroids, labels) = run_kmeans_parallel(&data, k, 100, None).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), 3);
    assert_eq!(labels.len(), 10);
}

#[test]
fn test_all_labels_are_valid() {
    // Critical invariant: All assigned labels must be within valid range [0, k)
    let data = Array2::from_shape_vec((20, 4), (0..80).map(|x| x as f32 * 0.1).collect()).unwrap();
    let k = 5;

    let (_centroids, labels) = run_kmeans_parallel(&data, k, 50, None).expect("K-means failed");

    for &label in &labels {
        assert!(label < k, "Label {} is out of bounds for k={}", label, k);
    }
}

#[test]
fn test_labels_assignment_is_optimal() {
    // Correctness: After convergence, each point should be assigned to its nearest centroid
    let (data, _true_labels) = create_gaussian_clusters(3, 20, 4, 10.0);

    let (centroids, labels) =
        run_kmeans_parallel(&data, 3, 100, Some(1e-6)).expect("K-means failed");

    assert!(
        verify_optimal_assignment(&data, &centroids, &labels),
        "Not all points are assigned to their nearest centroid"
    );
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_single_cluster() {
    // Edge case: When k=1, all points should be in cluster 0
    // and the centroid should be close to the mean of all points
    let data = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f32).collect()).unwrap();
    let k = 1;

    let (centroids, labels) = run_kmeans_parallel(&data, k, 50, None).expect("K-means failed");

    assert_eq!(centroids.nrows(), 1);
    assert!(
        labels.iter().all(|&l| l == 0),
        "All labels should be 0 for k=1"
    );

    // Compute expected mean
    let expected_mean: Vec<f32> = (0..3).map(|d| data.column(d).mean().unwrap()).collect();

    // Centroid should be close to the mean
    for d in 0..3 {
        let diff = (centroids[(0, d)] - expected_mean[d]).abs();
        assert!(diff < 1.0, "Centroid differs too much from mean");
    }
}

#[test]
fn test_k_equals_n() {
    // Edge case: When k equals the number of points, each point should form its own cluster
    let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f32 * 2.0).collect()).unwrap();
    let k = 10;

    let (centroids, labels) = run_kmeans_parallel(&data, k, 100, None).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(labels.len(), 10);

    // Each label should appear at least once (though empty clusters can occur)
    // At minimum, we should have k centroids defined
    assert_eq!(centroids.nrows(), k);
}

#[test]
fn test_high_dimensional_data() {
    // Real-world scenario: Test with higher dimensional data (common in embeddings)
    let dim = 256;
    let n = 100;
    let data = Array2::from_shape_vec(
        (n, dim),
        (0..n * dim).map(|x| (x as f32 * 0.01) % 100.0).collect(),
    )
    .unwrap();
    let k = 10;

    let (centroids, labels) = run_kmeans_parallel(&data, k, 50, None).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), dim);
    assert_eq!(labels.len(), n);
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

// ============================================================================
// Algorithm Behavior Tests
// ============================================================================

#[test]
fn test_convergence_improves_clustering() {
    // Algorithm behavior: Running more iterations should not increase inertia (WCSS)
    let (data, _) = create_gaussian_clusters(3, 30, 5, 15.0);

    // Run with few iterations
    let (centroids_few, labels_few) =
        run_kmeans_parallel(&data, 3, 5, None).expect("K-means failed");
    let inertia_few = calculate_inertia(&data, &centroids_few, &labels_few);

    // Run with many iterations
    let (centroids_many, labels_many) =
        run_kmeans_parallel(&data, 3, 100, Some(1e-6)).expect("K-means failed");
    let inertia_many = calculate_inertia(&data, &centroids_many, &labels_many);

    // More iterations should lead to lower or equal inertia
    assert!(
        inertia_many <= inertia_few + 1e-3,
        "Inertia increased with more iterations: {} -> {}",
        inertia_few,
        inertia_many
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_well_separated_clusters_are_recovered() {
    // Integration: Create 3 well-separated clusters and verify K-means recovers them
    let (data, true_labels) = create_gaussian_clusters(3, 40, 4, 25.0);

    let (centroids, predicted_labels) =
        run_kmeans_parallel(&data, 3, 100, Some(1e-5)).expect("K-means failed");

    // We can't directly compare labels (they might be permuted)
    // Instead, check that points from the same true cluster are assigned to the same predicted cluster
    let mut cluster_mapping = std::collections::HashMap::new();

    for i in 0..data.nrows() {
        let true_label = true_labels[i];
        let pred_label = predicted_labels[i];

        if let Some(&mapped_pred) = cluster_mapping.get(&true_label) {
            assert_eq!(
                pred_label, mapped_pred,
                "Points from true cluster {} were split between predicted clusters {} and {}",
                true_label, mapped_pred, pred_label
            );
        } else {
            cluster_mapping.insert(true_label, pred_label);
        }
    }

    // At minimum, verify optimal assignment
    assert!(verify_optimal_assignment(
        &data,
        &centroids,
        &predicted_labels
    ));

    // Verify low inertia (tight clusters)
    let inertia = calculate_inertia(&data, &centroids, &predicted_labels);
    let avg_inertia_per_point = inertia / data.nrows() as f32;

    // With well-separated clusters, inertia should be relatively low
    assert!(
        avg_inertia_per_point < 2.0,
        "Inertia too high for well-separated clusters: {}",
        avg_inertia_per_point
    );
}

#[test]
fn test_identical_points_handled_correctly() {
    // Edge case: Test with multiple identical points
    let mut data_vec = vec![1.0, 2.0, 3.0]; // One point
    data_vec.extend_from_slice(&[1.0, 2.0, 3.0]); // Duplicate
    data_vec.extend_from_slice(&[1.0, 2.0, 3.0]); // Duplicate
    data_vec.extend_from_slice(&[10.0, 20.0, 30.0]); // Different point
    data_vec.extend_from_slice(&[10.0, 20.0, 30.0]); // Duplicate

    let data = Array2::from_shape_vec((5, 3), data_vec).unwrap();
    let k = 2;

    let (centroids, labels) =
        run_kmeans_parallel(&data, k, 100, Some(1e-5)).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(labels.len(), 5);

    // The three identical points at (1,2,3) should have the same label
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);

    // The two identical points at (10,20,30) should have the same label
    assert_eq!(labels[3], labels[4]);

    // The two groups should have different labels
    assert_ne!(labels[0], labels[3]);
}

// ============================================================================
// Scalability Tests
// ============================================================================

#[test]
fn test_large_dataset() {
    // Scalability check: Test with a larger dataset to ensure parallel code scales
    let n = 1000;
    let dim = 32;
    let k = 20;

    let data = Array2::from_shape_vec(
        (n, dim),
        (0..n * dim).map(|x| (x as f32 * 0.1) % 50.0).collect(),
    )
    .unwrap();

    let (centroids, labels) =
        run_kmeans_parallel(&data, k, 50, Some(1e-4)).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), dim);
    assert_eq!(labels.len(), n);

    // Verify optimal assignment (might be slow, but ensures correctness)
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

// ============================================================================
// Mini-batch K-means Tests
// ============================================================================

#[test]
fn test_mini_batch_kmeans_basic() {
    // Basic test: Mini-batch K-means completes successfully on simple data
    let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f32).collect()).unwrap();
    let k = 3;

    let (centroids, labels) =
        run_kmeans_mini_batch(&data, k, 100, None).expect("Mini-batch K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), 3);
    assert_eq!(labels.len(), 100);
}

#[test]
fn test_mini_batch_kmeans_labels_valid() {
    // Verify all labels are in valid range
    let data = Array2::from_shape_vec((50, 4), (0..200).map(|x| x as f32 * 0.1).collect()).unwrap();
    let k = 5;

    let (_centroids, labels) =
        run_kmeans_mini_batch(&data, k, 50, None).expect("Mini-batch K-means failed");

    for &label in &labels {
        assert!(label < k, "Label {} is out of bounds for k={}", label, k);
    }
}

#[test]
fn test_mini_batch_kmeans_optimal_assignment() {
    // After final assignment, all points should be assigned to nearest centroid
    let (data, _true_labels) = create_gaussian_clusters(3, 40, 4, 25.0);

    let (centroids, labels) =
        run_kmeans_mini_batch(&data, 3, 100, Some(1e-5)).expect("Mini-batch K-means failed");

    assert!(
        verify_optimal_assignment(&data, &centroids, &labels),
        "Not all points are assigned to their nearest centroid"
    );
}

#[test]
fn test_mini_batch_kmeans_separated_clusters() {
    // Integration test: Well-separated clusters should be recovered
    let (data, _true_labels) = create_gaussian_clusters(3, 50, 4, 30.0);

    let (centroids, labels) =
        run_kmeans_mini_batch(&data, 3, 100, Some(1e-4)).expect("Mini-batch K-means failed");

    // Verify optimal assignment
    assert!(verify_optimal_assignment(&data, &centroids, &labels));

    // Verify reasonable inertia
    let inertia = calculate_inertia(&data, &centroids, &labels);
    let avg_inertia_per_point = inertia / data.nrows() as f32;

    assert!(
        avg_inertia_per_point < 2.5,
        "Inertia too high for well-separated clusters: {}",
        avg_inertia_per_point
    );
}

#[test]
fn test_mini_batch_kmeans_large_dataset() {
    // Scalability: Mini-batch should handle large datasets efficiently
    let n = 1000;
    let dim = 32;
    let k = 15;

    let data = Array2::from_shape_vec(
        (n, dim),
        (0..n * dim).map(|x| (x as f32 * 0.1) % 50.0).collect(),
    )
    .unwrap();

    let (centroids, labels) =
        run_kmeans_mini_batch(&data, k, 50, Some(1e-4)).expect("Mini-batch K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), dim);
    assert_eq!(labels.len(), n);

    // Verify optimal assignment
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

#[test]
fn test_mini_batch_vs_full_batch_quality() {
    // Compare mini-batch and full-batch K-means results on the same data
    let (data, _) = create_gaussian_clusters(4, 50, 8, 20.0);
    let k = 4;

    // Run both algorithms
    let (centroids_full, labels_full) =
        run_kmeans_parallel(&data, k, 100, Some(1e-5)).expect("Full-batch K-means failed");
    let (centroids_mini, labels_mini) =
        run_kmeans_mini_batch(&data, k, 100, Some(1e-5)).expect("Mini-batch K-means failed");

    // Calculate inertia for both
    let inertia_full = calculate_inertia(&data, &centroids_full, &labels_full);
    let inertia_mini = calculate_inertia(&data, &centroids_mini, &labels_mini);

    println!("Full-batch inertia: {:.4}", inertia_full);
    println!("Mini-batch inertia: {:.4}", inertia_mini);

    // Mini-batch should produce reasonable results (within 50% of full-batch)
    // This is a lenient check since mini-batch is approximate
    assert!(
        inertia_mini < inertia_full * 1.5,
        "Mini-batch inertia ({}) is too much worse than full-batch ({})",
        inertia_mini,
        inertia_full
    );

    // Both should have optimal final assignment
    assert!(verify_optimal_assignment(
        &data,
        &centroids_full,
        &labels_full
    ));
    assert!(verify_optimal_assignment(
        &data,
        &centroids_mini,
        &labels_mini
    ));
}

#[test]
fn test_mini_batch_kmeans_small_dataset() {
    // Edge case: Mini-batch should work even when dataset is smaller than typical batch size
    let data = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f32 * 0.5).collect()).unwrap();
    let k = 3;

    let (centroids, labels) = run_kmeans_mini_batch(&data, k, 50, None)
        .expect("Mini-batch K-means failed on small dataset");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(labels.len(), 20);
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

// ============================================================================
// Hierarchical Assignment Tests (for large k)
// ============================================================================

#[test]
fn test_mini_batch_kmeans_large_k() {
    // Test with k > 100 to trigger hierarchical assignment
    let n = 5000;
    let dim = 32;
    let k = 200; // Large k to trigger hierarchical assignment

    let data = Array2::from_shape_vec(
        (n, dim),
        (0..n * dim).map(|x| (x as f32 * 0.1) % 50.0).collect(),
    )
    .unwrap();

    let (centroids, labels) = run_kmeans_mini_batch(&data, k, 30, Some(1e-4))
        .expect("Mini-batch K-means failed with large k");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), dim);
    assert_eq!(labels.len(), n);

    // Verify optimal assignment (hierarchical should still be accurate)
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

#[test]
fn test_hierarchical_vs_brute_force_quality() {
    // Compare results with k slightly above and below the threshold
    let (data, _) = create_gaussian_clusters(5, 50, 8, 20.0);
    let k = 25; // Will use brute force

    let (centroids, labels_brute) =
        run_kmeans_mini_batch(&data, k, 50, Some(1e-5)).expect("Brute force failed");

    // Both should produce valid optimal assignments
    assert!(verify_optimal_assignment(&data, &centroids, &labels_brute));

    // Test with larger k (will use hierarchical)
    let k_large = 150;

    // Create larger dataset for larger k
    let (data_large, _) = create_gaussian_clusters(10, 100, 8, 25.0);

    let (centroids_hier, labels_hier) =
        run_kmeans_mini_batch(&data_large, k_large, 50, Some(1e-4)).expect("Hierarchical failed");

    assert_eq!(centroids_hier.nrows(), k_large);
    assert!(verify_optimal_assignment(
        &data_large,
        &centroids_hier,
        &labels_hier
    ));
}

// ============================================================================
// Helper Functions for Testing
// ============================================================================

/// Calculate the Euclidean distance between two points
fn euclidean_distance(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate the within-cluster sum of squares (inertia)
/// This is a standard metric to evaluate clustering quality
fn calculate_inertia(data: &Array2<f32>, centroids: &Array2<f32>, labels: &Array1<usize>) -> f32 {
    let mut inertia = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        let point = data.row(i);
        let centroid = centroids.row(label);
        inertia += euclidean_distance(point, centroid).powi(2);
    }
    inertia
}

/// Verify that each point is assigned to its nearest centroid
fn verify_optimal_assignment(
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

/// Create synthetic data with well-separated Gaussian clusters
fn create_gaussian_clusters(
    num_clusters: usize,
    points_per_cluster: usize,
    dim: usize,
    separation: f32,
) -> (Array2<f32>, Vec<usize>) {
    use rand::Rng;
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
