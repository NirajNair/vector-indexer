mod test_utils;

use ndarray::Array2;
use test_utils::*;
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
    // Edge case: When k equals the number of points, each point could form its own cluster
    let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f32 * 2.0).collect()).unwrap();
    let k = 10;

    let (centroids, labels) = run_kmeans_parallel(&data, k, 100, None).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(labels.len(), 10);

    // All labels should be valid
    for &label in &labels {
        assert!(label < k, "Label {} is out of bounds for k={}", label, k);
    }
}

#[test]
fn test_high_dimensional_data() {
    // Real-world scenario: Test with higher dimensional data (common in embeddings)
    let dim = 1536;
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

#[test]
fn test_very_small_dataset() {
    // Edge case: K-means should work with very small datasets
    let data = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 10.0, 11.0, 12.0, 10.1, 11.1, 12.1, 5.0, 6.0, 7.0,
        ],
    )
    .unwrap();
    let k = 2;

    let (centroids, labels) = run_kmeans_parallel(&data, k, 50, None).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(labels.len(), 5);

    // All labels should be valid
    for &label in &labels {
        assert!(label < k);
    }

    // Verify optimal assignment
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

#[test]
fn test_deterministic_with_same_initialization() {
    // Test that running K-means multiple times produces valid results
    // Note: K-means++ initialization is random, so exact reproducibility isn't guaranteed
    // Test with three different dataset sizes to verify 20% threshold is appropriate

    println!("\n=== Testing with different dataset dimensions ===");

    // Test 1: Small dataset (100, 8)
    println!("\nTest 1: Dataset (100, 8)");
    let data1 = create_deterministic_vectors(100, 8, 42);
    let k1 = 5;

    let (centroids1a, labels1a) =
        run_kmeans_parallel(&data1, k1, 50, Some(1e-4)).expect("First run failed");
    let (centroids1b, labels1b) =
        run_kmeans_parallel(&data1, k1, 50, Some(1e-4)).expect("Second run failed");

    assert_eq!(centroids1a.nrows(), k1);
    assert_eq!(centroids1b.nrows(), k1);
    assert!(verify_optimal_assignment(&data1, &centroids1a, &labels1a));
    assert!(verify_optimal_assignment(&data1, &centroids1b, &labels1b));

    let inertia1a = calculate_inertia(&data1, &centroids1a, &labels1a);
    let inertia1b = calculate_inertia(&data1, &centroids1b, &labels1b);

    let ratio1 = if inertia1a > inertia1b {
        inertia1a / inertia1b
    } else {
        inertia1b / inertia1a
    };
    println!(
        "  Inertia 1a: {:.4}, Inertia 1b: {:.4}, Ratio: {:.4}",
        inertia1a, inertia1b, ratio1
    );
    assert!(
        ratio1 < 1.2,
        "Dataset (100, 8): Inertia ratio {} too high",
        ratio1
    );

    // Test 2: Medium dataset (1000, 128)
    println!("\nTest 2: Dataset (1000, 128)");
    let data2 = create_deterministic_vectors(1000, 128, 42);
    let k2 = 10;

    let (centroids2a, labels2a) =
        run_kmeans_parallel(&data2, k2, 50, Some(1e-4)).expect("First run failed");
    let (centroids2b, labels2b) =
        run_kmeans_parallel(&data2, k2, 50, Some(1e-4)).expect("Second run failed");

    assert_eq!(centroids2a.nrows(), k2);
    assert_eq!(centroids2b.nrows(), k2);
    assert!(verify_optimal_assignment(&data2, &centroids2a, &labels2a));
    assert!(verify_optimal_assignment(&data2, &centroids2b, &labels2b));

    let inertia2a = calculate_inertia(&data2, &centroids2a, &labels2a);
    let inertia2b = calculate_inertia(&data2, &centroids2b, &labels2b);

    let ratio2 = if inertia2a > inertia2b {
        inertia2a / inertia2b
    } else {
        inertia2b / inertia2a
    };
    println!(
        "  Inertia 2a: {:.4}, Inertia 2b: {:.4}, Ratio: {:.4}",
        inertia2a, inertia2b, ratio2
    );
    assert!(
        ratio2 < 1.2,
        "Dataset (1000, 128): Inertia ratio {} too high",
        ratio2
    );

    // Test 3: Large dataset (5000, 256)
    println!("\nTest 3: Dataset (5000, 256)");
    let data3 = create_deterministic_vectors(5000, 256, 42);
    let k3 = 15;

    let (centroids3a, labels3a) =
        run_kmeans_parallel(&data3, k3, 50, Some(1e-4)).expect("First run failed");
    let (centroids3b, labels3b) =
        run_kmeans_parallel(&data3, k3, 50, Some(1e-4)).expect("Second run failed");

    assert_eq!(centroids3a.nrows(), k3);
    assert_eq!(centroids3b.nrows(), k3);
    assert!(verify_optimal_assignment(&data3, &centroids3a, &labels3a));
    assert!(verify_optimal_assignment(&data3, &centroids3b, &labels3b));

    let inertia3a = calculate_inertia(&data3, &centroids3a, &labels3a);
    let inertia3b = calculate_inertia(&data3, &centroids3b, &labels3b);

    let ratio3 = if inertia3a > inertia3b {
        inertia3a / inertia3b
    } else {
        inertia3b / inertia3a
    };
    println!(
        "  Inertia 3a: {:.4}, Inertia 3b: {:.4}, Ratio: {:.4}",
        inertia3a, inertia3b, ratio3
    );
    assert!(
        ratio3 < 1.2,
        "Dataset (5000, 256): Inertia ratio {} too high",
        ratio3
    );

    println!("\n=== Summary ===");
    println!(
        "Dataset (100, 8):    Ratio = {:.4} ({})",
        ratio1,
        if ratio1 < 1.2 { "PASS" } else { "FAIL" }
    );
    println!(
        "Dataset (1000, 128): Ratio = {:.4} ({})",
        ratio2,
        if ratio2 < 1.2 { "PASS" } else { "FAIL" }
    );
    println!(
        "Dataset (5000, 256): Ratio = {:.4} ({})",
        ratio3,
        if ratio3 < 1.2 { "PASS" } else { "FAIL" }
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
fn test_clustering_quality_metric() {
    // Verify that inertia is calculated correctly
    let (data, _) = create_gaussian_clusters(2, 30, 3, 15.0);
    let (centroids, labels) =
        run_kmeans_parallel(&data, 2, 100, Some(1e-5)).expect("K-means failed");

    let inertia = calculate_inertia(&data, &centroids, &labels);

    // Inertia should be positive
    assert!(inertia > 0.0, "Inertia should be positive");

    // For well-separated clusters, inertia should be reasonable
    assert!(inertia < 1000.0, "Inertia too high: {}", inertia);
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

    // Verify optimal assignment
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

#[test]
fn test_many_clusters() {
    // Test K-means with many clusters (k = 10% of data)
    let n = 500;
    let dim = 16;
    let k = 50;

    let data = Array2::from_shape_vec(
        (n, dim),
        (0..n * dim).map(|x| (x as f32 * 0.05) % 30.0).collect(),
    )
    .unwrap();

    let (centroids, labels) =
        run_kmeans_parallel(&data, k, 50, Some(1e-4)).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);
    assert_eq!(centroids.ncols(), dim);
    assert_eq!(labels.len(), n);

    // All labels should be valid
    for &label in &labels {
        assert!(label < k, "Label {} out of bounds", label);
    }

    // Verify optimal assignment
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

#[test]
fn test_mini_batch_with_single_cluster() {
    // Edge case: Mini-batch K-means with k=1
    let data = Array2::from_shape_vec((50, 4), (0..200).map(|x| x as f32 * 0.3).collect()).unwrap();
    let k = 1;

    let (centroids, labels) =
        run_kmeans_mini_batch(&data, k, 50, None).expect("Mini-batch K-means failed");

    assert_eq!(centroids.nrows(), 1);
    assert_eq!(labels.len(), 50);

    // All labels should be 0
    assert!(
        labels.iter().all(|&l| l == 0),
        "All labels should be 0 for k=1"
    );

    // Compute expected mean
    let expected_mean: Vec<f32> = (0..4).map(|d| data.column(d).mean().unwrap()).collect();

    // Centroid should be close to the mean
    for d in 0..4 {
        let diff = (centroids[(0, d)] - expected_mean[d]).abs();
        assert!(diff < 2.0, "Centroid differs too much from mean");
    }
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
    // Compare brute force (k < 100) vs hierarchical (k > 100) assignment quality

    // Test with k=25 (brute force)
    let (data_small, _) = create_gaussian_clusters(5, 50, 8, 20.0);
    let k_small = 25;

    let (centroids_small, labels_small) =
        run_kmeans_mini_batch(&data_small, k_small, 50, Some(1e-5))
            .expect("Brute force K-means failed");

    assert_eq!(centroids_small.nrows(), k_small);
    assert!(verify_optimal_assignment(
        &data_small,
        &centroids_small,
        &labels_small
    ));

    // Test with k=150 (hierarchical)
    let (data_large, _) = create_gaussian_clusters(10, 100, 8, 25.0);
    let k_large = 150;

    let (centroids_large, labels_large) =
        run_kmeans_mini_batch(&data_large, k_large, 50, Some(1e-4))
            .expect("Hierarchical K-means failed");

    assert_eq!(centroids_large.nrows(), k_large);
    // Hierarchical assignment should still maintain optimal assignment
    assert!(verify_optimal_assignment(
        &data_large,
        &centroids_large,
        &labels_large
    ));
}

#[test]
fn test_hierarchical_assignment_accuracy() {
    // Verify that hierarchical assignment still maintains accuracy
    let (data, _) = create_gaussian_clusters(10, 100, 8, 25.0);
    let k = 150;

    let (centroids, labels) =
        run_kmeans_mini_batch(&data, k, 50, Some(1e-4)).expect("Hierarchical assignment failed");

    // Should still maintain optimal assignment
    assert!(verify_optimal_assignment(&data, &centroids, &labels));
}

#[test]
fn test_kmeans_plus_plus_initialization() {
    // Test that K-means++ initialization produces good results
    let (data, _) = create_gaussian_clusters(4, 50, 8, 30.0);
    let k = 4;

    // Run K-means with early stopping
    let (centroids, labels) =
        run_kmeans_parallel(&data, k, 20, Some(1e-3)).expect("K-means failed");

    assert_eq!(centroids.nrows(), k);

    // With good initialization (K-means++), should converge quickly
    // Verify optimal assignment even with few iterations
    assert!(verify_optimal_assignment(&data, &centroids, &labels));

    // Centroids should be well-separated (K-means++ property)
    for i in 0..k {
        for j in (i + 1)..k {
            let mut dist_sq = 0.0;
            for d in 0..8 {
                let diff = centroids[(i, d)] - centroids[(j, d)];
                dist_sq += diff * diff;
            }
            // Centroids should not be too close to each other
            assert!(dist_sq > 10.0, "Centroids {} and {} too close", i, j);
        }
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_empty_data_returns_error() {
    // Error case: Empty data should return an error
    let data = Array2::<f32>::zeros((0, 5));
    let result = run_kmeans_parallel(&data, 3, 100, None);

    assert!(result.is_err(), "Empty data should return an error");
}

#[test]
fn test_k_larger_than_n_handled() {
    // Edge case: k > n should be handled gracefully
    let data = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 11.0, 12.0, 13.0,
        ],
    )
    .unwrap();
    let k = 10; // k > n

    // This might succeed with empty clusters or fail gracefully
    // Either behavior is acceptable, but should not panic
    let result = run_kmeans_parallel(&data, k, 50, None);

    match result {
        Ok((centroids, labels)) => {
            // If it succeeds, verify basic properties
            assert!(centroids.nrows() <= k);
            assert_eq!(labels.len(), 5);
            for &label in &labels {
                assert!(label < k);
            }
        }
        Err(_) => {
            // Failing gracefully is also acceptable
            // The important thing is it didn't panic
        }
    }
}

// ============================================================================
// Parallel Execution Tests
// ============================================================================

#[test]
fn test_parallel_execution_produces_valid_results() {
    // Verify parallel execution doesn't introduce race conditions
    let (data, _) = create_gaussian_clusters(5, 100, 16, 20.0);
    let k = 5;

    // Run multiple times
    for _ in 0..3 {
        let (centroids, labels) =
            run_kmeans_parallel(&data, k, 50, Some(1e-4)).expect("K-means failed");

        // Each run should produce valid results
        assert_eq!(centroids.nrows(), k);
        assert!(verify_optimal_assignment(&data, &centroids, &labels));
    }
}
