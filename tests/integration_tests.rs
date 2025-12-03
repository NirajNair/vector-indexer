mod test_utils;

use test_utils::*;
use vector_indexer::ivf_index::{load_index, IvfIndex};
use vector_indexer::shards::Shard;

// Integration tests must run serially to avoid file conflicts
use serial_test::serial;

// ============================================================================
// Full Pipeline Integration Tests
// ============================================================================

#[test]
#[serial]
fn test_full_pipeline_small_dataset() {
    // Test complete pipeline: VectorStore → K-means → IVF Index → Shards → Search
    cleanup_test_files(); // Clean up any leftover files from previous tests

    // 1. Create vector store
    let (vector_store, _labels) = create_gaussian_vector_store(3, 50, 8, 20.0);
    println!("Created vector store with 150 vectors");

    // 2. Build IVF index (this internally runs K-means)
    let mut index = IvfIndex::new(8);
    index.fit(&vector_store);
    println!("Fitted IVF index");

    // 3. Save index and shards
    index.save().expect("Failed to save index");
    println!("Saved index to disk");

    // 4. Verify shards were created
    let shards_dir = std::path::Path::new("shards");
    assert!(shards_dir.exists(), "Shards directory should exist");

    let shard_count = std::fs::read_dir(shards_dir)
        .expect("Failed to read shards dir")
        .count();
    assert!(shard_count > 0, "At least one shard should be created");
    println!("Created {} shards", shard_count);

    // 5. Load index from disk
    let loaded_index = load_index().expect("Failed to load index");
    println!("Loaded index from disk");

    // 6. Perform searches
    let vectors = vector_store.get_vectors();
    let query = vectors.row(0).to_vec();

    let results = loaded_index.search(&query, 10, 5).expect("Search failed");

    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 10, "Should return at most k results");

    // First result should be close to query (since query is from dataset)
    let (_id, distance) = results[0];
    assert!(distance < 1.0, "Nearest neighbor should be close");

    println!(
        "Search completed successfully with {} results",
        results.len()
    );

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_full_pipeline_with_persistence() {
    // Test that the full pipeline works across save/load cycles
    cleanup_test_files(); // Clean up any leftover files from previous tests

    // Phase 1: Build and save
    println!("Phase 1: Building index...");
    let vector_store = create_test_vector_store(200, 16);
    let query_vector = vector_store.get_vectors().row(10).to_vec(); // Save for later

    {
        let mut index = IvfIndex::new(16);
        index.fit(&vector_store);
        index.save().expect("Failed to save index");
    }
    println!("Index built and saved");

    // Phase 2: Load and search
    println!("Phase 2: Loading index and searching...");
    {
        let index = load_index().expect("Failed to load index");

        let results = index.search(&query_vector, 10, 8).expect("Search failed");

        assert!(!results.is_empty(), "Search should return results");
        println!("Search returned {} results", results.len());

        // Query vector should be in results (it's from the dataset)
        let found_self = results.iter().any(|(id, dist)| *id == 10 || *dist < 0.01);
        assert!(found_self, "Should find the query vector itself");
    }

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_multiple_queries_consistency() {
    // Test that searching for the same query multiple times gives consistent results
    cleanup_test_files(); // Clean up any leftover files from previous tests

    let (vector_store, _) = create_gaussian_vector_store(4, 60, 12, 25.0);
    let mut index = IvfIndex::new(12);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let loaded_index = load_index().expect("Failed to load index");

    let query = vec![5.0; 12];

    // Perform same search 5 times
    let mut all_results = Vec::new();
    for i in 0..5 {
        let results = loaded_index
            .search(&query, 15, 10)
            .expect(&format!("Search {} failed", i + 1));
        all_results.push(results);
    }

    // All results should be identical
    for i in 1..5 {
        assert_eq!(
            all_results[0].len(),
            all_results[i].len(),
            "Result count should be consistent"
        );

        for j in 0..all_results[0].len() {
            assert_eq!(
                all_results[0][j].0, all_results[i][j].0,
                "Result IDs should be consistent"
            );
            assert!(
                (all_results[0][j].1 - all_results[i][j].1).abs() < 1e-6,
                "Result distances should be consistent"
            );
        }
    }

    println!("All 5 searches returned consistent results");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_result_validation() {
    // Validate that search results make sense
    cleanup_test_files(); // Clean up any leftover files from previous tests

    let (vector_store, _) = create_gaussian_vector_store(3, 50, 8, 30.0);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let vectors = vector_store.get_vectors();
    let query = vectors.row(0).to_vec();

    let results = index.search(&query, 20, 15).expect("Search failed");

    // Validate results
    assert!(!results.is_empty(), "Should have results");

    // All IDs should be valid (< number of vectors)
    for (id, _dist) in &results {
        assert!(*id < vectors.nrows(), "Vector ID {} out of range", id);
    }

    // Distances should be non-negative
    for (_id, dist) in &results {
        assert!(*dist >= 0.0, "Distance should be non-negative");
    }

    // Distances should be sorted (ascending)
    for i in 1..results.len() {
        assert!(
            results[i].1 >= results[i - 1].1,
            "Results should be sorted by distance"
        );
    }

    println!("All {} results validated successfully", results.len());

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_cross_module_data_flow() {
    // Test that data flows correctly through all modules
    cleanup_test_files(); // Clean up any leftover files from previous tests

    // 1. Create synthetic data with known structure
    let (vector_store, true_labels) = create_gaussian_vector_store(4, 50, 16, 30.0);
    let num_vectors = vector_store.data.len();

    println!("Created {} vectors in {} true clusters", num_vectors, 4);

    // 2. Build index
    let mut index = IvfIndex::new(16);
    index.fit(&vector_store);
    index.save().expect("Failed to save");

    // 3. Verify shards contain all vectors
    let shards_dir = std::path::Path::new("shards");
    let shard_files: Vec<_> = std::fs::read_dir(shards_dir)
        .expect("Failed to read shards dir")
        .filter_map(|e| e.ok())
        .collect();

    println!("Created {} shard files", shard_files.len());

    // 4. Load and search multiple times
    let loaded_index = load_index().expect("Failed to load index");

    let vectors = vector_store.get_vectors();

    // Test searches from different clusters
    for cluster_id in 0..4 {
        // Find a vector from this cluster
        let idx = true_labels
            .iter()
            .position(|&label| label == cluster_id)
            .expect("Should find vector from cluster");

        let query = vectors.row(idx).to_vec();
        let results = loaded_index.search(&query, 10, 10).expect("Search failed");

        assert!(
            !results.is_empty(),
            "Should find neighbors for cluster {}",
            cluster_id
        );
    }

    println!("Successfully searched from all clusters");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_recall_quality_on_known_data() {
    // Test recall quality with known true nearest neighbors
    cleanup_test_files(); // Clean up any leftover files from previous tests

    let (vector_store, _) = create_gaussian_vector_store(4, 60, 16, 25.0);
    let mut index = IvfIndex::new(16);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let loaded_index = load_index().expect("Failed to load index");
    let vectors = vector_store.get_vectors();

    // Test with multiple queries
    let num_queries = 10;
    let k = 10;
    let mut total_recall_low = 0.0;
    let mut total_recall_high = 0.0;

    for i in 0..num_queries {
        let query = vectors.row(i * 20).to_vec();

        // Find true nearest neighbors (brute force)
        let true_neighbors = find_true_nearest_neighbors(&query, &vectors, k);

        // Search with low n_probe
        let results_low = loaded_index.search(&query, k, 5).expect("Search failed");
        let found_low: Vec<usize> = results_low.iter().map(|(id, _)| *id).collect();
        let recall_low = calculate_recall(&true_neighbors, &found_low);
        total_recall_low += recall_low;

        // Search with high n_probe
        let results_high = loaded_index.search(&query, k, 15).expect("Search failed");
        let found_high: Vec<usize> = results_high.iter().map(|(id, _)| *id).collect();
        let recall_high = calculate_recall(&true_neighbors, &found_high);
        total_recall_high += recall_high;
    }

    let avg_recall_low = total_recall_low / num_queries as f32;
    let avg_recall_high = total_recall_high / num_queries as f32;

    println!(
        "Average recall with n_probe=5: {:.2}%",
        avg_recall_low * 100.0
    );
    println!(
        "Average recall with n_probe=15: {:.2}%",
        avg_recall_high * 100.0
    );

    // Average recall with n_probe=15 should be > 70%
    assert!(
        avg_recall_high >= 0.7,
        "Recall too low: {:.2}%",
        avg_recall_high * 100.0
    );

    // Higher n_probe should improve recall
    assert!(
        avg_recall_high >= avg_recall_low,
        "Higher n_probe should improve recall"
    );

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Multi-Shard Integration Tests
// ============================================================================

#[test]
#[serial]
fn test_search_across_multiple_shards() {
    // Test that search correctly queries multiple shards
    cleanup_test_files(); // Clean up any leftover files from previous tests

    // Create enough data to generate multiple shards
    let vector_store = create_test_vector_store(500, 32);
    let mut index = IvfIndex::new(32);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    // Verify multiple shards were created
    let shard_count = std::fs::read_dir("shards")
        .expect("Failed to read shards dir")
        .count();

    println!("Created {} shards", shard_count);
    assert!(shard_count >= 2, "Should create multiple shards");

    // Load and search
    let loaded_index = load_index().expect("Failed to load index");
    let query = vec![10.0; 32];

    // Search with high n_probe to hit multiple shards
    let results = loaded_index.search(&query, 20, 20).expect("Search failed");

    assert!(
        results.len() >= 10,
        "Should find results from multiple shards"
    );

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_kmeans_to_shards_integration() {
    // Test that K-means clustering is properly integrated with IVF index and shards
    cleanup_test_files(); // Clean up any leftover files from previous tests

    use vector_indexer::kmeans::run_kmeans_mini_batch;

    let (vector_store, _) = create_gaussian_vector_store(3, 50, 8, 25.0);
    let vectors = vector_store.get_vectors();

    // Run K-means directly
    let k = 10;
    let (centroids, labels) =
        run_kmeans_mini_batch(&vectors, k, 100, Some(1e-4)).expect("K-means failed");

    // Verify all points are assigned to nearest centroid
    assert!(verify_optimal_assignment(&vectors, &centroids, &labels));

    // Build IVF index (which also runs K-means internally)
    let mut index = IvfIndex::new(8);
    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    // Load shards and verify structure
    let shard_files: Vec<_> = std::fs::read_dir("shards")
        .expect("Failed to read shards dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("bin"))
        .collect();

    let mut total_vectors = 0;
    let mut total_centroids = 0;

    for entry in &shard_files {
        let filename = entry.file_name();
        let filename_str = filename.to_str().unwrap();

        if let Some(id_str) = filename_str
            .strip_prefix("shard_")
            .and_then(|s| s.strip_suffix(".bin"))
        {
            if let Ok(shard_id) = id_str.parse::<u64>() {
                match Shard::load_from_disk(shard_id) {
                    Ok(shard) => {
                        total_centroids += shard.centroids.len();

                        // Verify each IVF list structure
                        for ivf_list in &shard.ivf_lists {
                            // Centroid dimension should match shard dimension
                            assert_eq!(ivf_list.centroid.vector.len(), shard.dimension as usize);

                            // All vectors in the list should have correct dimension
                            for vector in &ivf_list.vectors {
                                assert_eq!(vector.data.len(), shard.dimension as usize);
                                total_vectors += 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to load shard {}: {}", shard_id, e);
                    }
                }
            }
        }
    }

    // All vectors should be in shards
    assert_eq!(
        total_vectors,
        vector_store.data.len(),
        "All vectors should be in shards"
    );

    // Should have created centroids
    assert!(total_centroids > 0, "Should have created centroids");

    println!(
        "Integration test passed: {} vectors across {} centroids",
        total_vectors, total_centroids
    );

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Error Handling Integration Tests
// ============================================================================

#[test]
#[serial]
fn test_missing_shards_handling() {
    // Test behavior when shard files are missing
    cleanup_test_files(); // Clean up any leftover files from previous tests

    let vector_store = create_test_vector_store(100, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    // Delete a shard file
    let shard_files: Vec<_> = std::fs::read_dir("shards")
        .expect("Failed to read shards dir")
        .filter_map(|e| e.ok())
        .collect();

    if !shard_files.is_empty() {
        let first_shard = &shard_files[0].path();
        std::fs::remove_file(first_shard).expect("Failed to delete shard");

        // Try to search (might fail or return partial results)
        let loaded_index = load_index().expect("Failed to load index");
        let query = vec![1.0; 8];

        // Search might fail or return fewer results
        let result = loaded_index.search(&query, 10, 5);

        // Just verify it doesn't panic - behavior may vary
        match result {
            Ok(results) => println!(
                "Search returned {} results despite missing shard",
                results.len()
            ),
            Err(e) => println!("Search failed as expected with missing shard: {}", e),
        }
    }

    // Cleanup
    cleanup_test_files();
}
