mod test_utils;

use serial_test::serial;
use test_utils::*;
use vector_indexer::ivf_index::{load_index, IvfIndex};
use vector_indexer::shards::Shard;
use vector_indexer::vector_store::VectorStore;

// ============================================================================
// Core Functionality Tests
// ============================================================================

#[test]
fn test_ivf_index_creation() {
    // Test that an IVF index can be created without panic
    let index = IvfIndex::new(128);

    drop(index);
}

#[test]
#[serial]
fn test_ivf_index_fit_basic() {
    cleanup_test_files();
    // Test basic fit operation
    let vector_store = create_test_vector_store(100, 8);
    let mut index = IvfIndex::new(8);

    // Fit should complete without panic
    index.fit(&vector_store);

    cleanup_test_files();
}

#[test]
#[serial]
fn test_ivf_index_fit_creates_shards() {
    cleanup_test_files();
    // Verify that fitting creates shard files
    let vector_store = create_test_vector_store(200, 16);
    let mut index = IvfIndex::new(16);

    index.fit(&vector_store);

    // Check that shards directory exists
    let shards_dir = std::path::Path::new("shards");
    assert!(shards_dir.exists(), "Shards directory should be created");

    // Check that at least one shard file exists
    let entries = std::fs::read_dir(shards_dir).expect("Failed to read shards directory");
    let shard_count = entries.count();
    assert!(shard_count > 0, "At least one shard file should be created");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_ivf_index_save_and_load() {
    cleanup_test_files();
    // Test save and load roundtrip
    let vector_store = create_test_vector_store(150, 12);
    let mut index = IvfIndex::new(12);

    index.fit(&vector_store);

    // Save index
    index.save().expect("Failed to save index");

    // Verify index file exists
    let index_path = std::path::Path::new("index/index.bin");
    assert!(index_path.exists(), "Index file should be created");

    // Load index
    let loaded_index = load_index().expect("Failed to load index");

    // Loaded index should be valid (we can't test internals directly)
    drop(loaded_index);

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Search Functionality Tests
// ============================================================================

#[test]
#[serial]
fn test_search_returns_results() {
    cleanup_test_files();
    // Test that search returns results
    let (vector_store, _) = create_gaussian_vector_store(3, 50, 8, 15.0);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    // Create a query vector
    let query = vec![1.0; 8];

    // Search
    let results = index.search(&query, 10, 5).expect("Search failed");

    // Should return some results
    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 10, "Should return at most k results");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_exact_match() {
    cleanup_test_files();
    // Test searching for a vector that exists in the index
    let vector_store = create_test_vector_store(100, 4);
    let mut index = IvfIndex::new(4);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    // Get first vector from store as query
    let first_vector = vector_store.get_vectors().row(0).to_vec();

    // Search
    let results = index.search(&first_vector, 5, 10).expect("Search failed");

    // First result should be the query vector itself (or very close)
    assert!(!results.is_empty());
    let (_id, distance, vector) = &results[0];
    assert!(
        *distance < 0.1,
        "First result should be very close to query"
    );
    assert_eq!(
        *vector, first_vector,
        "First result should be the query vector itself"
    );

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_returns_k_results() {
    cleanup_test_files();
    // Verify that search returns exactly k results (or less if not enough data)
    let vector_store = create_test_vector_store(200, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![5.0; 8];

    // Search for k=15
    let results = index.search(&query, 15, 10).expect("Search failed");

    // Should return 15 results (we have 200 vectors)
    assert_eq!(results.len(), 15, "Should return exactly k results");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_results_sorted_by_distance() {
    cleanup_test_files();
    // Verify that search results are sorted by distance in ascending order
    let vector_store = create_test_vector_store(150, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![5.0; 8];
    let results = index.search(&query, 10, 5).expect("Search failed");

    assert!(!results.is_empty());

    // Verify results are sorted by distance (ascending)
    for i in 1..results.len() {
        assert!(
            results[i].1 >= results[i - 1].1,
            "Results not sorted: distance at {} ({}) < distance at {} ({})",
            i,
            results[i].1,
            i - 1,
            results[i - 1].1
        );
    }

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_n_probe_affects_results() {
    cleanup_test_files();
    // Test that n_probe parameter affects search results
    let (vector_store, _) = create_gaussian_vector_store(5, 40, 16, 20.0);
    let mut index = IvfIndex::new(16);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![10.0; 16];

    // Search with low n_probe
    let results_low = index
        .search(&query, 10, 1)
        .expect("Search with n_probe=1 failed");

    // Search with high n_probe
    let results_high = index
        .search(&query, 10, 10)
        .expect("Search with n_probe=10 failed");

    // Both should return results
    assert!(!results_low.is_empty());
    assert!(!results_high.is_empty());

    // Results might differ (though not guaranteed)
    // At minimum, both searches should complete successfully

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
#[serial]
fn test_search_with_k_larger_than_dataset() {
    cleanup_test_files();
    // Test search when k > number of vectors
    let vector_store = create_test_vector_store(50, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![1.0; 8];

    // Search with k=100 (more than the 50 vectors we have)
    let results = index.search(&query, 100, 20).expect("Search failed");

    // Should return at most 50 results (all available vectors)
    assert!(
        results.len() == 50,
        "Returned more results than available vectors"
    );
    assert!(!results.is_empty(), "Should return some results");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_with_n_probe_larger_than_centroids() {
    cleanup_test_files();
    // Test that n_probe larger than number of centroids works
    let vector_store = create_test_vector_store(50, 8); // Will create small number of clusters
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![1.0; 8];

    // Use very large n_probe
    let results = index
        .search(&query, 5, 1000)
        .expect("Search with large n_probe should work");

    assert!(!results.is_empty());

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_index_with_small_dataset() {
    cleanup_test_files();
    // Test indexing a very small dataset
    let vector_store = create_test_vector_store(10, 4);
    let mut index = IvfIndex::new(4);

    // Should not panic
    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![1.0; 4];
    let results = index
        .search(&query, 3, 2)
        .expect("Search on small index failed");

    assert!(!results.is_empty());

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_index_with_single_vector() {
    cleanup_test_files();
    // Edge case: Index with only one vector
    let vectors_data = vec![(0u64, vec![1.0, 2.0, 3.0, 4.0], 0u64)];
    let vector_store = VectorStore::new(vectors_data);
    let mut index = IvfIndex::new(4);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![1.1, 2.1, 3.1, 4.1];
    let results = index.search(&query, 1, 1).expect("Search failed");

    assert_eq!(results.len(), 1, "Should return the single vector");
    assert_eq!(results[0].0, 0, "Should return vector ID 0");

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_with_k_and_n_probe_zero() {
    cleanup_test_files();
    // Edge case: Search with k=0 and n_probe=0 should return errors
    let vector_store = create_test_vector_store(100, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let query = vec![1.0; 8];

    // Search with k=0 should error
    let result = index.search(&query, 0, 5);
    assert!(result.is_err(), "Search with k=0 should return an error");
    assert_eq!(
        result.unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput,
        "Error should be InvalidInput"
    );

    // Search with n_probe=0 should error
    let result = index.search(&query, 5, 0);
    assert!(
        result.is_err(),
        "Search with n_probe=0 should return an error"
    );
    assert_eq!(
        result.unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput,
        "Error should be InvalidInput"
    );

    // Search with both k=0 and n_probe=0 should error
    let result = index.search(&query, 0, 0);
    assert!(
        result.is_err(),
        "Search with k=0 and n_probe=0 should return an error"
    );
    assert_eq!(
        result.unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput,
        "Error should be InvalidInput"
    );

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
#[serial]
fn test_search_finds_nearest_neighbors() {
    cleanup_test_files();
    // Test that search actually finds nearby vectors
    let (vector_store, _) = create_gaussian_vector_store(3, 60, 8, 30.0);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let vectors = vector_store.get_vectors();
    let query = vectors.row(0).to_vec();

    // Find true nearest neighbors (brute force)
    let true_neighbors = find_true_nearest_neighbors(&query, &vectors, 10);

    // Search using index
    let results = index.search(&query, 10, 15).expect("Search failed");

    let found_neighbors: Vec<usize> = results.iter().map(|(id, _, _)| *id).collect();

    // Calculate recall
    let recall = calculate_recall(&true_neighbors, &found_neighbors);

    // Should have reasonable recall (at least 50%)
    assert!(recall >= 0.5, "Recall too low: {}", recall);

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_index_with_well_separated_clusters() {
    cleanup_test_files();
    // Test indexing well-separated clusters
    let (vector_store, true_labels) = create_gaussian_vector_store(5, 40, 16, 30.0);
    let mut index = IvfIndex::new(16);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let vectors = vector_store.get_vectors();

    // Search for a vector from cluster 0
    let cluster0_idx = true_labels.iter().position(|&l| l == 0).unwrap();
    let query = vectors.row(cluster0_idx).to_vec();

    let results = index.search(&query, 20, 10).expect("Search failed");

    assert!(!results.is_empty());

    // Since clusters are well-separated, most results should be from the same cluster
    let same_cluster_count = results
        .iter()
        .filter(|(id, _, _)| true_labels[*id] == 0)
        .count();

    // At least 50% of results should be from the same cluster
    assert!(
        same_cluster_count as f32 / results.len() as f32 > 0.5,
        "Only {}/{} results from same cluster",
        same_cluster_count,
        results.len()
    );

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
#[serial]
fn test_all_vectors_assigned_to_centroids() {
    cleanup_test_files();
    // Verify that all vectors are assigned to some centroid
    let n = 100;
    let vector_store = create_test_vector_store(n, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);

    // Count vectors in all shards
    let shard_files: Vec<_> = std::fs::read_dir("shards")
        .expect("Failed to read shards dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("bin"))
        .collect();

    let mut total_vectors = 0;

    for entry in shard_files {
        let filename = entry.file_name();
        let filename_str = filename.to_str().unwrap();

        // Extract shard ID from filename (shard_X.bin)
        if let Some(id_str) = filename_str
            .strip_prefix("shard_")
            .and_then(|s| s.strip_suffix(".bin"))
        {
            if let Ok(shard_id) = id_str.parse::<u64>() {
                if let Ok(shard) = Shard::load_from_disk(shard_id) {
                    for ivf_list in &shard.ivf_lists {
                        total_vectors += ivf_list.vectors.len();
                    }
                }
            }
        }
    }

    // All vectors should be assigned
    assert_eq!(
        total_vectors, n,
        "Expected {} vectors, found {} in shards",
        n, total_vectors
    );

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_no_duplicate_vectors_across_shards() {
    cleanup_test_files();
    // Verify that no vector appears in multiple shards/clusters
    let n = 150;
    let vector_store = create_test_vector_store(n, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);

    // Collect all vector IDs from all shards
    let shard_files: Vec<_> = std::fs::read_dir("shards")
        .expect("Failed to read shards dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("bin"))
        .collect();

    let mut all_vector_ids = Vec::new();

    for entry in shard_files {
        let filename = entry.file_name();
        let filename_str = filename.to_str().unwrap();

        if let Some(id_str) = filename_str
            .strip_prefix("shard_")
            .and_then(|s| s.strip_suffix(".bin"))
        {
            if let Ok(shard_id) = id_str.parse::<u64>() {
                if let Ok(shard) = Shard::load_from_disk(shard_id) {
                    for ivf_list in &shard.ivf_lists {
                        for vector in &ivf_list.vectors {
                            all_vector_ids.push(vector.id);
                        }
                    }
                }
            }
        }
    }

    // Check for duplicates
    let mut sorted_ids = all_vector_ids.clone();
    sorted_ids.sort();

    for i in 1..sorted_ids.len() {
        assert_ne!(
            sorted_ids[i],
            sorted_ids[i - 1],
            "Duplicate vector ID {} found",
            sorted_ids[i]
        );
    }

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// High-Dimensional Data Tests
// ============================================================================

#[test]
#[serial]
fn test_index_high_dimensional_vectors() {
    cleanup_test_files();
    // Test with realistic high-dimensional data (e.g., embeddings)
    let dim = 1536;
    let vector_store = create_test_vector_store(200, dim);
    let mut index = IvfIndex::new(dim as u32);

    index.fit(&vector_store);
    index.save().expect("Failed to save high-dim index");

    let query = vec![1.0; dim];
    let results = index
        .search(&query, 10, 5)
        .expect("Search on high-dim index failed");

    assert!(!results.is_empty());

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_search_recall_quality() {
    cleanup_test_files();
    // Test that search achieves reasonable recall
    let (vector_store, _) = create_gaussian_vector_store(5, 50, 16, 25.0);
    let mut index = IvfIndex::new(16);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let vectors = vector_store.get_vectors();

    // Test multiple queries
    let mut total_recall = 0.0;
    let num_queries = 10;

    for i in 0..num_queries {
        let query = vectors.row(i * 10).to_vec();
        let true_neighbors = find_true_nearest_neighbors(&query, &vectors, 10);

        let results = index.search(&query, 10, 20).expect("Search failed");
        let found_neighbors: Vec<usize> = results.iter().map(|(id, _, _)| *id).collect();

        let recall = calculate_recall(&true_neighbors, &found_neighbors);
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f32;

    // Average recall should be decent (at least 60%)
    assert!(avg_recall >= 0.6, "Average recall too low: {}", avg_recall);

    // Cleanup
    cleanup_test_files();
}

// ============================================================================
// Persistence Tests
// ============================================================================

#[test]
#[serial]
fn test_index_persists_across_sessions() {
    cleanup_test_files();
    // Test that index can be saved and loaded in different "sessions"
    let vector_store = create_test_vector_store(100, 8);

    // Session 1: Create and save index
    {
        let mut index = IvfIndex::new(8);
        index.fit(&vector_store);
        index.save().expect("Failed to save index");
    } // index dropped

    // Session 2: Load and use index
    {
        let index = load_index().expect("Failed to load index");
        let query = vec![1.0; 8];
        let results = index.search(&query, 5, 3).expect("Search failed");

        assert!(!results.is_empty());
    }

    // Cleanup
    cleanup_test_files();
}

#[test]
#[serial]
fn test_concurrent_searches() {
    cleanup_test_files();
    // Test that multiple searches can run concurrently
    use std::sync::Arc;
    use std::thread;

    let vector_store = create_test_vector_store(200, 8);
    let mut index = IvfIndex::new(8);

    index.fit(&vector_store);
    index.save().expect("Failed to save index");

    let index = Arc::new(load_index().expect("Failed to load index"));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let index_clone = Arc::clone(&index);
            thread::spawn(move || {
                let query = vec![i as f32; 8];
                let results = index_clone
                    .search(&query, 5, 3)
                    .expect("Search failed in thread");
                assert!(!results.is_empty());
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Cleanup
    cleanup_test_files();
}
