mod test_utils;

use serial_test::serial;
use std::sync::mpsc;
use test_utils::*;
use vector_indexer::ivf_index::{Centroid, IVFList};
use vector_indexer::shards::Shard;

// ============================================================================
// Core Functionality Tests
// ============================================================================

#[test]
#[serial]
fn test_shard_creation_and_saving() {
    // Test that a shard can be created with valid properties
    let centroid = Centroid::new(0, vec![1.0, 2.0, 3.0]);
    let vectors = create_test_vectors_from_data(vec![vec![1.0, 2.0, 3.0]]);
    let ivf_list = IVFList::new(centroid.clone(), vectors);

    let shard = Shard::new(1, vec![centroid], vec![ivf_list], 3);

    assert_eq!(shard.id, 1);
    assert_eq!(shard.dimension, 3);
    assert_eq!(shard.centroids.len(), 1);
    assert_eq!(shard.ivf_lists.len(), 1);

    // Save shard
    shard.save().expect("Failed to save shard");

    // Verify file exists
    let path = std::path::Path::new("shards/shard_1.bin");
    assert!(path.exists(), "Shard file was not created");

    // Cleanup
    cleanup_test_shards(&[1]);
}

#[test]
#[serial]
fn test_shard_roundtrip_single_centroid() {
    // Test saving and loading a shard with one centroid
    let vector_id: u32 = 5;
    let shard_id = 2000;
    let centroid = Centroid::new(vector_id.clone() as usize, vec![10.0, 20.0, 30.0]);
    let vectors_data = vec![
        vec![10.1, 20.1, 30.1],
        vec![10.2, 20.2, 30.2],
        vec![10.3, 20.3, 30.3],
    ];
    let vectors = create_test_vectors_from_data(vectors_data.clone());
    let ivf_list = IVFList::new(centroid.clone(), vectors);

    let shard = Shard::new(shard_id.clone(), vec![centroid.clone()], vec![ivf_list], 3);

    // Save
    shard.save().expect("Failed to save shard");

    // Load using get_centroid_vectors
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(shard_id.clone(), &[vector_id.clone() as u64])
            .await
            .expect("Failed to load centroid vectors");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 1);
    let (loaded_id, loaded_centroid, loaded_vectors) = &loaded[0];

    // Verify centroid
    assert_eq!(*loaded_id, vector_id as u64);
    assert_eq!(loaded_centroid.len(), 3);
    assert!((loaded_centroid[0] - 10.0).abs() < 1e-5);
    assert!((loaded_centroid[1] - 20.0).abs() < 1e-5);
    assert!((loaded_centroid[2] - 30.0).abs() < 1e-5);

    // Verify vectors
    assert_eq!(loaded_vectors.len(), 3);
    for (i, (_meta, vec)) in loaded_vectors.iter().enumerate() {
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - vectors_data[i][0]).abs() < 1e-5);
        assert!((vec[1] - vectors_data[i][1]).abs() < 1e-5);
        assert!((vec[2] - vectors_data[i][2]).abs() < 1e-5);
    }

    // Cleanup
    cleanup_test_shards(&[2000]);
}

#[test]
#[serial]
fn test_shard_roundtrip_multiple_centroids() {
    // Test saving and loading a shard with multiple centroids
    let centroid1 = Centroid::new(10, vec![1.0, 2.0]);
    let centroid2 = Centroid::new(11, vec![5.0, 6.0]);
    let centroid3 = Centroid::new(12, vec![9.0, 10.0]);

    let ivf1 = IVFList::new(
        centroid1.clone(),
        create_test_vectors_from_data(vec![vec![1.1, 2.1]]),
    );
    let ivf2 = IVFList::new(
        centroid2.clone(),
        create_test_vectors_from_data(vec![vec![5.1, 6.1], vec![5.2, 6.2]]),
    );
    let ivf3 = IVFList::new(
        centroid3.clone(),
        create_test_vectors_from_data(vec![vec![9.1, 10.1]]),
    );

    let shard = Shard::new(
        2001,
        vec![centroid1, centroid2, centroid3],
        vec![ivf1, ivf2, ivf3],
        2,
    );

    // Save
    shard.save().expect("Failed to save shard");

    // Load all centroids
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2001, &[10, 11, 12])
            .await
            .expect("Failed to load centroids");
        tx.send(result).unwrap();
    });
    let loaded_centroids = rx.recv().unwrap();

    assert_eq!(loaded_centroids.len(), 3);

    // Verify each centroid was loaded
    let ids: Vec<u64> = loaded_centroids.iter().map(|(id, _, _)| *id).collect();
    assert!(ids.contains(&10));
    assert!(ids.contains(&11));
    assert!(ids.contains(&12));

    // Cleanup
    cleanup_test_shards(&[2001]);
}

#[test]
#[serial]
fn test_shard_selective_centroid_loading() {
    // Test loading only specific centroids from a shard
    let centroids: Vec<Centroid> = (0..10)
        .map(|i| Centroid::new(i, vec![i as f32, (i + 1) as f32]))
        .collect();

    let ivf_lists: Vec<IVFList> = centroids
        .iter()
        .map(|c| {
            IVFList::new(
                c.clone(),
                create_test_vectors_from_data(vec![vec![0.0, 0.0]]),
            )
        })
        .collect();

    let shard = Shard::new(2002, centroids, ivf_lists, 2);
    shard.save().expect("Failed to save shard");

    // Load only centroids 2, 5, 7
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2002, &[2, 5, 7])
            .await
            .expect("Failed to load selective centroids");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 3);

    let ids: Vec<u64> = loaded.iter().map(|(id, _, _)| *id).collect();
    assert!(ids.contains(&2));
    assert!(ids.contains(&5));
    assert!(ids.contains(&7));

    // Cleanup
    cleanup_test_shards(&[2002]);
}

#[test]
#[serial]
fn test_shard_structure_validation() {
    // Test the verify_shard_structure helper
    let centroid = Centroid::new(0, vec![1.0, 2.0, 3.0]);
    let ivf_list = IVFList::new(
        centroid.clone(),
        create_test_vectors_from_data(vec![vec![1.0, 2.0, 3.0]]),
    );

    let valid_shard = Shard::new(3000, vec![centroid], vec![ivf_list], 3);

    assert!(
        verify_shard_structure(&valid_shard),
        "Valid shard should pass validation"
    );
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
#[serial]
fn test_shard_with_empty_ivf_list() {
    let centroid = Centroid::new(0, vec![1.0, 2.0, 3.0]);
    let ivf_list = IVFList::new(centroid.clone(), vec![]);

    let shard = Shard::new(3000, vec![centroid], vec![ivf_list], 3);
    shard.save().expect("Failed to save shard");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(3000, &[0])
            .await
            .expect("Failed to load centroids");
        tx.send(result).unwrap();
    });
    let loaded_centroids = rx.recv().unwrap();

    assert_eq!(loaded_centroids.len(), 1);
    // loaded_centroids[0] is (centroid_id, centroid_vector, vectors)
    // Check that centroid vector has correct length
    assert_eq!(loaded_centroids[0].1.len(), 3);
    // Check that vectors list is empty
    assert_eq!(loaded_centroids[0].2.len(), 0);

    cleanup_test_shards(&[3000]);
}

#[test]
#[serial]
fn test_shard_with_high_dimensional_vectors() {
    // Test with high-dimensional vectors (e.g., 512 dimensions)
    let dim = 512;
    let centroid = Centroid::new(20, vec![1.0; dim]);
    let vectors = create_test_vectors_from_data(vec![vec![2.0; dim], vec![3.0; dim]]);
    let ivf_list = IVFList::new(centroid.clone(), vectors);

    let shard = Shard::new(2003, vec![centroid], vec![ivf_list], dim as u32);

    // Save
    shard.save().expect("Failed to save high-dimensional shard");

    // Load
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2003, &[20])
            .await
            .expect("Failed to load high-dimensional shard");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 1);
    let (_id, loaded_centroid, loaded_vectors) = &loaded[0];

    assert_eq!(loaded_centroid.len(), dim);
    assert_eq!(loaded_vectors.len(), 2);
    assert_eq!(loaded_vectors[0].1.len(), dim);

    // Cleanup
    cleanup_test_shards(&[2003]);
}

#[test]
#[serial]
fn test_shard_with_many_vectors_per_centroid() {
    // Test that shards can handle many vectors per centroid efficiently
    let num_vectors = 1000;
    let centroid = Centroid::new(100, vec![1.0, 2.0, 3.0]);

    // Create 1000 vectors
    let vectors_data: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            vec![
                1.0 + i as f32 * 0.001,
                2.0 + i as f32 * 0.001,
                3.0 + i as f32 * 0.001,
            ]
        })
        .collect();
    let vectors = create_test_vectors_from_data(vectors_data);

    let ivf_list = IVFList::new(centroid.clone(), vectors);
    let shard = Shard::new(2100, vec![centroid], vec![ivf_list], 3);

    // Save
    shard.save().expect("Failed to save large shard");

    // Load
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2100, &[100])
            .await
            .expect("Failed to load large shard");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 1);
    let (_id, _centroid, loaded_vectors) = &loaded[0];

    // All 1000 vectors should be retrieved
    assert_eq!(loaded_vectors.len(), num_vectors);

    // Verify some vectors to ensure data integrity
    for i in [0, 500, 999] {
        let expected_val = 1.0 + i as f32 * 0.001;
        assert!((loaded_vectors[i].1[0] - expected_val).abs() < 1e-5);
    }

    // Cleanup
    cleanup_test_shards(&[2100]);
}

#[test]
#[serial]
fn test_shard_with_single_vector() {
    // Edge case: centroid with exactly one vector
    let centroid = Centroid::new(30, vec![1.0, 2.0]);
    let ivf_list = IVFList::new(
        centroid.clone(),
        create_test_vectors_from_data(vec![vec![1.1, 2.1]]),
    );

    let shard = Shard::new(2004, vec![centroid], vec![ivf_list], 2);
    shard.save().expect("Failed to save shard");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2004, &[30])
            .await
            .expect("Failed to load shard");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].2.len(), 1); // One vector

    // Cleanup
    cleanup_test_shards(&[2004]);
}

// ============================================================================
// Data Integrity Tests
// ============================================================================

#[test]
#[serial]
fn test_vector_metadata_preserved() {
    // Test that vector metadata (IDs, timestamps) is preserved through save/load
    let centroid = Centroid::new(200, vec![1.0, 2.0]);

    // Create vectors with specific metadata
    let vectors = vec![
        create_test_vector(100, 1001, vec![1.1, 2.1], 1234567890),
        create_test_vector(101, 1002, vec![1.2, 2.2], 1234567891),
        create_test_vector(102, 1003, vec![1.3, 2.3], 1234567892),
    ];

    let ivf_list = IVFList::new(centroid.clone(), vectors);
    let shard = Shard::new(2200, vec![centroid], vec![ivf_list], 2);

    // Save
    shard.save().expect("Failed to save shard");

    // Load
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2200, &[200])
            .await
            .expect("Failed to load shard");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 1);
    let (_id, _centroid, loaded_vectors) = &loaded[0];

    assert_eq!(loaded_vectors.len(), 3);

    // Verify metadata is preserved
    let expected_metadata = vec![
        (100u64, 1001u64, 1234567890u64),
        (101u64, 1002u64, 1234567891u64),
        (102u64, 1003u64, 1234567892u64),
    ];

    for (i, (meta, _vec)) in loaded_vectors.iter().enumerate() {
        assert_eq!(meta.id, expected_metadata[i].0, "Vector ID mismatch");
        assert_eq!(
            meta.external_id, expected_metadata[i].1,
            "External ID mismatch"
        );
        assert_eq!(meta.timestamp, expected_metadata[i].2, "Timestamp mismatch");
    }

    // Cleanup
    cleanup_test_shards(&[2200]);
}

#[test]
#[serial]
fn test_centroid_id_non_sequential() {
    // Test that non-sequential centroid IDs work correctly
    let centroid1 = Centroid::new(100, vec![1.0, 2.0]);
    let centroid2 = Centroid::new(250, vec![3.0, 4.0]);
    let centroid3 = Centroid::new(500, vec![5.0, 6.0]);

    let ivf1 = IVFList::new(
        centroid1.clone(),
        create_test_vectors_from_data(vec![vec![1.0, 2.0]]),
    );
    let ivf2 = IVFList::new(
        centroid2.clone(),
        create_test_vectors_from_data(vec![vec![3.0, 4.0]]),
    );
    let ivf3 = IVFList::new(
        centroid3.clone(),
        create_test_vectors_from_data(vec![vec![5.0, 6.0]]),
    );

    let shard = Shard::new(
        2005,
        vec![centroid1, centroid2, centroid3],
        vec![ivf1, ivf2, ivf3],
        2,
    );
    shard.save().expect("Failed to save shard");

    // Load by non-sequential IDs
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2005, &[250, 500])
            .await
            .expect("Failed to load non-sequential centroids");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 2);
    let ids: Vec<u64> = loaded.iter().map(|(id, _, _)| *id).collect();
    assert!(ids.contains(&250));
    assert!(ids.contains(&500));

    // Cleanup
    cleanup_test_shards(&[2005]);
}

#[test]
#[serial]
fn test_float_precision_preserved() {
    // Test that float values maintain precision through save/load
    let precise_values = vec![0.123456789, -0.987654321, 1234.5678, 0.0000001];

    let centroid = Centroid::new(40, precise_values.clone());
    let ivf_list = IVFList::new(
        centroid.clone(),
        create_test_vectors_from_data(vec![precise_values.clone()]),
    );

    let shard = Shard::new(2006, vec![centroid], vec![ivf_list], 4);
    shard.save().expect("Failed to save shard");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2006, &[40])
            .await
            .expect("Failed to load shard");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    let (_id, loaded_centroid, loaded_vectors) = &loaded[0];

    // Check centroid precision
    for (i, &expected) in precise_values.iter().enumerate() {
        let diff = (loaded_centroid[i] - expected).abs();
        assert!(
            diff < 1e-6,
            "Precision lost: expected {}, got {}",
            expected,
            loaded_centroid[i]
        );
    }

    // Check vector precision
    for (i, &expected) in precise_values.iter().enumerate() {
        let diff = (loaded_vectors[0].1[i] - expected).abs();
        assert!(diff < 1e-6, "Vector precision lost");
    }

    // Cleanup
    cleanup_test_shards(&[2006]);
}

#[test]
#[serial]
fn test_large_centroid_ids() {
    // Test with very large centroid IDs
    let large_id = u64::MAX - 1000;
    let centroid = Centroid::new(large_id as usize, vec![1.0, 2.0]);
    let ivf_list = IVFList::new(
        centroid.clone(),
        create_test_vectors_from_data(vec![vec![1.0, 2.0]]),
    );

    let shard = Shard::new(2007, vec![centroid], vec![ivf_list], 2);
    shard.save().expect("Failed to save shard with large ID");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2007, &[large_id])
            .await
            .expect("Failed to load shard with large ID");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, large_id);

    // Cleanup
    cleanup_test_shards(&[2007]);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
#[serial]
fn test_load_nonexistent_shard() {
    // Test that loading a non-existent shard returns an error
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(99999, &[0]).await;
        tx.send(result).unwrap();
    });
    let result = rx.recv().unwrap();

    assert!(
        result.is_err(),
        "Loading non-existent shard should return error"
    );
}

#[test]
#[serial]
fn test_load_centroid_not_in_shard() {
    // Test that requesting a centroid that doesn't exist returns error
    let centroid = Centroid::new(50, vec![1.0, 2.0]);
    let ivf_list = IVFList::new(
        centroid.clone(),
        create_test_vectors_from_data(vec![vec![1.0, 2.0]]),
    );

    let shard = Shard::new(2008, vec![centroid], vec![ivf_list], 2);
    shard.save().expect("Failed to save shard");

    // Try to load centroid that doesn't exist
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2008, &[999]).await;
        tx.send(result).unwrap();
    });
    let result = rx.recv().unwrap();

    assert!(
        result.is_err(),
        "Loading non-existent centroid should return error"
    );

    // Cleanup
    cleanup_test_shards(&[2008]);
}

#[test]
#[serial]
fn test_corrupted_shard_header() {
    // Test that corrupted shard headers are handled gracefully
    use std::fs::OpenOptions;
    use std::io::Seek;

    // Create a valid shard first
    let centroid = Centroid::new(300, vec![1.0, 2.0]);
    let vectors = create_test_vectors_from_data(vec![vec![1.1, 2.1]]);
    let ivf_list = IVFList::new(centroid.clone(), vectors);
    let shard = Shard::new(2300, vec![centroid], vec![ivf_list], 2);

    shard.save().expect("Failed to save shard");

    // Corrupt the header by overwriting first few bytes
    let shard_path = "shards/shard_2300.bin";
    let mut file = OpenOptions::new()
        .write(true)
        .open(shard_path)
        .expect("Failed to open shard file");

    file.seek(std::io::SeekFrom::Start(0))
        .expect("Failed to seek");
    std::io::Write::write_all(&mut file, &[0xFF, 0xFF, 0xFF, 0xFF])
        .expect("Failed to corrupt file");
    drop(file);

    // Try to load the corrupted shard
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2300, &[300]).await;
        tx.send(result).unwrap();
    });
    let result = rx.recv().unwrap();

    // Should return error, not panic
    assert!(
        result.is_err(),
        "Loading corrupted shard should return error"
    );

    // Cleanup
    cleanup_test_shards(&[2300]);
}

#[test]
#[serial]
fn test_load_from_disk_full_shard() {
    // Test the load_from_disk method
    let centroid1 = Centroid::new(60, vec![1.0, 2.0, 3.0]);
    let centroid2 = Centroid::new(61, vec![4.0, 5.0, 6.0]);

    let ivf1 = IVFList::new(
        centroid1.clone(),
        create_test_vectors_from_data(vec![vec![1.1, 2.1, 3.1]]),
    );
    let ivf2 = IVFList::new(
        centroid2.clone(),
        create_test_vectors_from_data(vec![vec![4.1, 5.1, 6.1]]),
    );

    let original_shard = Shard::new(2009, vec![centroid1, centroid2], vec![ivf1, ivf2], 3);
    original_shard.save().expect("Failed to save shard");

    // Load entire shard
    let loaded_shard = Shard::load_from_disk(2009).expect("Failed to load shard from disk");

    assert_eq!(loaded_shard.id, 2009);
    assert_eq!(loaded_shard.dimension, 3);
    assert_eq!(loaded_shard.centroids.len(), 2);
    assert_eq!(loaded_shard.ivf_lists.len(), 2);

    // Verify structure
    assert!(verify_shard_structure(&loaded_shard));

    // Cleanup
    cleanup_test_shards(&[2009]);
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
#[serial]
fn test_multiple_saves_same_id() {
    // Test that saving twice with same ID overwrites
    let centroid1 = Centroid::new(70, vec![1.0, 2.0]);
    let ivf1 = IVFList::new(
        centroid1.clone(),
        create_test_vectors_from_data(vec![vec![1.0, 2.0]]),
    );
    let shard1 = Shard::new(2010, vec![centroid1], vec![ivf1], 2);

    shard1.save().expect("First save failed");

    // Save different data with same shard ID
    let centroid2 = Centroid::new(70, vec![3.0, 4.0]);
    let ivf2 = IVFList::new(
        centroid2.clone(),
        create_test_vectors_from_data(vec![vec![3.0, 4.0]]),
    );
    let shard2 = Shard::new(2010, vec![centroid2], vec![ivf2], 2);

    shard2.save().expect("Second save failed");

    // Load and verify we get the second version
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = Shard::get_centroid_vectors(2010, &[70])
            .await
            .expect("Failed to load");
        tx.send(result).unwrap();
    });
    let loaded = rx.recv().unwrap();

    let (_id, centroid, _vectors) = &loaded[0];
    assert!((centroid[0] - 3.0).abs() < 1e-5);
    assert!((centroid[1] - 4.0).abs() < 1e-5);

    // Cleanup
    cleanup_test_shards(&[2010]);
}

#[test]
#[serial]
fn test_shard_with_zero_dimension_invalid() {
    // Test that shards with dimension=0 fail validation
    let centroid = Centroid::new(400, vec![]);
    let ivf_list = IVFList::new(centroid.clone(), vec![]);

    let invalid_shard = Shard::new(2400, vec![centroid], vec![ivf_list], 0);

    // Structure validation should fail
    assert!(
        !verify_shard_structure(&invalid_shard),
        "Shard with dimension=0 should fail validation"
    );
}

#[test]
#[serial]
fn test_concurrent_shard_reads() {
    // Test that multiple threads can read from the same shard safely
    use std::thread;

    // Create and save shard
    let centroid = Centroid::new(80, vec![1.0, 2.0, 3.0, 4.0]);
    let ivf = IVFList::new(
        centroid.clone(),
        create_test_vectors_from_data(vec![vec![1.0, 2.0, 3.0, 4.0]; 10]),
    );
    let shard = Shard::new(2011, vec![centroid], vec![ivf], 4);
    shard.save().expect("Failed to save shard");

    // Spawn multiple threads to read
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let (tx, rx) = mpsc::channel();
                tokio_uring::start(async {
                    let loaded = Shard::get_centroid_vectors(2011, &[80])
                        .await
                        .expect("Failed to load in thread");
                    tx.send(loaded).unwrap();
                });
                let loaded = rx.recv().unwrap();
                assert_eq!(loaded.len(), 1);
                assert_eq!(loaded[0].2.len(), 10); // 10 vectors
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Cleanup
    cleanup_test_shards(&[2011]);
}
