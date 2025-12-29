use std::sync::mpsc;
use vector_indexer::{SearchRequest, VectorIndexer, VectorIndexerConfig, VectorRecord};

fn temp_subdir(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("vector_indexer_{}_{}", name, nanos))
}

fn make_records(dim: u32, n: u64) -> Vec<VectorRecord> {
    (0..n)
        .map(|i| {
            let values = (0..dim as usize)
                .map(|j| ((i as f32) * 0.01) + (j as f32))
                .collect::<Vec<f32>>();
            VectorRecord {
                external_id: i,
                values,
                timestamp: None,
            }
        })
        .collect()
}

#[test]
fn config_new_sets_expected_defaults() {
    let cfg = VectorIndexerConfig::new(8);
    assert_eq!(cfg.dimension, 8);
    assert_eq!(cfg.index_dir.to_string_lossy(), "index");
    assert_eq!(cfg.shards_dir.to_string_lossy(), "shards");
    assert_eq!(cfg.default_k, 10);
    assert_eq!(cfg.default_n_probe, 20);
    assert_eq!(cfg.max_k, 10_000);
    assert_eq!(cfg.max_n_probe, 10_000);
}

#[test]
fn build_writes_to_configured_dirs_and_load_uses_them() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("index");
    let shards_dir = temp_subdir("shards");

    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir.clone())
        .with_shards_dir(shards_dir.clone());

    let records = make_records(dim, 150);
    let query_external_id = 42u64;
    let query_vec = records[query_external_id as usize].values.clone();

    let _built = VectorIndexer::new(cfg.clone())
        .build_from_records(records)
        .expect("build_from_records failed");

    // Assert index file exists where configured.
    assert!(index_dir.join("index.bin").exists());

    // Assert shard files exist where configured.
    let mut shard_count = 0usize;
    if let Ok(rd) = std::fs::read_dir(&shards_dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("shard_") && name.ends_with(".bin") {
                shard_count += 1;
            }
        }
    }
    assert!(shard_count > 0);

    // Load + search roundtrip
    let loaded = VectorIndexer::load(cfg).expect("load failed");
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let results = loaded
            .search(SearchRequest {
                query: query_vec,
                k: 10,
                n_probe: 50,
                include_vectors: false,
            })
            .await
            .expect("search failed");
        tx.send(results).unwrap();
    });
    let results = rx.recv().unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].external_id, query_external_id);
    assert!(results[0].distance >= 0.0);
}

#[test]
fn search_uses_config_defaults_when_request_is_none() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("index");
    let shards_dir = temp_subdir("shards");

    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir)
        .with_shards_dir(shards_dir);

    let records = make_records(dim, 40);
    let query_vec = records[0].values.clone();
    let indexer = VectorIndexer::new(cfg)
        .build_from_records(records)
        .expect("build failed");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let results = indexer
            .search(SearchRequest {
                query: query_vec,
                k: indexer.config().default_k,
                n_probe: indexer.config().default_n_probe,
                include_vectors: false,
            })
            .await
            .expect("search failed");
        tx.send(results).unwrap();
    });
    let results = rx.recv().unwrap();

    // default_k is 10; dataset is larger.
    assert_eq!(results.len(), 10);
}

#[test]
fn search_overrides_k_and_n_probe_when_provided() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("index");
    let shards_dir = temp_subdir("shards");

    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir)
        .with_shards_dir(shards_dir);

    let records = make_records(dim, 60);
    let query_vec = records[0].values.clone();
    let indexer = VectorIndexer::new(cfg)
        .build_from_records(records)
        .expect("build failed");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let results = indexer
            .search(SearchRequest {
                query: query_vec,
                k: 5,
                n_probe: 999,
                include_vectors: false,
            })
            .await
            .expect("search failed");
        tx.send(results).unwrap();
    });
    let results = rx.recv().unwrap();

    assert_eq!(results.len(), 5);
}

#[test]
fn search_clamps_to_max_k_and_max_n_probe() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("index");
    let shards_dir = temp_subdir("shards");

    let mut cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir)
        .with_shards_dir(shards_dir);
    cfg.max_k = 3;
    cfg.max_n_probe = 1;

    let records = make_records(dim, 80);
    let query_vec = records[0].values.clone();
    let indexer = VectorIndexer::new(cfg)
        .build_from_records(records)
        .expect("build failed");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let results = indexer
            .search(SearchRequest {
                query: query_vec,
                k: 10,
                n_probe: 999,
                include_vectors: false,
            })
            .await
            .expect("search failed");
        tx.send(results).unwrap();
    });
    let results = rx.recv().unwrap();

    assert_eq!(results.len(), 3);
}

#[test]
fn include_vectors_controls_payload() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("index");
    let shards_dir = temp_subdir("shards");

    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir)
        .with_shards_dir(shards_dir);

    let records = make_records(dim, 50);
    let query_vec = records[0].values.clone();
    let indexer = VectorIndexer::new(cfg)
        .build_from_records(records)
        .expect("build failed");

    let (tx1, rx1) = mpsc::channel();
    tokio_uring::start(async {
        let results = indexer
            .search(SearchRequest {
                query: query_vec.clone(),
                k: 1,
                n_probe: 10,
                include_vectors: false,
            })
            .await
            .expect("search failed");
        tx1.send(results).unwrap();
    });
    let without = rx1.recv().unwrap();
    assert_eq!(without.len(), 1);
    assert!(without[0].vector.is_none());

    let (tx2, rx2) = mpsc::channel();
    tokio_uring::start(async {
        let results = indexer
            .search(SearchRequest {
                query: query_vec,
                k: 1,
                n_probe: 10,
                include_vectors: true,
            })
            .await
            .expect("search failed");
        tx2.send(results).unwrap();
    });
    let with = rx2.recv().unwrap();
    assert_eq!(with.len(), 1);
    assert!(with[0].vector.is_some());
    assert_eq!(with[0].vector.as_ref().unwrap().len(), dim as usize);
}

#[test]
fn load_fails_for_missing_index() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("missing_index");
    let shards_dir = temp_subdir("missing_shards");

    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir)
        .with_shards_dir(shards_dir);

    assert!(VectorIndexer::load(cfg).is_err());
}

#[test]
fn build_from_records_errors_on_empty_input() {
    let cfg = VectorIndexerConfig::new(8)
        .with_index_dir(temp_subdir("index"))
        .with_shards_dir(temp_subdir("shards"));

    assert!(VectorIndexer::new(cfg).build_from_records(vec![]).is_err());
}

#[test]
fn search_errors_on_query_dimension_mismatch() {
    let dim: u32 = 8;
    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(temp_subdir("index"))
        .with_shards_dir(temp_subdir("shards"));

    let records = make_records(dim, 30);
    let indexer = VectorIndexer::new(cfg)
        .build_from_records(records)
        .expect("build failed");

    // Wrong dimension (7 instead of 8)
    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let result = indexer
            .search(SearchRequest {
                query: vec![0.0; 7],
                k: 5,
                n_probe: 5,
                include_vectors: false,
            })
            .await;
        tx.send(result).unwrap();
    });
    assert!(rx.recv().unwrap().is_err());
}

#[test]
fn search_errors_on_k_or_n_probe_zero() {
    let dim: u32 = 8;
    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(temp_subdir("index"))
        .with_shards_dir(temp_subdir("shards"));

    let records = make_records(dim, 30);
    let query_vec = records[0].values.clone();
    let indexer = VectorIndexer::new(cfg)
        .build_from_records(records)
        .expect("build failed");

    let (tx1, rx1) = mpsc::channel();
    tokio_uring::start(async {
        let result = indexer
            .search(SearchRequest {
                query: query_vec.clone(),
                k: 0,
                n_probe: 5,
                include_vectors: false,
            })
            .await;
        tx1.send(result).unwrap();
    });
    assert!(rx1.recv().unwrap().is_err());

    let (tx2, rx2) = mpsc::channel();
    tokio_uring::start(async {
        let result = indexer
            .search(SearchRequest {
                query: query_vec,
                k: 5,
                n_probe: 0,
                include_vectors: false,
            })
            .await;
        tx2.send(result).unwrap();
    });
    assert!(rx2.recv().unwrap().is_err());
}

#[test]
fn build_from_vector_file_smoke_and_dimension_validation() {
    let dim: u32 = 8;
    let index_dir = temp_subdir("index");
    let shards_dir = temp_subdir("shards");

    // Write one batch in the same format as utils::read_vectors_from_file expects:
    // a bincode-encoded Vec<(u64, Vec<f32>, u64)> appended to a file.
    let vector_file = temp_subdir("vectors").join("vectors.bin");
    std::fs::create_dir_all(vector_file.parent().unwrap()).unwrap();

    let batch: Vec<(u64, Vec<f32>, u64)> = (0..50u64)
        .map(|i| (i, vec![i as f32; dim as usize], 0u64))
        .collect();
    let encoded = bincode::encode_to_vec(&batch, bincode::config::standard()).unwrap();
    std::fs::write(&vector_file, encoded).unwrap();

    let cfg = VectorIndexerConfig::new(dim)
        .with_index_dir(index_dir)
        .with_shards_dir(shards_dir);

    let indexer = VectorIndexer::new(cfg)
        .build_from_vector_file(&vector_file)
        .expect("build_from_vector_file failed");

    let (tx, rx) = mpsc::channel();
    tokio_uring::start(async {
        let results = indexer
            .search(SearchRequest {
                query: vec![0.0; dim as usize],
                k: 5,
                n_probe: 10,
                include_vectors: false,
            })
            .await
            .expect("search failed");
        tx.send(results).unwrap();
    });
    let results = rx.recv().unwrap();
    assert_eq!(results.len(), 5);

    // Dimension mismatch should error.
    let bad_cfg = VectorIndexerConfig::new(dim + 1)
        .with_index_dir(temp_subdir("index_bad"))
        .with_shards_dir(temp_subdir("shards_bad"));
    assert!(VectorIndexer::new(bad_cfg)
        .build_from_vector_file(&vector_file)
        .is_err());
}
