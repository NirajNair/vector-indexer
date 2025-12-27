fn main() {
    println!("Starting vector_indexer demo.");

    let vector_file = "test_vectors_500k.bin";
    let dim: u32 = 768;

    let cfg = vector_indexer::VectorIndexerConfig::new(dim);

    // Load if present; otherwise build from the existing vector file format.
    // (This demo assumes `test_vectors.bin` exists or you created it earlier.)
    let indexer = match vector_indexer::VectorIndexer::load(cfg.clone()) {
        Ok(ix) => {
            println!("Loaded existing index from disk.");
            ix
        }
        Err(_) => {
            println!("No index found; building from vector file: {}", vector_file);
            vector_indexer::VectorIndexer::new(cfg)
                .build_from_vector_file(vector_file)
                .expect("Failed to build index")
        }
    };

    println!("\n=== Performing Search ===");
    // NOTE: This demo doesn't generate vectors anymore. Provide a query vector yourself.
    // For quick local testing you can reuse a vector from your own dataset.
    let query: Vec<f32> = vec![0.0; dim as usize];

    let results = indexer
        .search(indexer.search_request(query))
        .expect("Search failed");

    println!("Got {} results", results.len());
    for (i, r) in results.iter().enumerate() {
        println!(
            "  {}. external_id={}, distance={:.6}",
            i + 1,
            r.external_id,
            r.distance
        );
    }
}
