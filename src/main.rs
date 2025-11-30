use vector_indexer::ivf_index::{load_index, IvfIndex};
use vector_indexer::utils::*;
use vector_indexer::vector_store::VectorStore;

fn main() {
    println!("Starting vector indexer.");
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().unwrap().get())
        .build_global()
        .unwrap();

    // Generate and save test vectors if file doesn't exist
    let vector_file = "test_vectors.bin";
    let dim: u32 = 768;
    if !std::path::Path::new(vector_file).exists() {
        let num_vectors = 1000000;
        println!("Generating {} test vectors...", num_vectors.clone());
        generate_test_vectors_parallel(vector_file, num_vectors, dim as usize);
        println!("Test vectors saved to {}", vector_file);
    }

    // Read vector data from file
    println!("Reading vectors from file...");
    let vectors = read_vectors_from_file(vector_file).expect("Failed to read vectors");
    println!("Loaded {} vectors", vectors.len());

    let vector_store: VectorStore = VectorStore::new(vectors.clone());

    // Build or load index
    let ivf_index = if std::path::Path::new("index/index.bin").exists() {
        println!("Loading existing index...");
        load_index().expect("Failed to load index")
    } else {
        println!("Building new index...");
        let mut ivf_index = IvfIndex::new(dim);
        ivf_index.fit(&vector_store);

        if let Err(e) = ivf_index.save() {
            eprintln!("Failed to write index to disk: {}", e);
        }
        ivf_index
    };

    // Perform search
    println!("\n=== Performing Search ===");

    // Use the first vector as query (or generate random query)
    let query_vector = &vectors[0];
    println!("Query vector ID: 0");

    let k = 10; // Number of nearest neighbors
    let n_probe = 20; // Number of centroids to probe (higher = more accurate but slower)

    println!(
        "Searching for {} nearest neighbors (probing {} centroids)...",
        k, n_probe
    );

    match ivf_index.search(&query_vector.1, k, n_probe) {
        Ok(results) => {
            println!("\nTop {} results:", results.len());
            for (i, (vector_id, distance)) in results.iter().enumerate() {
                println!(
                    "  {}. Vector ID: {}, Distance: {:.6}",
                    i + 1,
                    vector_id,
                    distance
                );
            }
        }
        Err(e) => {
            eprintln!("Search failed: {}", e);
        }
    }

    // Additional search with different query
    println!("\n=== Testing with another query ===");
    if vectors.len() > 100 {
        let query_vector2 = &vectors[100];
        println!("Query vector ID: 100");

        match ivf_index.search(&query_vector2.1, k, n_probe) {
            Ok(results) => {
                println!("\nTop {} results:", results.len());
                for (i, (vector_id, distance)) in results.iter().enumerate() {
                    println!(
                        "  {}. Vector ID: {}, Distance: {:.6}",
                        i + 1,
                        vector_id,
                        distance
                    );
                }
            }
            Err(e) => {
                eprintln!("Search failed: {}", e);
            }
        }
    }
}
