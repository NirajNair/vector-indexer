use vector_indexer::ivf_index::IvfIndex;
use vector_indexer::kmeans::*;
use vector_indexer::utils::*;
use vector_indexer::vector_store::VectorStore;
use rand::Rng;
use std::fs::File;
use std::io::{Write, Read};

fn main() {
    println!("Starting vector indexer.");

    // Generate and save test vectors if file doesn't exist
    let vector_file = "test_vectors.bin";
    if !std::path::Path::new(vector_file).exists() {
        println!("Generating 10000 test vectors...");
        generate_test_vectors(vector_file, 10000, 128);
        println!("Test vectors saved to {}", vector_file);
    }

    // Read vector data from file
    println!("Reading vectors from file...");
    let vectors = read_vectors_from_file(vector_file).expect("Failed to read vectors");
    println!("Loaded {} vectors", vectors.len());

    let vector_store: VectorStore = VectorStore::new(vectors);

    let k = calculate_num_clusters(vector_store.data.len());
    let max_iters = calculate_max_iterations(vector_store.data.len());
    println!("Calculated k: {}, max_iters: {}", k, max_iters);

    let vector_arr = vector_store.get_vectors();
    println!("Vector array shape: {}x{}", vector_arr.nrows(), vector_arr.ncols());

    let (centroids, labels) =
        run_kmeans_parallel(&vector_arr, k, max_iters, None).expect("Failed to run KMeans");

    println!("Centroids shape: {}x{}", centroids.nrows(), centroids.ncols());

    let _ivf_index = IvfIndex::create_index(&vector_arr, &centroids, &labels);
}

/// Generate test vectors and save them to a file
/// Each vector is a tuple of (id: u64, vector: Vec<f32>, metadata: u64)
fn generate_test_vectors(filename: &str, count: usize, dimension: usize) {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(count);
    
    for i in 0..count {
        let id = i as u64;
        let vector: Vec<f32> = (0..dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let metadata = rng.gen::<u64>();
        
        vectors.push((id, vector, metadata));
    }
    
    // Serialize and write to file using bincode
    let encoded = bincode::encode_to_vec(&vectors, bincode::config::standard())
        .expect("Failed to encode vectors");
    
    let mut file = File::create(filename).expect("Failed to create file");
    file.write_all(&encoded).expect("Failed to write to file");
}

/// Read vectors from a binary file
fn read_vectors_from_file(filename: &str) -> Result<Vec<(u64, Vec<f32>, u64)>, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let (vectors, _): (Vec<(u64, Vec<f32>, u64)>, usize) = 
        bincode::decode_from_slice(&buffer, bincode::config::standard())?;
    
    Ok(vectors)
}
