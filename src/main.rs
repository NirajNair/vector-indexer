use vector_indexer::ivf_index::IvfIndex;
use vector_indexer::kmeans::*;
use vector_indexer::utils::*;
use vector_indexer::vector_store::VectorStore;

fn main() {
    println!("Starting vector indexer.");
    // Add at start of main() or before kmeans call
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().unwrap().get())
        .build_global()
        .unwrap();

    // Generate and save test vectors if file doesn't exist
    let vector_file = "test_vectors.bin";
    if !std::path::Path::new(vector_file).exists() {
        let num_vectors = 1000000;
        println!("Generating {} test vectors...", num_vectors.clone());
        generate_test_vectors_parallel(vector_file, num_vectors, 768);
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
    println!(
        "Vector array shape: {}x{}",
        vector_arr.nrows(),
        vector_arr.ncols()
    );

    let (centroids, labels) =
        run_kmeans_mini_batch(&vector_arr, k, max_iters, None).expect("Failed to run KMeans");

    println!(
        "Centroids shape: {}x{}",
        centroids.nrows(),
        centroids.ncols()
    );

    let _ivf_index = IvfIndex::create_index(&vector_arr, &centroids, &labels);
}

