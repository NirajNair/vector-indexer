pub fn calculate_num_clusters(num_vectors: usize) -> usize {
    let clusters = match num_vectors {
        n if n < 10_000 => (n as f64).sqrt() as usize,
        n if n < 100_000 => 2 * (n as f64).sqrt().ceil() as usize,
        _ => 4 * (num_vectors as f64).sqrt().ceil() as usize,
    };
    clusters
}

pub fn calculate_max_iterations(num_vectors: usize) -> usize {
    let max_iterations = match num_vectors {
        n if n < 10_000 => 300,
        n if n < 100_000 => 100,
        n if n < 1_000_000 => 50,
        _ => 20,
    };
    max_iterations
}
