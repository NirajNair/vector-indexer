use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Generate deterministic vectors and save to disk using existing bincode format
pub fn generate_dataset(
    vector_count: usize,
    dimension: usize,
    seed: u64,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if dataset already exists
    if output_path.exists() {
        eprintln!("Dataset already exists at {:?}, skipping generation", output_path);
        return Ok(());
    }

    eprintln!(
        "Generating {} vectors of dimension {} with seed {}...",
        vector_count, dimension, seed
    );

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Use seeded RNG for deterministic generation
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate vectors in batches (compatible with existing format)
    let batch_size = 1000;
    let total_batches = (vector_count + batch_size - 1) / batch_size;

    let mut file = File::create(output_path)?;

    for batch_num in 0..total_batches {
        let start_idx = batch_num * batch_size;
        let end_idx = ((batch_num + 1) * batch_size).min(vector_count);
        let current_batch_size = end_idx - start_idx;

        let mut vectors = Vec::with_capacity(current_batch_size);

        for i in start_idx..end_idx {
            let id = i as u64;
            // Generate random vector values in range [-1.0, 1.0]
            let vector: Vec<f32> = (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            let metadata = rng.gen::<u64>();

            vectors.push((id, vector, metadata));
        }

        // Serialize the batch using bincode (compatible with read_vectors_from_file)
        let encoded = bincode::encode_to_vec(&vectors, bincode::config::standard())?;
        file.write_all(&encoded)?;

        if (batch_num + 1) % 100 == 0 || batch_num == total_batches - 1 {
            eprintln!(
                "Generated batch {}/{} ({} vectors)",
                batch_num + 1,
                total_batches,
                end_idx
            );
        }
    }

    eprintln!("Dataset saved to {:?}", output_path);
    Ok(())
}

/// Get the path for a dataset file based on parameters
pub fn dataset_path(vector_count: usize, dimension: u32, seed: u64) -> PathBuf {
    PathBuf::from("bench_data")
        .join(format!("{}_{}_{}.bin", vector_count, dimension, seed))
}

/// Load vectors from disk using existing format
pub fn load_dataset(
    path: &Path,
) -> Result<Vec<(u64, Vec<f32>, u64)>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut all_vectors = Vec::new();
    let mut offset = 0;

    // Read multiple batches that were appended to the file
    while offset < buffer.len() {
        match bincode::decode_from_slice::<Vec<(u64, Vec<f32>, u64)>, _>(
            &buffer[offset..],
            bincode::config::standard(),
        ) {
            Ok((vectors, bytes_read)) => {
                all_vectors.extend(vectors);
                offset += bytes_read;
            }
            Err(_) => break,
        }
    }

    Ok(all_vectors)
}

/// Ensure dataset exists, generating it if necessary
pub fn ensure_dataset(
    vector_count: usize,
    dimension: u32,
    seed: u64,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = dataset_path(vector_count, dimension, seed);
    generate_dataset(vector_count, dimension as usize, seed, &path)?;
    Ok(path)
}

/// Get file size in bytes
pub fn get_file_size(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}

