use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

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

pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Generate test vectors and save them to a file
/// Each vector is a tuple of (id: u64, vector: Vec<f32>, metadata: u64)
pub fn generate_test_vectors_parallel(filename: &str, count: usize, dimension: usize) {
    let batch_size = 1000;
    let total_batches = (count + batch_size - 1) / batch_size;
    let filename = filename.to_string();

    let (sender, receiver) = sync_channel::<Vec<u8>>(5);

    let writer_thread = thread::spawn(move || {
        let mut file = File::create(&filename).expect("Failed to create file");
        let mut batch_count = 0;

        for encoded_batch in receiver {
            file.write_all(&encoded_batch).expect("Failed to write");
            batch_count += 1;
            println!("Written batch {}/{}", batch_count, total_batches);
        }
    });

    (0..total_batches)
        .into_par_iter()
        .for_each_with(sender.clone(), |s, batch_num| {
            let start_idx = batch_num * batch_size;
            let end_idx = ((batch_num + 1) * batch_size).min(count);
            let current_batch_size = end_idx - start_idx;

            let mut rng = rand::thread_rng();
            let mut vectors = Vec::with_capacity(current_batch_size);

            for i in start_idx..end_idx {
                let id = i as u64;
                let vector: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let metadata = rng.gen::<u64>();

                vectors.push((id, vector, metadata));
            }

            // Serialize the batch
            let encoded = bincode::encode_to_vec(&vectors, bincode::config::standard())
                .expect("Failed to encode vectors");

            s.send(encoded).expect("Failed to send encoded vectors");
        });

    drop(sender);
    writer_thread.join().expect("Writer thread panicked");
}

/// Read vectors from a binary file (handles batched format)
pub fn read_vectors_from_file(
    filename: &str,
) -> Result<Vec<(u64, Vec<f32>, u64)>, Box<dyn std::error::Error>> {
    let mut file = File::open(filename)?;
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

pub fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}
