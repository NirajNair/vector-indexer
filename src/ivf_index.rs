use crate::kmeans::run_kmeans_mini_batch;
use crate::shards::Shard;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Error, ErrorKind, Read, Result, Write};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Centroid {
    pub id: usize,
    pub vector: Vec<f32>,
}

impl Centroid {
    pub fn new(id: usize, vector: Vec<f32>) -> Self {
        Centroid { id, vector }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IVFList {
    pub centroid: Centroid,
    pub vectors: Vec<Vec<f32>>,
    pub ids: Vec<usize>,
}

impl IVFList {
    pub fn new(centroid: Centroid, vectors: Vec<Vec<f32>>, ids: Vec<usize>) -> Self {
        IVFList {
            centroid,
            vectors,
            ids,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    centroids: Array1<Centroid>,
    centroids_to_shard: Array1<usize>,
    dimension: u32,
}

impl IvfIndex {
    pub fn create_index(
        vectors: &Array2<f32>,
        centroids_arr: &Array2<f32>,
        labels: &Array1<usize>,
    ) -> Self {
        let centroids: Array1<Centroid> = centroids_arr
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, centroid)| Centroid::new(i, centroid.to_owned().to_vec()))
            .collect();

        let k = centroids.len();
        let dimension = centroids[0].vector.len() as u32;

        // Create IVF lists for each centroid
        let mut ivf_lists: Vec<IVFList> = Vec::new();
        for centroid in centroids.clone() {
            ivf_lists.push(IVFList::new(centroid, Vec::new(), Vec::new()));
        }

        // Assign vectors to IVF lists
        for (i, vector) in vectors.axis_iter(Axis(0)).enumerate() {
            // Find ivf list correspoding to centroid id (labels[i])
            ivf_lists[labels[i]].vectors.push(vector.to_vec());
            ivf_lists[labels[i]].ids.push(i);
        }

        // Run kmeans to find super centroids to group similar centroids together
        let num_shards = (centroids.len() as f32).sqrt().ceil() as usize;
        let (super_centroids, super_centroid_labels) =
            run_kmeans_mini_batch(centroids_arr, num_shards, 100, None)
                .expect("Failed to run kmeans");

        println!(
            "Super centroids shape: {}x{}",
            super_centroids.nrows(),
            super_centroids.ncols()
        );

        // Creates shards where no. of shards = no. of super centroids
        let mut shards: Vec<Shard> = (0..super_centroids.nrows())
            .map(|i| Shard::new(i as u64, Vec::new(), Vec::new(), dimension))
            .collect();

        // Assign centroids & IVF lists to shards based on super centroid labels
        let mut centroids_to_shard: Array1<usize> = Array1::from_elem(k, 0);
        for (_i, ivf_list) in ivf_lists.iter().enumerate() {
            let shard_id = super_centroid_labels[ivf_list.centroid.id.clone()];
            let centroid_id = ivf_list.centroid.id.clone();

            centroids_to_shard[centroid_id] = shard_id;

            shards[shard_id].ivf_lists.push(ivf_list.clone());
            shards[shard_id]
                .centroids
                .push(centroids[centroid_id].clone());
        }

        let index = IvfIndex {
            centroids,
            centroids_to_shard,
            dimension,
        };

        if let Err(e) = write_index_to_disk(&index) {
            eprintln!("Failed to write index to disk: {}", e);
        }
        let shards_len = shards.len();
        for shard in shards {
            if let Err(e) = shard.write_shard_to_disk() {
                eprintln!("Failed to write shard {} to disk: {}", shard.id, e);
            }
        }
        println!("{} shards written to disk", shards_len);

        index
    }
}

fn write_index_to_disk(index: &IvfIndex) -> Result<()> {
    // Create index directory if it doesn't exist
    fs::create_dir_all("index")?;

    // Serialize the index
    let encoded_data = bincode::serde::encode_to_vec(index, bincode::config::standard())
        .map_err(|e| Error::new(ErrorKind::Other, format!("Bincode encoding error: {}", e)))?;

    // Write to disk
    println!("Writing IVF index to disk...");
    let mut file = File::create("index/index.bin")?;
    file.write_all(&encoded_data)?;
    println!(
        "IVF index written to index/index.bin ({} bytes)",
        encoded_data.len()
    );

    Ok(())
}

pub fn read_index_from_disk() -> Result<IvfIndex> {
    // Read file contents
    let mut file = File::open("index/index.bin")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    println!("Reading IVF index from disk ({} bytes)...", buffer.len());

    // Deserialize using bincode with serde compatibility
    let (index, _bytes_read): (IvfIndex, _) =
        bincode::serde::decode_from_slice(&buffer, bincode::config::standard())
            .map_err(|e| Error::new(ErrorKind::Other, format!("Bincode decoding error: {}", e)))?;

    println!("IVF index loaded successfully");
    Ok(index)
}
