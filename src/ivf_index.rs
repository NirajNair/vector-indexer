use crate::kmeans::run_kmeans_mini_batch;
use crate::shards::Shard;
use crate::utils::{calculate_max_iterations, calculate_num_clusters, euclidean_distance_squared};
use crate::vector_store::VectorStore;
use ndarray::{Array1, Axis};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Error, ErrorKind, Read, Result, Write};

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
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
    pub fn new(dim: u32) -> Self {
        IvfIndex {
            centroids: Array1::default(0),
            centroids_to_shard: Array1::default(0),
            dimension: dim,
        }
    }

    pub fn fit(&mut self, vector_store: &VectorStore) {
        let k = calculate_num_clusters(vector_store.data.len());
        let max_iters = calculate_max_iterations(vector_store.data.len());
        println!("Calculated k: {}, max_iters: {}", k, max_iters);

        let vector_arr = vector_store.get_vectors();
        println!(
            "Vector array shape: {}x{}",
            vector_arr.nrows(),
            vector_arr.ncols()
        );

        let (centroids_arr, labels) =
            run_kmeans_mini_batch(&vector_arr, k, max_iters, None).expect("Failed to run KMeans");

        println!(
            "Centroids shape: {}x{}",
            centroids_arr.nrows(),
            centroids_arr.ncols()
        );

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
        for (i, vector) in vector_arr.axis_iter(Axis(0)).enumerate() {
            // Find ivf list correspoding to centroid id (labels[i])
            ivf_lists[labels[i]].vectors.push(vector.to_vec());
            ivf_lists[labels[i]].ids.push(i);
        }

        // Run kmeans to find super centroids to group similar centroids together
        let num_shards = (centroids.len() as f32).sqrt().ceil() as usize;
        let (super_centroids, super_centroid_labels) =
            run_kmeans_mini_batch(&centroids_arr, num_shards, 100, None)
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

        let shards_len = shards.len();
        for shard in shards {
            if let Err(e) = shard.save() {
                eprintln!("Failed to write shard {} to disk: {}", shard.id, e);
            }
        }
        println!("{} shards written to disk", shards_len);

        self.centroids = centroids;
        self.centroids_to_shard = centroids_to_shard;
        self.dimension = dimension;
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize, // Number of centroids to search
    ) -> Result<Vec<(usize, f32)>> {
        // Find n_probe nearest centroids to query
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist = euclidean_distance_squared(query, &centroid.vector);
                (i, dist)
            })
            .collect();

        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let probe_centroids: Vec<usize> = centroid_distances
            .iter()
            .take(n_probe)
            .map(|(idx, _)| *idx)
            .collect();

        // Get shard IDs for these centroids
        let shard_ids: std::collections::HashSet<usize> = probe_centroids
            .iter()
            .map(|&c_id| self.centroids_to_shard[c_id])
            .collect();

        // Load shards and search vectors
        let mut candidates = Vec::new();

        for &shard_id in &shard_ids {
            // Get centroids in this shard that we want to probe
            // Use the actual centroid IDs, not the array indices
            let relevant_centroids: Vec<u64> = probe_centroids
                .iter()
                .filter(|&&c_id| self.centroids_to_shard[c_id] == shard_id)
                .map(|&c_id| self.centroids[c_id].id as u64)
                .collect();

            // Read vectors from shard
            if let Ok(cluster_data) =
                Shard::get_centroid_vectors(shard_id as u64, &relevant_centroids)
            {
                for (_, _, vectors_with_metadata) in cluster_data {
                    for (metadata, vector) in vectors_with_metadata {
                        let dist = euclidean_distance_squared(query, &vector);
                        candidates.push((metadata.vector_id as usize, dist));
                    }
                }
            }
        }

        // Sort and return top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(candidates.into_iter().take(k).collect())
    }

    pub fn save(&self) -> Result<()> {
        // Create index directory if it doesn't exist
        fs::create_dir_all("index")?;

        // Serialize the index
        let encoded_data = bincode::serde::encode_to_vec(self, bincode::config::standard())
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
}

pub fn load_index() -> Result<IvfIndex> {
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
