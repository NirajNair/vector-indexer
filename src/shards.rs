use crate::ivf_index::{Centroid, IVFList};
use crate::vector_store::Vector;
use memmap2::Mmap;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Error, ErrorKind, Result, Write};
use std::mem;
use std::path::Path;
use tokio_uring::fs::File as AsyncFile;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[derive(Serialize, Deserialize, Debug)]
pub struct Shard {
    pub id: u64,
    pub centroids: Vec<Centroid>,
    pub ivf_lists: Vec<IVFList>,
    pub dimension: u32,
}

// Header at the start of the file
#[repr(C)]
#[derive(IntoBytes, FromBytes, Immutable, KnownLayout, Debug, Clone, Copy)]
struct ShardHeader {
    shard_id: u64,
    version: u64,
    dimensions: u32,
    num_centroids: u32,
    index_offset: u64, // Where CentroidIndex array starts
    data_offset: u64,  // Where first cluster block starts
} // 48 bytes

// Index entry for each centroid - allows O(1) lookup by centroid_id
#[repr(C)]
#[derive(IntoBytes, FromBytes, Immutable, KnownLayout, Debug, Clone, Copy)]
struct CentroidIndex {
    centroid_id: u64, // Actual centroid ID (non-sequential)
    num_vectors: u32,
    _padding: u32,
    data_offset: u64, // Absolute byte position of cluster block
    data_size: u64,   // Total bytes for this cluster block
} // 32 bytes

// Metadata stored with each vector in a cluster block
#[repr(C)]
#[derive(IntoBytes, FromBytes, Immutable, KnownLayout, Debug, Clone, Copy)]
pub struct VectorMeta {
    pub id: u64,
    pub external_id: u64,
    pub timestamp: u64,
}

impl Shard {
    pub fn new(id: u64, centroids: Vec<Centroid>, ivf_lists: Vec<IVFList>, dimension: u32) -> Self {
        Shard {
            id,
            centroids,
            ivf_lists,
            dimension,
        }
    }

    /// Backwards-compatible convenience wrapper that writes to `shards/`.
    pub fn save(&self) -> Result<()> {
        self.save_to(Path::new("shards"))
    }

    pub fn save_to(&self, shards_dir: &Path) -> Result<()> {
        fs::create_dir_all(shards_dir)?;

        // Remove existing file if it exists to avoid "File exists" error
        let shard_path = shards_dir.join(format!("shard_{}.bin", self.id));
        let _ = fs::remove_file(&shard_path); // Ignore error if file doesn't exist

        let mut file = File::create(&shard_path)?;

        let header_size = mem::size_of::<ShardHeader>();
        let index_entry_size = mem::size_of::<CentroidIndex>();
        let vector_meta_size = mem::size_of::<VectorMeta>();
        let centroid_vector_size = self.dimension as usize * mem::size_of::<f32>();
        let data_vector_size = self.dimension as usize * mem::size_of::<f32>();

        // Calculate offsets
        let index_offset = header_size;
        let index_size = index_entry_size * self.ivf_lists.len();
        let data_offset = index_offset + index_size;

        // Prepare and write header
        let header = ShardHeader {
            shard_id: self.id,
            version: 1,
            dimensions: self.dimension,
            num_centroids: self.ivf_lists.len() as u32,
            index_offset: index_offset as u64,
            data_offset: data_offset as u64,
        };
        file.write_all(header.as_bytes())?;

        // Prepare index entries and cluster blocks
        let mut index_entries = Vec::new();
        let mut cluster_blocks = Vec::new();
        let mut current_data_offset = data_offset as u64;

        for (_i, ivf_list) in self.ivf_lists.iter().enumerate() {
            // Calculate padding needed after centroid vector to align to 8 bytes
            let centroid_padding = (8 - (centroid_vector_size % 8)) % 8;

            // Calculate padding needed after each vector to align next VectorMeta to 8 bytes
            let vector_padding = (8 - (data_vector_size % 8)) % 8;

            // Calculate size of this cluster block
            let cluster_size = centroid_vector_size
                + centroid_padding
                + ivf_list.vectors.len() * (vector_meta_size + data_vector_size + vector_padding);

            // Create index entry
            let index_entry = CentroidIndex {
                centroid_id: ivf_list.centroid.id as u64,
                num_vectors: ivf_list.vectors.len() as u32,
                _padding: 0,
                data_offset: current_data_offset,
                data_size: cluster_size as u64,
            };
            index_entries.push(index_entry);

            // Build cluster block
            let mut cluster_block = Vec::new();

            // 1. Write centroid vector
            let centroid_slice: &[f32] = &ivf_list.centroid.vector;
            cluster_block.extend_from_slice(<[f32]>::as_bytes(centroid_slice));

            // Add padding to align to 8 bytes for VectorMeta
            cluster_block.extend_from_slice(&vec![0u8; centroid_padding]);

            // 2. Write each vector with its metadata
            for vector in ivf_list.vectors.iter() {
                // Write vector metadata
                let vector_meta = VectorMeta {
                    id: vector.id,                   // internal vector ID
                    external_id: vector.external_id, // external vector ID
                    timestamp: vector.timestamp,
                };
                cluster_block.extend_from_slice(vector_meta.as_bytes());

                // Write vector data
                let vector_slice = vector.data.as_slice().unwrap();
                cluster_block.extend_from_slice(<[f32]>::as_bytes(vector_slice));

                // Add padding after each vector to align next VectorMeta to 8 bytes
                let vector_padding = (8 - (data_vector_size % 8)) % 8;
                cluster_block.extend_from_slice(&vec![0u8; vector_padding]);
            }

            cluster_blocks.push(cluster_block);
            current_data_offset += cluster_size as u64;
        }

        // Write index entries
        for entry in &index_entries {
            file.write_all(entry.as_bytes())?;
        }

        // Write cluster blocks
        for block in &cluster_blocks {
            file.write_all(block)?;
        }

        // println!(
        //     "Shard {} written: {} centroids, {} total bytes",
        //     self.id,
        //     self.ivf_lists.len(),
        //     current_data_offset
        // );

        Ok(())
    }

    /// Read vectors from multiple centroids efficiently
    /// Returns: Vec of (centroid_id, centroid_vector, vectors_with_metadata)
    pub async fn get_centroid_vectors(
        shard_id: u64,
        centroid_ids: &[u64],
    ) -> Result<Vec<(u64, Vec<f32>, Vec<(VectorMeta, Vec<f32>)>)>> {
        Self::get_centroid_vectors_from(Path::new("shards"), shard_id, centroid_ids).await
    }

    pub async fn get_centroid_vectors_from(
        shards_dir: &Path,
        shard_id: u64,
        centroid_ids: &[u64],
    ) -> Result<Vec<(u64, Vec<f32>, Vec<(VectorMeta, Vec<f32>)>)>> {
        let file = AsyncFile::open(shards_dir.join(format!("shard_{}.bin", shard_id)))
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Other,
                    format!("Failed to open shard_{}.bin file: {}", shard_id, e),
                )
            })?;

        // Read header
        let header_buffer = vec![0u8; mem::size_of::<ShardHeader>()];
        let (result, header_buf) = file.read_at(header_buffer, 0).await;
        result.map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!(
                    "Failed to read header of shard_{}.bin file: {}",
                    shard_id, e
                ),
            )
        })?;

        let (header, _) = ShardHeader::ref_from_prefix(&header_buf).or_else(|_| {
            Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid shard header, reading shard_{}.bin.", shard_id),
            ))
        })?;

        // Validate that the shard ID in the file matches the expected shard ID
        if header.shard_id != shard_id {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Shard ID mismatch: expected {}, found {} in file",
                    shard_id, header.shard_id
                ),
            ));
        }

        // Read index array
        let index_start = header.index_offset as usize;
        let index_entry_size = mem::size_of::<CentroidIndex>();

        let index_buffer = vec![0u8; header.num_centroids as usize * index_entry_size];
        let (result, index_buf) = file.read_at(index_buffer, header.index_offset).await;
        result.map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("Failed to read index in shard_{}: {}", shard_id, e),
            )
        })?;

        let mut index_entries = Vec::new();
        for i in 0..header.num_centroids as usize {
            let offset = i * index_entry_size;
            let (entry, _) =
                CentroidIndex::ref_from_prefix(&index_buf[offset..offset + index_entry_size])
                    .or_else(|_| Err(Error::new(ErrorKind::InvalidData, "Invalid index entry")))?;
            index_entries.push(*entry);
        }

        let mut read_futures = Vec::new();
        for &target_id in centroid_ids {
            let index_entry = index_entries
                .iter()
                .find(|e| e.centroid_id == target_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::NotFound,
                        format!("Centroid {} not found", target_id),
                    )
                })?;

            let data_size = index_entry.data_size as usize;
            let data_offset = index_entry.data_offset;
            let read_future = file.read_at(vec![0u8; data_size], data_offset);

            read_futures.push((target_id, index_entry.clone(), read_future));
        }

        // Find and read requested centroids
        let mut results = Vec::new();
        let dims = header.dimensions as usize;
        let centroid_size = dims * mem::size_of::<f32>();
        let centroid_padding = (8 - (centroid_size % 8)) % 8;
        let vector_meta_size = mem::size_of::<VectorMeta>();
        let vector_data_size = dims * mem::size_of::<f32>();
        let vector_padding = (8 - (vector_data_size % 8)) % 8;

        for (target_id, index_entry, read_future) in read_futures {
            let (result, block_buf) = read_future.await;
            result.map_err(|e| {
                Error::new(ErrorKind::Other, format!("Failed to read cluster: {}", e))
            })?;

            // Parse cluster block (same as sync)
            let cluster_block = &block_buf[..];

            // Read centroid vector
            let centroid_bytes = &cluster_block[..centroid_size];
            let centroid_vector = <[f32]>::ref_from_bytes(centroid_bytes)
                .or_else(|_| {
                    Err(Error::new(
                        ErrorKind::InvalidData,
                        "Invalid centroid vector",
                    ))
                })?
                .to_vec();

            // Read vectors with metadata
            let mut vectors = Vec::new();
            let mut offset = centroid_size + centroid_padding;

            for i in 0..index_entry.num_vectors {
                // Read vector metadata
                // Check if we have enough bytes
                if offset + vector_meta_size > cluster_block.len() {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Not enough bytes for vector {} metadata: need {} bytes at offset {}, but cluster_block is {} bytes",
                                i, vector_meta_size, offset, cluster_block.len()),
                    ));
                }

                let (meta, _) =
                    VectorMeta::ref_from_prefix(&cluster_block[offset..]).or_else(|err| {
                        Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("Invalid vector metadata at offset {}: {:?}", offset, err),
                        ))
                    })?;
                offset += vector_meta_size;

                // Read vector data
                let vector_bytes = &cluster_block[offset..offset + vector_data_size];
                let vector = <[f32]>::ref_from_bytes(vector_bytes)
                    .or_else(|err| {
                        Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("Invalid vector data: {}", err),
                        ))
                    })?
                    .to_vec();
                offset += vector_data_size;

                // Skip padding after vector data
                offset += vector_padding;

                vectors.push((*meta, vector));
            }

            results.push((target_id, centroid_vector, vectors));
        }

        Ok(results)
    }

    /// Load entire shard from disk
    pub fn load_from_disk(shard_id: u64) -> Result<Self> {
        Self::load_from_disk_in(Path::new("shards"), shard_id)
    }

    pub fn load_from_disk_in(shards_dir: &Path, shard_id: u64) -> Result<Self> {
        let file = File::open(shards_dir.join(format!("shard_{}.bin", shard_id)))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (header, _) = ShardHeader::ref_from_prefix(&mmap[..mem::size_of::<ShardHeader>()])
            .or_else(|_| Err(Error::new(ErrorKind::InvalidData, "Invalid header")))?;

        // Validate that the shard ID in the file matches the expected shard ID
        if header.shard_id != shard_id {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Shard ID mismatch: expected {}, found {} in file",
                    shard_id, header.shard_id
                ),
            ));
        }

        // Read index array to get actual centroid IDs
        let index_start = header.index_offset as usize;
        let index_entry_size = mem::size_of::<CentroidIndex>();
        let mut all_centroid_ids = Vec::new();

        for i in 0..header.num_centroids as usize {
            let offset = index_start + i * index_entry_size;
            let (entry, _) =
                CentroidIndex::ref_from_prefix(&mmap[offset..offset + index_entry_size])
                    .or_else(|_| Err(Error::new(ErrorKind::InvalidData, "Invalid index entry")))?;
            all_centroid_ids.push(entry.centroid_id);
        }

        // Read all centroids using their actual IDs
        let cluster_data = tokio_uring::start(async {
            Self::get_centroid_vectors_from(shards_dir, shard_id, &all_centroid_ids).await
        })?;

        // Reconstruct Shard structure
        let mut centroids = Vec::new();
        let mut ivf_lists = Vec::new();

        for (centroid_id, centroid_vec, vectors_with_meta) in cluster_data {
            // Create Centroid (assuming your Centroid struct has these fields)
            let centroid = Centroid {
                id: centroid_id as usize,
                vector: centroid_vec,
            };
            centroids.push(centroid.clone());

            // Create IVFList
            let mut vectors = Vec::new();
            for (meta, vec) in vectors_with_meta {
                vectors.push(Vector::new(
                    meta.id,
                    meta.external_id,
                    Array1::from_vec(vec),
                    meta.timestamp,
                ));
            }

            let ivf_list = IVFList { centroid, vectors };
            ivf_lists.push(ivf_list);
        }

        Ok(Shard {
            id: header.shard_id,
            centroids,
            ivf_lists,
            dimension: header.dimensions,
        })
    }
}
