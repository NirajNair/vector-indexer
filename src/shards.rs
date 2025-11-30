use crate::ivf_index::{Centroid, IVFList};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Error, ErrorKind, Result, Write};
use std::mem;
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
    pub vector_id: u64,
    pub timestamp: u64,
} // 16 bytes

impl Shard {
    pub fn new(id: u64, centroids: Vec<Centroid>, ivf_lists: Vec<IVFList>, dimension: u32) -> Self {
        Shard {
            id,
            centroids,
            ivf_lists,
            dimension,
        }
    }

    pub fn save(&self) -> Result<()> {
        fs::create_dir_all("shards")?;
        let mut file = File::create(format!("shards/shard_{}.bin", self.id))?;

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
            // Calculate size of this cluster block
            let cluster_size = centroid_vector_size
                + ivf_list.vectors.len() * (vector_meta_size + data_vector_size);

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
            cluster_block.extend_from_slice(IntoBytes::as_bytes(&ivf_list.centroid.vector[..]));

            // 2. Write each vector with its metadata
            for (vec_idx, vector) in ivf_list.vectors.iter().enumerate() {
                // Write vector metadata
                let vector_meta = VectorMeta {
                    vector_id: vec_idx as u64, // Or use actual vector ID if available
                    timestamp: 0,
                };
                cluster_block.extend_from_slice(vector_meta.as_bytes());

                // Write vector data
                cluster_block.extend_from_slice(IntoBytes::as_bytes(&vector[..]));
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
    pub fn get_centroid_vectors(
        shard_id: u64,
        centroid_ids: &[u64],
    ) -> Result<Vec<(u64, Vec<f32>, Vec<(VectorMeta, Vec<f32>)>)>> {
        let file = File::open(format!("shards/shard_{}.bin", shard_id))?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read header
        let (header, _) = ShardHeader::ref_from_prefix(&mmap[..mem::size_of::<ShardHeader>()])
            .or_else(|_| Err(Error::new(ErrorKind::InvalidData, "Invalid shard header")))?;

        println!(
            "Reading shard {}: {} centroids, {} dimensions",
            header.shard_id, header.num_centroids, header.dimensions
        );

        // Read index array
        let index_start = header.index_offset as usize;
        let index_entry_size = mem::size_of::<CentroidIndex>();
        let mut index_entries = Vec::new();

        for i in 0..header.num_centroids as usize {
            let offset = index_start + i * index_entry_size;
            let (entry, _) =
                CentroidIndex::ref_from_prefix(&mmap[offset..offset + index_entry_size])
                    .or_else(|_| Err(Error::new(ErrorKind::InvalidData, "Invalid index entry")))?;
            index_entries.push(*entry);
        }

        // Find and read requested centroids
        let mut results = Vec::new();

        for &target_id in centroid_ids {
            // Find index entry for this centroid
            let index_entry = index_entries
                .iter()
                .find(|e| e.centroid_id == target_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::NotFound,
                        format!("Centroid {} not found in shard", target_id),
                    )
                })?;

            // Read cluster block
            let block_start = index_entry.data_offset as usize;
            let block_end = block_start + index_entry.data_size as usize;
            let cluster_block = &mmap[block_start..block_end];

            // Parse cluster block
            let dims = header.dimensions as usize;
            let centroid_size = dims * mem::size_of::<f32>();
            let vector_meta_size = mem::size_of::<VectorMeta>();
            let vector_data_size = dims * mem::size_of::<f32>();

            // 1. Read centroid vector
            let centroid_bytes = &cluster_block[..centroid_size];
            let centroid_vector = <[f32]>::ref_from_bytes(centroid_bytes)
                .or_else(|_| {
                    Err(Error::new(
                        ErrorKind::InvalidData,
                        "Invalid centroid vector",
                    ))
                })?
                .to_vec();

            // 2. Read vectors with metadata
            let mut vectors = Vec::new();
            let mut offset = centroid_size;

            for _ in 0..index_entry.num_vectors {
                // Read vector metadata
                let (meta, _) =
                    VectorMeta::ref_from_prefix(&cluster_block[offset..]).or_else(|_| {
                        Err(Error::new(
                            ErrorKind::InvalidData,
                            "Invalid vector metadata",
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

                vectors.push((*meta, vector));
            }

            results.push((target_id, centroid_vector, vectors));
            println!(
                "Loaded centroid {}: {} vectors",
                target_id, index_entry.num_vectors
            );
        }

        Ok(results)
    }

    /// Load entire shard from disk
    pub fn load_from_disk(shard_id: u64) -> Result<Self> {
        let file = File::open(format!("shards/shard_{}.bin", shard_id))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (header, _) = ShardHeader::ref_from_prefix(&mmap[..mem::size_of::<ShardHeader>()])
            .or_else(|_| Err(Error::new(ErrorKind::InvalidData, "Invalid header")))?;

        // Read all centroids
        let all_centroid_ids: Vec<u64> = (0..header.num_centroids as u64).collect();
        let cluster_data = Self::get_centroid_vectors(shard_id, &all_centroid_ids)?;

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
            let mut ids = Vec::new();
            for (meta, vec) in vectors_with_meta {
                ids.push(meta.vector_id as usize);
                vectors.push(vec);
            }

            let ivf_list = IVFList {
                centroid,
                vectors,
                ids,
            };
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
