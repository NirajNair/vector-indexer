use crate::ivf_index::{load_index_from, IvfIndex};
use crate::utils::read_vectors_from_file;
use crate::vector_store::VectorStore;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

/// Configuration for vector indexer.
#[derive(Clone, Debug)]
pub struct VectorIndexerConfig {
    /// Vector dimension.
    pub dimension: u32,

    /// Directory for the index metadata file (default: `index/`).
    pub index_dir: PathBuf,

    /// Directory for shard files (default: `shards/`).
    pub shards_dir: PathBuf,

    /// Default `k` when caller doesn't specify.
    pub default_k: usize,

    /// Default `n_probe` when caller doesn't specify.
    pub default_n_probe: usize,

    /// Hard limit to protect CPU/memory in offline applications.
    pub max_k: usize,

    /// Hard limit to protect CPU/IO in offline applications.
    pub max_n_probe: usize,
}

impl VectorIndexerConfig {
    pub fn new(dimension: u32) -> Self {
        Self {
            dimension,
            index_dir: PathBuf::from("index"),
            shards_dir: PathBuf::from("shards"),
            default_k: 10,
            default_n_probe: 20,
            max_k: 10_000,
            max_n_probe: 10_000,
        }
    }

    pub fn with_index_dir(mut self, index_dir: impl Into<PathBuf>) -> Self {
        self.index_dir = index_dir.into();
        self
    }

    pub fn with_shards_dir(mut self, shards_dir: impl Into<PathBuf>) -> Self {
        self.shards_dir = shards_dir.into();
        self
    }
}

/// A single input vector record users provide.
#[derive(Clone, Debug)]
pub struct VectorRecord {
    pub external_id: u64,
    pub values: Vec<f32>,
    pub timestamp: Option<u64>, // If `None`, current timestamp will be used.
}

#[derive(Clone, Debug)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    pub include_vectors: bool, // If true, returns the matched vector payload in each result.
    pub k: usize,
    pub n_probe: usize,
}

impl SearchRequest {
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_n_probe(mut self, n_probe: usize) -> Self {
        self.n_probe = n_probe;
        self
    }

    pub fn with_include_vectors(mut self, include_vectors: bool) -> Self {
        self.include_vectors = include_vectors;
        self
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub external_id: u64,
    pub distance: f32,
    pub vector: Option<Vec<f32>>,
}

pub struct VectorIndexer {
    cfg: VectorIndexerConfig,
    index: IvfIndex,
}

impl VectorIndexer {
    /// Create a new, empty index wrapper. Use `build_from_*` to train and persist it.
    pub fn new(cfg: VectorIndexerConfig) -> Self {
        let index = IvfIndex::new(cfg.dimension);
        Self { cfg, index }
    }

    /// Load an existing index from disk.
    pub fn load(cfg: VectorIndexerConfig) -> Result<Self> {
        let index = load_index_from(&cfg.index_dir)?;
        Ok(Self { cfg, index })
    }

    /// Build index from in-memory records and persist to disk.
    pub fn build_from_records(mut self, records: Vec<VectorRecord>) -> Result<Self> {
        if records.is_empty() {
            return Err(Error::new(ErrorKind::InvalidInput, "no vectors provided"));
        }

        // Validate dimension.
        let dim = self.cfg.dimension as usize;
        for (i, r) in records.iter().enumerate() {
            if r.values.len() != dim {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "vector dimension mismatch at index {}: expected {}, got {}",
                        i,
                        dim,
                        r.values.len()
                    ),
                ));
            }
        }

        let vectors_data: Vec<(u64, Vec<f32>, u64)> = records
            .into_iter()
            .map(|r| (r.external_id, r.values, r.timestamp.unwrap_or(0)))
            .collect();

        let store = VectorStore::new(vectors_data);
        self.index.fit_with_paths(&store, &self.cfg.shards_dir);
        self.index.save_to(&self.cfg.index_dir)?;
        Ok(self)
    }

    /// Build index from the project's existing batched vector file format and persist it.
    pub fn build_from_vector_file(mut self, vector_file: impl AsRef<Path>) -> Result<Self> {
        let vectors = read_vectors_from_file(
            vector_file
                .as_ref()
                .to_str()
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "invalid vector_file path"))?,
        )
        .map_err(|e| Error::new(ErrorKind::Other, format!("read_vectors_from_file: {}", e)))?;

        if vectors.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "no vectors in vector_file",
            ));
        }

        // Validate dimension.
        let dim = self.cfg.dimension as usize;
        for (i, (_id, v, _ts)) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "vector dimension mismatch at index {}: expected {}, got {}",
                        i,
                        dim,
                        v.len()
                    ),
                ));
            }
        }

        let store = VectorStore::new(vectors);
        self.index.fit_with_paths(&store, &self.cfg.shards_dir);
        self.index.save_to(&self.cfg.index_dir)?;
        Ok(self)
    }

    pub async fn search(&self, req: SearchRequest) -> Result<Vec<SearchResult>> {
        let k = req.k.min(self.cfg.max_k);
        let n_probe = req.n_probe.min(self.cfg.max_n_probe);

        if req.query.len() != self.cfg.dimension as usize {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "query dimension mismatch: expected {}, got {}",
                    self.cfg.dimension,
                    req.query.len()
                ),
            ));
        }

        let raw = self
            .index
            .search_with_paths(&req.query, k, n_probe, &self.cfg.shards_dir)
            .await?;

        let results = raw
            .into_iter()
            .map(|(external_id, distance, vector)| SearchResult {
                external_id: external_id as u64,
                distance,
                vector: if req.include_vectors {
                    Some(vector)
                } else {
                    None
                },
            })
            .collect();

        Ok(results)
    }

    /// Convenience helper to create a `SearchRequest` using this indexer's configured defaults.
    pub fn search_request(&self, query: Vec<f32>) -> SearchRequest {
        SearchRequest {
            query,
            include_vectors: false,
            k: self.cfg.default_k,
            n_probe: self.cfg.default_n_probe,
        }
    }

    pub fn config(&self) -> &VectorIndexerConfig {
        &self.cfg
    }
}
