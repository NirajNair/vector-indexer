pub mod api;
pub use api::{SearchRequest, SearchResult, VectorIndexer, VectorIndexerConfig, VectorRecord};

// Internal implementation modules (not part of the public API).
#[cfg(not(feature = "internal_tests"))]
mod ivf_index;
#[cfg(feature = "internal_tests")]
pub mod ivf_index;

#[cfg(not(feature = "internal_tests"))]
mod kmeans;
#[cfg(feature = "internal_tests")]
pub mod kmeans;

#[cfg(not(feature = "internal_tests"))]
mod shards;
#[cfg(feature = "internal_tests")]
pub mod shards;

#[cfg(not(feature = "internal_tests"))]
mod utils;
#[cfg(feature = "internal_tests")]
pub mod utils;

#[cfg(not(feature = "internal_tests"))]
mod vector_store;
#[cfg(feature = "internal_tests")]
pub mod vector_store;
