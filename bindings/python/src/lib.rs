//! Python bindings for vector_indexer with async search support.
//!
//! This module exposes:
//! - `build(xb)` - one-shot index build from numpy array
//! - `search(xq, k, n_probe)` - async search returning (D, I) arrays
//! - `suggest_nlist(n)` - returns the nlist that would be computed for n vectors

use crossbeam_channel::{bounded, Receiver, Sender};
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use vector_indexer::{SearchRequest, VectorIndexer, VectorIndexerConfig, VectorRecord};

/// Message sent to the tokio_uring worker thread
enum WorkerRequest {
    Search {
        id: u64,
        queries: Vec<Vec<f32>>,
        k: usize,
        n_probe: usize,
    },
    Shutdown,
}

/// Response from the tokio_uring worker thread
struct WorkerResponse {
    id: u64,
    result: Result<Vec<Vec<(u64, f32)>>, String>,
}

/// Global worker thread handle and channels
struct WorkerHandle {
    request_tx: Sender<WorkerRequest>,
    response_rx: Receiver<WorkerResponse>,
    _thread: thread::JoinHandle<()>,
}

static WORKER: OnceCell<WorkerHandle> = OnceCell::new();
static REQUEST_ID: AtomicU64 = AtomicU64::new(0);

/// The Python-visible index handle
#[pyclass]
pub struct PyVectorIndex {
    indexer: Arc<VectorIndexer>,
    dimension: usize,
}

/// Start the tokio_uring worker thread if not already running
fn ensure_worker(indexer: Arc<VectorIndexer>) -> &'static WorkerHandle {
    WORKER.get_or_init(|| {
        let (req_tx, req_rx) = bounded::<WorkerRequest>(100);
        let (resp_tx, resp_rx) = bounded::<WorkerResponse>(100);

        let indexer_clone = indexer.clone();
        let handle = thread::spawn(move || {
            tokio_uring::start(async move {
                loop {
                    match req_rx.recv() {
                        Ok(WorkerRequest::Search {
                            id,
                            queries,
                            k,
                            n_probe,
                        }) => {
                            let indexer = indexer_clone.clone();
                            let mut all_results = Vec::with_capacity(queries.len());

                            for query in queries {
                                let req = SearchRequest {
                                    query,
                                    include_vectors: false,
                                    k,
                                    n_probe,
                                };
                                match indexer.search(req).await {
                                    Ok(results) => {
                                        let pairs: Vec<(u64, f32)> = results
                                            .into_iter()
                                            .map(|r| (r.external_id, r.distance))
                                            .collect();
                                        all_results.push(pairs);
                                    }
                                    Err(e) => {
                                        let _ = resp_tx.send(WorkerResponse {
                                            id,
                                            result: Err(e.to_string()),
                                        });
                                        continue;
                                    }
                                }
                            }

                            let _ = resp_tx.send(WorkerResponse {
                                id,
                                result: Ok(all_results),
                            });
                        }
                        Ok(WorkerRequest::Shutdown) | Err(_) => {
                            break;
                        }
                    }
                }
            });
        });

        WorkerHandle {
            request_tx: req_tx,
            response_rx: resp_rx,
            _thread: handle,
        }
    })
}

#[pymethods]
impl PyVectorIndex {
    /// Blocking search - used internally by the async wrapper
    fn search_blocking(
        &self,
        py: Python<'_>,
        xq: PyReadonlyArray2<'_, f32>,
        k: usize,
        n_probe: usize,
    ) -> PyResult<Py<PyTuple>> {
        let nq = xq.shape()[0];
        let dim = xq.shape()[1];

        if dim != self.dimension {
            return Err(PyRuntimeError::new_err(format!(
                "Query dimension {} doesn't match index dimension {}",
                dim, self.dimension
            )));
        }

        // Convert numpy array to Vec<Vec<f32>> (copy data before releasing GIL)
        let data = xq.as_slice()
            .map_err(|_| PyRuntimeError::new_err("Query array must be contiguous"))?;
        
        let queries: Vec<Vec<f32>> = (0..nq)
            .map(|i| data[i * dim..(i + 1) * dim].to_vec())
            .collect();

        let worker = ensure_worker(self.indexer.clone());
        let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

        // Send request to worker
        worker
            .request_tx
            .send(WorkerRequest::Search {
                id,
                queries,
                k,
                n_probe,
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to send request: {}", e)))?;

        // Wait for response (this blocks, but we release the GIL)
        let response = py.allow_threads(|| {
            // Find our response (in case of out-of-order responses)
            loop {
                match worker.response_rx.recv() {
                    Ok(resp) if resp.id == id => return Ok(resp),
                    Ok(_) => continue, // Not our response, keep waiting
                    Err(e) => return Err(format!("Failed to receive response: {}", e)),
                }
            }
        });

        let response = response.map_err(PyRuntimeError::new_err)?;
        let results = response.result.map_err(PyRuntimeError::new_err)?;

        // Build output arrays: D (distances) and I (indices)
        // Shape: (nq, k), fill with inf/-1 for missing results
        let mut distances = vec![f32::INFINITY; nq * k];
        let mut indices = vec![-1i64; nq * k];

        for (q_idx, query_results) in results.into_iter().enumerate() {
            for (r_idx, (ext_id, dist)) in query_results.into_iter().take(k).enumerate() {
                distances[q_idx * k + r_idx] = dist;
                indices[q_idx * k + r_idx] = ext_id as i64;
            }
        }

        // Create numpy arrays as 1D first, then reshape
        let d_array_1d = distances.to_pyarray_bound(py);
        let d_array = d_array_1d
            .reshape([nq, k])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape D: {}", e)))?
            .unbind();
        
        let i_array_1d = indices.to_pyarray_bound(py);
        let i_array = i_array_1d
            .reshape([nq, k])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape I: {}", e)))?
            .unbind();

        Ok(PyTuple::new_bound(py, [d_array.into_any(), i_array.into_any()]).unbind())
    }

    /// Get the dimension of the index
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Build an index from a numpy array of vectors.
/// 
/// Args:
///     xb: numpy array of shape (n, d) containing the database vectors
///     work_dir: optional directory to store index files (default: temp dir)
/// 
/// Returns:
///     PyVectorIndex handle for searching
#[pyfunction]
#[pyo3(signature = (xb, work_dir=None))]
fn build(py: Python<'_>, xb: PyReadonlyArray2<'_, f32>, work_dir: Option<String>) -> PyResult<PyVectorIndex> {
    let n = xb.shape()[0];
    let d = xb.shape()[1];

    if n == 0 {
        return Err(PyRuntimeError::new_err("Cannot build index from empty array"));
    }

    // Determine work directory
    let work_path = match work_dir {
        Some(p) => PathBuf::from(p),
        None => std::env::temp_dir().join("vector_indexer_bench"),
    };

    let index_dir = work_path.join("index");
    let shards_dir = work_path.join("shards");

    // Create directories
    std::fs::create_dir_all(&index_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index dir: {}", e)))?;
    std::fs::create_dir_all(&shards_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create shards dir: {}", e)))?;

    println!(
        "Building index: {} vectors, {} dimensions, work_dir: {}",
        n,
        d,
        work_path.display()
    );

    // Convert numpy array to VectorRecords (copy data before releasing GIL)
    let data = xb.as_slice()
        .map_err(|_| PyRuntimeError::new_err("Array must be contiguous"))?;
    
    let records: Vec<VectorRecord> = (0..n)
        .map(|i| VectorRecord {
            external_id: i as u64,
            values: data[i * d..(i + 1) * d].to_vec(),
            timestamp: None,
        })
        .collect();

    // Build the index (release GIL during heavy computation)
    let config = VectorIndexerConfig::new(d as u32)
        .with_index_dir(&index_dir)
        .with_shards_dir(&shards_dir);

    let indexer = VectorIndexer::new(config);
    let indexer = py.allow_threads(|| {
        indexer.build_from_records(records)
    }).map_err(|e| PyRuntimeError::new_err(format!("Failed to build index: {}", e)))?;

    println!("Index built successfully");

    Ok(PyVectorIndex {
        indexer: Arc::new(indexer),
        dimension: d,
    })
}

/// Load an existing index from disk.
/// 
/// Args:
///     index_dir: path to the index directory
///     shards_dir: path to the shards directory
///     dimension: vector dimension
/// 
/// Returns:
///     PyVectorIndex handle for searching
#[pyfunction]
fn load(index_dir: String, shards_dir: String, dimension: u32) -> PyResult<PyVectorIndex> {
    let config = VectorIndexerConfig::new(dimension)
        .with_index_dir(&index_dir)
        .with_shards_dir(&shards_dir);

    let indexer = VectorIndexer::load(config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load index: {}", e)))?;

    Ok(PyVectorIndex {
        indexer: Arc::new(indexer),
        dimension: dimension as usize,
    })
}

/// Returns the number of clusters (nlist) that would be computed for n vectors.
/// This mirrors the logic in vector_indexer::utils::calculate_num_clusters.
#[pyfunction]
fn suggest_nlist(n: usize) -> usize {
    match n {
        n if n < 10_000 => (n as f64).sqrt() as usize,
        n if n < 100_000 => 2 * (n as f64).sqrt().ceil() as usize,
        _ => 4 * (n as f64).sqrt().ceil() as usize,
    }
}

/// Python module definition
#[pymodule]
fn vector_indexer_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(suggest_nlist, m)?)?;
    m.add_class::<PyVectorIndex>()?;
    Ok(())
}

