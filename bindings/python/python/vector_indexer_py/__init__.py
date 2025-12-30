"""
vector_indexer_py - Python bindings for vector_indexer with async search support.

This module provides:
- build(xb) - one-shot index build from numpy array
- VectorIndex - index handle with async search method
- suggest_nlist(n) - returns the nlist that would be computed for n vectors
"""

import asyncio
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

# Import the native Rust extension
from .vector_indexer_py import (
    build as _build,
    load as _load,
    suggest_nlist,
    PyVectorIndex as _PyVectorIndex,
)

__all__ = ["build", "load", "suggest_nlist", "VectorIndex"]


class VectorIndex:
    """
    Vector index handle with async search support.
    
    Use build() or load() to create an instance.
    """
    
    def __init__(self, native_index: _PyVectorIndex):
        self._native = native_index
    
    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in this index."""
        return self._native.dimension
    
    async def search(
        self,
        xq: NDArray[np.float32],
        k: int,
        n_probe: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Search for k nearest neighbors of query vectors.
        
        Args:
            xq: Query vectors, shape (nq, d)
            k: Number of nearest neighbors to return
            n_probe: Number of clusters to probe
            
        Returns:
            Tuple of (D, I) where:
            - D: Distances, shape (nq, k), dtype float32
            - I: Indices, shape (nq, k), dtype int64
        """
        # Ensure input is contiguous float32
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        
        # Run the blocking search in a thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        D, I = await loop.run_in_executor(
            None,  # Use default thread pool
            self._native.search_blocking,
            xq,
            k,
            n_probe,
        )
        return D, I
    
    def search_sync(
        self,
        xq: NDArray[np.float32],
        k: int,
        n_probe: int,
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Synchronous search for k nearest neighbors.
        
        Args:
            xq: Query vectors, shape (nq, d)
            k: Number of nearest neighbors to return
            n_probe: Number of clusters to probe
            
        Returns:
            Tuple of (D, I) where:
            - D: Distances, shape (nq, k), dtype float32
            - I: Indices, shape (nq, k), dtype int64
        """
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        return self._native.search_blocking(xq, k, n_probe)


def build(
    xb: NDArray[np.float32],
    work_dir: Optional[str] = None,
) -> VectorIndex:
    """
    Build an index from a numpy array of vectors.
    
    Args:
        xb: Database vectors, shape (n, d), dtype float32
        work_dir: Optional directory to store index files (default: temp dir)
        
    Returns:
        VectorIndex handle for searching
    """
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    native_index = _build(xb, work_dir)
    return VectorIndex(native_index)


def load(
    index_dir: str,
    shards_dir: str,
    dimension: int,
) -> VectorIndex:
    """
    Load an existing index from disk.
    
    Args:
        index_dir: Path to the index directory
        shards_dir: Path to the shards directory
        dimension: Vector dimension
        
    Returns:
        VectorIndex handle for searching
    """
    native_index = _load(index_dir, shards_dir, dimension)
    return VectorIndex(native_index)

