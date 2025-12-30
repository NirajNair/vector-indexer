"""
Official Faiss-style benchmark adapted for vector_indexer.

This module contains:
- bench_all_ivf.py: Main benchmark script with official eval_setting() methodology
- vector_indexer_adapter.py: Adapter to make vector_indexer look like a Faiss index
"""

from .vector_indexer_adapter import VectorIndexerFaissAdapter

__all__ = ["VectorIndexerFaissAdapter"]

