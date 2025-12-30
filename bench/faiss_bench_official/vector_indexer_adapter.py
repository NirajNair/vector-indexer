"""
Adapter to make vector_indexer look like a Faiss index for benchmarking.

This adapter:
- Exposes a synchronous .search(xq, k) method (what Faiss bench expects)
- Internally runs async vector_indexer search via a dedicated event loop thread
- Uses asyncio.run_coroutine_threadsafe() to bridge sync -> async safely

The Rust search stays truly async (via tokio_uring worker thread), while the
Python benchmark loop (which calls sync .search()) remains unchanged.
"""

import asyncio
import threading
from typing import Optional, Tuple

import numpy as np


class AsyncLoopThread:
    """
    A dedicated thread running an asyncio event loop.

    This allows calling async functions from synchronous code by submitting
    coroutines to this loop via run_coroutine_threadsafe().
    """

    _instance: Optional["AsyncLoopThread"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    @classmethod
    def get_instance(cls) -> "AsyncLoopThread":
        """Get or create the singleton async loop thread."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.start()
        return cls._instance

    def start(self):
        """Start the event loop thread."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait()  # Wait until loop is running

    def _run_loop(self):
        """Run the event loop forever in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run_coroutine(self, coro) -> any:
        """
        Submit a coroutine to the loop and wait for result.

        This is safe to call from any thread.
        """
        if self._loop is None:
            raise RuntimeError("Event loop not started")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # Blocks until complete


class VectorIndexerFaissAdapter:
    """
    Adapter that makes vector_indexer compatible with Faiss benchmark scripts.

    Provides a synchronous .search(xq, k) method that internally runs
    the async vector_indexer search.

    Usage:
        idx = vector_indexer_py.build(xb, work_dir)
        adapter = VectorIndexerFaissAdapter(idx, k=100)
        adapter.nprobe = 16
        D, I = adapter.search(xq, k)  # Synchronous call
    """

    def __init__(self, vector_index, k: int = 100):
        """
        Initialize the adapter.

        Args:
            vector_index: A VectorIndex instance from vector_indexer_py
            k: Default k value (can be overridden in search())
        """
        self._idx = vector_index
        self._k = k
        self._nprobe = 1
        self._async_loop = AsyncLoopThread.get_instance()

    @property
    def d(self) -> int:
        """Vector dimension."""
        return self._idx.dimension

    @property
    def nprobe(self) -> int:
        """Number of clusters to probe during search."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int):
        """Set number of clusters to probe."""
        self._nprobe = value

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors (synchronous interface).

        This blocks while internally running the async vector_indexer search.

        Args:
            xq: Query vectors, shape (nq, d)
            k: Number of neighbors to retrieve

        Returns:
            D: Distances, shape (nq, k)
            I: Indices, shape (nq, k)
        """
        # Ensure xq is contiguous float32
        xq = np.ascontiguousarray(xq, dtype=np.float32)

        # Create the async search coroutine
        coro = self._idx.search(xq, k, self._nprobe)

        # Run it on the dedicated async thread and wait for result
        D, I = self._async_loop.run_coroutine(coro)

        return D, I

    def __repr__(self):
        return f"VectorIndexerFaissAdapter(d={self.d}, nprobe={self.nprobe})"
