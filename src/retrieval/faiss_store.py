import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.utils import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

class FallbackNumpyIndex:
    """NumPy-based cosine/L2 vector index fallback when FAISS package is missing."""
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self.vectors: List[np.ndarray] = []
        self.ntotal = 0

    def add(self, embeddings: np.ndarray):
        for vec in embeddings:
            self.vectors.append(vec.copy())
        self.ntotal = len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.vectors:
            return np.array([[]]), np.array([[]])

        matrix = np.array(self.vectors) # (N, dim)
        q = query[0] # (dim,)

        if self.metric in ["cosine", "inner_product", "ip"]:
            # Cosine similarity scores
            scores = np.dot(matrix, q)
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_distances = scores[top_k_indices]
        else:
            # L2 distances
            diffs = matrix - q
            dists = np.linalg.norm(diffs, axis=1)
            top_k_indices = np.argsort(dists)[:k]
            top_k_distances = dists[top_k_indices]

        return np.array([top_k_distances]), np.array([top_k_indices])

class FaissBugStore:
    """FAISS vector store for high-performance similarity search over bug issue embeddings."""

    def __init__(
        self,
        embedding_dim: int = 384,
        metric: str = "cosine",
        index_type: str = "Flat"
    ):
        self.embedding_dim = embedding_dim
        self.metric = metric.lower()
        self.index_type = index_type
        self.index = self._create_index()
        self.metadatas: List[Dict[str, Any]] = []
        logger.info(f"Initialized FaissBugStore (dim={embedding_dim}, metric={self.metric}, index={self.index_type}).")

    def _create_index(self):
        """Instantiates appropriate FAISS index or NumPy fallback."""
        if FAISS_AVAILABLE:
            if self.metric in ["cosine", "inner_product", "ip"]:
                if self.index_type == "HNSW":
                    return faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
                else:
                    return faiss.IndexFlatIP(self.embedding_dim)
            else:
                return faiss.IndexFlatL2(self.embedding_dim)
        else:
            logger.warning("faiss package not installed. Using FallbackNumpyIndex.")
            return FallbackNumpyIndex(self.embedding_dim, metric=self.metric)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Appends embeddings and associated metadata into the index."""
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        embeddings = embeddings.astype(np.float32)
        if self.metric in ["cosine", "inner_product", "ip"] and FAISS_AVAILABLE:
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadatas.extend(metadatas)
        logger.info(f"Added {embeddings.shape[0]} vectors to vector index. Total count: {self.index.ntotal}.")

    def build_from_scratch(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Resets the store and builds a fresh vector index."""
        self.index = self._create_index()
        self.metadatas = []
        self.add(embeddings, metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Searches index for nearest K vectors to query_embedding."""
        if self.index.ntotal == 0:
            logger.warning("Vector search called on empty index.")
            return np.array([]), np.array([]), []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = query_embedding.astype(np.float32)
        if self.metric in ["cosine", "inner_product", "ip"] and FAISS_AVAILABLE:
            faiss.normalize_L2(query_embedding)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        matched_metadatas = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadatas):
                matched_metadatas.append(self.metadatas[idx])
            else:
                matched_metadatas.append({})

        return distances[0], indices[0], matched_metadatas

    def save(self, index_path: str, metadata_path: str):
        """Persists index to disk alongside JSON/PKL metadata."""
        idx_file = Path(index_path)
        meta_file = Path(metadata_path)

        idx_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        if FAISS_AVAILABLE and not isinstance(self.index, FallbackNumpyIndex):
            faiss.write_index(self.index, str(idx_file))
            logger.info(f"Persisted FAISS index to {idx_file}")
        else:
            with open(idx_file.with_suffix(".pkl"), "wb") as f:
                pickle.dump(getattr(self.index, "vectors", []), f)

        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self.metadatas, f, indent=2, ensure_ascii=False)
            logger.info(f"Persisted metadata to {meta_file}")
        except Exception as e:
            pkl_file = meta_file.with_suffix(".pkl")
            with open(pkl_file, "wb") as f:
                pickle.dump(self.metadatas, f)

    def load(self, index_path: str, metadata_path: str):
        """Loads index and metadata from disk files."""
        idx_file = Path(index_path)
        meta_file = Path(metadata_path)

        if FAISS_AVAILABLE and idx_file.exists():
            self.index = faiss.read_index(str(idx_file))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {idx_file}")
        else:
            pkl_idx = idx_file.with_suffix(".pkl")
            if pkl_idx.exists():
                with open(pkl_idx, "rb") as f:
                    vecs = pickle.load(f)
                self.index = FallbackNumpyIndex(self.embedding_dim, metric=self.metric)
                self.index.add(np.array(vecs))

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)
        else:
            pkl_file = meta_file.with_suffix(".pkl")
            if pkl_file.exists():
                with open(pkl_file, "rb") as f:
                    self.metadatas = pickle.load(f)
            else:
                self.metadatas = []

    def get_stats(self) -> Dict[str, Any]:
        """Returns index metrics and total item counts."""
        return {
            "ntotal": int(self.index.ntotal) if self.index else 0,
            "dimension": self.embedding_dim,
            "metric": self.metric,
            "index_type": self.index_type,
            "metadata_count": len(self.metadatas)
        }
