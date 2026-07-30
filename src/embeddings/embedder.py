import numpy as np
import hashlib
import math
from typing import List, Dict, Any, Union
from src.utils import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

class FallbackEmbedder:
    """Lightweight hash-based vector embedder fallback when sentence_transformers is missing."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            single = True
            text_list = [texts]
        else:
            single = False
            text_list = texts

        matrix = []
        for t in text_list:
            vec = np.zeros((self.embedding_dim,), dtype=np.float32)
            words = (t or " ").lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self.embedding_dim
                val = (h % 100) / 100.0
                vec[idx] += val
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            matrix.append(vec)

        res = np.array(matrix, dtype=np.float32)
        return res[0] if single else res

class BugEmbedder:
    """Generates dense vector embeddings for bug titles and descriptions using SentenceTransformers or Hash Fallback."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized BugEmbedder with model '{self.model_name}', dim={self.embedding_dim}, normalize={self.normalize}.")

    def _load_model(self):
        """Loads SentenceTransformer model with fallback."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                return SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Failed to load embedding model '{self.model_name}': {e}. Falling back to 'all-MiniLM-L6-v2'.")
                try:
                    return SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                except Exception:
                    pass

        logger.warning("sentence_transformers missing or failed to load. Operating in Fallback Hash Vectorizer mode.")
        return FallbackEmbedder(embedding_dim=384)

    def embed_text(self, text: str) -> np.ndarray:
        """Generates embedding vector for a single text string."""
        if not text:
            text = " "
        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generates embedding matrix for a batch of text strings."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        sanitized_texts = [t if t and t.strip() else " " for t in texts]
        embeddings = self.model.encode(
            sanitized_texts,
            batch_size=self.batch_size,
            show_progress_bar=len(sanitized_texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return embeddings.astype(np.float32)

    def embed_issues(self, issues: List[Dict[str, Any]]) -> np.ndarray:
        """Extracts text from a list of issue dictionaries and computes embeddings."""
        texts = []
        for issue in issues:
            text = (
                issue.get("combined_clean")
                or issue.get("combined_text")
                or f"{issue.get('title', '')} - {issue.get('description', '')}"
            )
            texts.append(text)

        logger.info(f"Generating embeddings for {len(texts)} issues...")
        return self.embed_batch(texts)

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encodes query string for FAISS vector search."""
        return self.embed_text(query_text)

    @staticmethod
    def get_embedding_stats(embeddings: np.ndarray) -> Dict[str, Any]:
        """Returns statistical diagnostics of the embedding matrix."""
        if embeddings.size == 0:
            return {"shape": (0, 0)}

        norms = np.linalg.norm(embeddings, axis=1)
        return {
            "shape": embeddings.shape,
            "mean": float(np.mean(embeddings)),
            "std": float(np.std(embeddings)),
            "min": float(np.min(embeddings)),
            "max": float(np.max(embeddings)),
            "norm_mean": float(np.mean(norms))
        }
