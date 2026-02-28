"""
Embedding service: converts text queries into 768-D dense vectors using MPNet.
Singleton pattern — model loaded once, reused across requests.
"""
import time
import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading embedding model: {config.embedding.model_name}")
        start = time.time()
        self.model = SentenceTransformer(
            config.embedding.model_name,
            device=config.embedding.device,
        )
        self.dimension = config.embedding.dimension
        elapsed = time.time() - start
        logger.info(f"Embedding model loaded in {elapsed:.2f}s (dim={self.dimension})")
        self._initialized = True

    def encode(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into dense vectors.
        Returns shape (dim,) for single string, (n, dim) for list.
        Always L2-normalized for cosine similarity via dot product.
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        embeddings = self.model.encode(
            text,
            batch_size=config.embedding.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        if single:
            return embeddings[0]
        return embeddings

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors (= dot product)."""
        return float(np.dot(a, b))
