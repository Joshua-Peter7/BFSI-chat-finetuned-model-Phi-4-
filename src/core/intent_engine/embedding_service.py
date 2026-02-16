from typing import List, Union
import numpy as np
import yaml
import hashlib
import importlib


class EmbeddingService:
    """Generate text embeddings for similarity search."""

    def __init__(self, config_path: str = "config/intent_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['intent_engine']['embedding']
        self.model = None
        self.device = self.config.get('device', 'cpu')
        self.embedding_dim = self.config.get('embedding_dim', 1024)
        self.max_length = self.config.get('max_length', 512)
        self.normalize = self.config.get('normalize', True)
        self.force_fallback = self.config.get('force_fallback', False)
        self.using_fallback = False

        # Cache for embeddings
        self.cache = {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_size = self.config.get('cache_size', 10000)

    def load_model(self):
        """Load sentence-transformers model, or fallback if unavailable."""
        if self.model is not None:
            return

        if self.force_fallback:
            self.using_fallback = True
            self.model = "fallback"
            print("Using deterministic fallback embeddings (force_fallback=true)")
            return

        sentence_transformer_cls = None
        try:
            sentence_transformers_module = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(sentence_transformers_module, "SentenceTransformer")
        except Exception as exc:  # pragma: no cover - depends on local env
            self.using_fallback = True
            self.model = "fallback"
            print(
                "sentence_transformers unavailable "
                f"({exc.__class__.__name__}: {exc}); using deterministic fallback embeddings"
            )
            return

        print("Loading BGE-M3 model...")
        model_name = self.config.get('model_name', 'BAAI/bge-m3')

        try:
            self.model = sentence_transformer_cls(model_name)
            self.model.to(self.device)
            print(f"BGE-M3 model loaded on {self.device}")
        except Exception as exc:  # pragma: no cover - depends on local env
            self.using_fallback = True
            self.model = "fallback"
            print(
                "Error loading embedding model "
                f"({exc.__class__.__name__}: {exc}); using deterministic fallback embeddings"
            )

    def _fallback_embed_one(self, text: str) -> np.ndarray:
        """Create deterministic pseudo-embeddings when model dependency is missing."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(self.embedding_dim).astype(np.float32)
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for text(s)."""
        if self.model is None:
            self.load_model()

        # Handle single text
        if isinstance(text, str):
            # Check cache
            if self.cache_enabled and text in self.cache:
                return self.cache[text]

            # Generate embedding
            if self.using_fallback:
                embedding = self._fallback_embed_one(text)
            else:
                embedding = self.model.encode(
                    text,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False
                )

            # Cache result
            if self.cache_enabled:
                if len(self.cache) >= self.cache_size:
                    # Simple FIFO cache eviction
                    first_key = next(iter(self.cache))
                    del self.cache[first_key]
                self.cache[text] = embedding

            return embedding

        # Handle list of texts
        if self.using_fallback:
            return np.array([self._fallback_embed_one(item) for item in text])

        embeddings = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            batch_size=self.config.get('batch_size', 32)
        )
        return embeddings

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()


__all__ = ['EmbeddingService']
