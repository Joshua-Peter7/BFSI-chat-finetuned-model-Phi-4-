"""Similarity Search Engine"""

from typing import List, Dict, Optional
from .embedding_service import EmbeddingService
from src.data.vector_store.qdrant_client import QdrantClient
import yaml


class SimilaritySearch:
    """Vector similarity search using BGE-M3 + Qdrant"""

    def __init__(self, config_path: str = "config/intent_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['intent_engine']['similarity_search']
        self.top_k = self.config.get('top_k', 5)
        self.score_threshold = self.config.get('score_threshold', 0.70)

        # Initialize components
        self.embedding_service = EmbeddingService(config_path)
        self.vector_db = QdrantClient(config_path)

    def index_dataset(self, dataset: List[Dict]):
        """
        Index BFSI dataset into vector database

        Args:
            dataset: List of dicts with 'input', 'output', 'instruction'
        """
        print(f"Indexing {len(dataset)} examples...")

        # Extract texts and metadata
        texts = []
        metadata = []

        for example in dataset:
            text = example.get('input', '')
            texts.append(text)

            metadata.append({
                'instruction': example.get('instruction', ''),
                'output': example.get('output', ''),
                'intent': example.get('intent', 'unknown')
            })

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_service.embed(texts)

        # Insert into vector DB
        print("Inserting into Qdrant...")
        ids = self.vector_db.insert(texts, embeddings, metadata)

        print(f"Indexed {len(ids)} examples")

        return ids

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        intent_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar queries

        Args:
            query: User query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            intent_filter: Filter by specific intent

        Returns:
            List of similar examples with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)

        # Build filter
        filter_dict = None
        if intent_filter:
            filter_dict = {'intent': intent_filter}

        # Search
        results = self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k or self.top_k,
            score_threshold=score_threshold or self.score_threshold,
            filter_dict=filter_dict
        )

        return results

    def get_database_info(self) -> Dict:
        """Get vector database information"""
        return self.vector_db.get_collection_info()


__all__ = ['SimilaritySearch']
