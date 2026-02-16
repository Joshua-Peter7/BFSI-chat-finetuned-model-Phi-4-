from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import yaml
import uuid


class QdrantClient:
    """Qdrant vector database wrapper"""

    def __init__(self, config_path: str = "config/intent_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['intent_engine']['qdrant']
        self.collection_name = self.config.get('collection_name', 'bfsi_intents')
        self.vector_size = self.config.get('vector_size', 1024)
        self.use_memory = self.config.get('use_memory', True)
        self._fallback_store = []

        # Initialize client
        if self.use_memory:
            # In-memory mode for development
            self.client = QdrantClientBase(":memory:")
            print("Qdrant initialized in-memory mode")
        else:
            # Server mode for production
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 6333)
            self.client = QdrantClientBase(host=host, port=port)
            print(f"Qdrant connected to {host}:{port}")

        self._create_collection()

    def _create_collection(self):
        """Create collection if not exists"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection exists: {self.collection_name}")

    def insert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None
    ) -> List[str]:
        """Insert vectors into collection"""
        if metadata is None:
            metadata = [{}] * len(texts)

        points = []
        ids = []

        for text, embedding, meta in zip(texts, embeddings, metadata):
            point_id = str(uuid.uuid4())
            ids.append(point_id)

            payload = {
                'text': text,
                **meta
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    payload=payload
                )
            )
            self._fallback_store.append(
                {
                    "id": point_id,
                    "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    "payload": payload,
                }
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=filter_dict
            )
        elif hasattr(self.client, "search_points"):
            results = self.client.search_points(
                collection_name=self.collection_name,
                vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter=filter_dict
            )
        else:
            # Fallback to in-memory search if client lacks search API.
            results = self._fallback_search(query_vector, top_k, score_threshold, filter_dict)

        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'score': result.score,
                'text': result.payload.get('text', ''),
                'metadata': {k: v for k, v in result.payload.items() if k != 'text'}
            })

        return formatted_results

    def _fallback_search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: float,
        filter_dict: Optional[Dict]
    ) -> List:
        import numpy as np

        if not self._fallback_store:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scored = []
        for item in self._fallback_store:
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if item["payload"].get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            v = np.array(item["embedding"], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            score = float(np.dot(q, v / v_norm))
            if score < score_threshold:
                continue
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            results.append(
                type("ScoredPoint", (), {
                    "id": item["id"],
                    "score": score,
                    "payload": item["payload"],
                })
            )
        return results

    def delete_collection(self):
        """Delete collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def get_collection_info(self) -> Dict:
        """Get collection information"""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            'name': self.collection_name,
            'vectors_count': info.vectors_count,
            'points_count': info.points_count
        }


__all__ = ['QdrantClient']
