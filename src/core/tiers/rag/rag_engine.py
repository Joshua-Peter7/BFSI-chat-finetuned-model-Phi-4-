"""RAG Engine using embeddings + retrieval"""

from typing import List, Dict
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.intent_engine.embedding_service import EmbeddingService
from src.data.vector_store.qdrant_client import QdrantClient
from .document_loader import DocumentLoader
from .chunker import DocumentChunker


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine

    Steps:
    1. Load documents
    2. Chunk documents
    3. Embed chunks
    4. Store in vector DB
    5. Retrieve relevant chunks for queries
    """

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['rag']

        # Components
        self.doc_loader = DocumentLoader(
            source_dir=self.config['documents']['source_dir']
        )
        self.chunker = DocumentChunker(
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap']
        )
        self.embedding_service = EmbeddingService()

        # Use separate collection for RAG
        self.vector_db = QdrantClient()
        self.collection_name = "bfsi_documents"

        self.indexed = False

    def index_documents(self):
        """Index all documents for RAG"""
        print("=" * 60)
        print("Indexing Documents for RAG")
        print("=" * 60)

        # Step 1: Load documents
        print("\nLoading documents...")
        documents = self.doc_loader.load_all()

        if len(documents) == 0:
            print("\nWARNING: No documents found!")
            print(f"Add documents to: {self.config['documents']['source_dir']}")
            print("Supported formats: PDF, DOCX, TXT, MD")
            return

        # Step 2: Chunk documents
        print("\nChunking documents...")
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: Generate embeddings
        print("\nGenerating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed(texts)

        # Step 4: Store in vector DB
        print("\nStoring in vector database...")
        metadata = [chunk.metadata for chunk in chunks]

        self.vector_db.insert(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )

        self.indexed = True
        print("\nDocument indexing complete!")
        print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks

        Returns:
            List of relevant chunks with scores
        """
        if not self.indexed:
            print("WARNING: Documents not indexed. Returning empty results.")
            return []

        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)

        # Retrieve from vector DB
        results = self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k or self.config['retrieval']['top_k'],
            score_threshold=score_threshold or self.config['retrieval']['score_threshold']
        )

        # If using fallback embeddings and no results, do a simple keyword fallback
        if (not results) and getattr(self.embedding_service, "using_fallback", False):
            return self._keyword_fallback(query, top_k or self.config['retrieval']['top_k'])

        return results

    def _keyword_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Naive keyword fallback retrieval when embeddings are unavailable."""
        query_terms = {t.lower() for t in query.split() if len(t) > 2}
        if not query_terms:
            return []

        scored = []
        # Use the in-memory store from vector DB if available
        store = getattr(self.vector_db, "_fallback_store", [])
        for item in store:
            text = item.get("payload", {}).get("text", "")
            text_lower = text.lower()
            hits = sum(1 for t in query_terms if t in text_lower)
            if hits > 0:
                score = hits / max(len(query_terms), 1)
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            results.append({
                "id": item.get("id"),
                "score": float(score),
                "text": item.get("payload", {}).get("text", ""),
                "metadata": {k: v for k, v in item.get("payload", {}).items() if k != "text"},
            })
        return results

    def generate_context(self, retrieved_chunks: List[Dict]) -> str:
        """Generate context string from retrieved chunks"""
        if not retrieved_chunks:
            return ""

        context_parts = []
        max_length = self.config['generation']['max_context_length']
        current_length = 0

        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_text = chunk['text']
            source = chunk['metadata'].get('filename', 'Unknown')

            # Format chunk with source
            formatted = f"[Source: {source}]\n{chunk_text}\n"

            # Check length limit
            if current_length + len(formatted) > max_length:
                break

            context_parts.append(formatted)
            current_length += len(formatted)

        return "\n".join(context_parts)


__all__ = ['RAGEngine']
