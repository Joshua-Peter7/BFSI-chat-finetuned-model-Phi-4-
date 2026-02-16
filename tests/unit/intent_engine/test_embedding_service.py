"""Test embedding service"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.intent_engine.embedding_service import EmbeddingService


def test_embedding_generation():
    service = EmbeddingService()
    service.load_model()

    text = "what is my emi"
    embedding = service.embed(text)

    assert embedding is not None
    assert len(embedding) == 1024  # BGE-M3 dimension
    print(f"PASS Embedding generated: {len(embedding)} dimensions")


def test_batch_embedding():
    service = EmbeddingService()
    service.load_model()

    texts = ["what is my emi", "check loan status", "account locked"]
    embeddings = service.embed(texts)

    assert len(embeddings) == 3
    assert len(embeddings[0]) == 1024
    print(f"PASS Batch embeddings: {len(embeddings)} texts")


if __name__ == "__main__":
    test_embedding_generation()
    test_batch_embedding()
    print("\nAll embedding tests passed!")
