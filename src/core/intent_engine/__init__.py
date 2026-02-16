"""Intent & Similarity Engine"""

from .embedding_service import EmbeddingService
from .intent_classifier import IntentClassifier
from .similarity_search import SimilaritySearch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class IntentResult:
    """Intent classification result"""
    intent: str
    confidence: float
    category: str
    similar_queries: List[Dict]


class IntentEngine:
    """
    Main Intent & Similarity Engine
    
    Combines:
    1. Intent classification (keyword-based)
    2. Similarity search (vector-based)
    """
    
    def __init__(self, config_path: str = "config/intent_config.yaml"):
        self.intent_classifier = IntentClassifier(config_path)
        self.similarity_search = SimilaritySearch(config_path)
    
    def analyze(self, query: str, top_k: int = 5) -> IntentResult:
        """
        Analyze query for intent and find similar examples
        
        Args:
            query: User query text
            top_k: Number of similar queries to return
            
        Returns:
            IntentResult with intent and similar queries
        """
        # Classify intent
        intent, confidence = self.intent_classifier.classify(query)
        category = self.intent_classifier.get_category(intent)
        
        # Find similar queries (no intent filter so your KB data is always searchable)
        similar_queries = self.similarity_search.search(
            query=query,
            top_k=top_k,
            intent_filter=None
        )
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            category=category,
            similar_queries=similar_queries
        )
    
    def index_dataset(self, dataset: List[Dict]):
        """Index BFSI dataset"""
        return self.similarity_search.index_dataset(dataset)


__all__ = ['IntentEngine', 'IntentResult']