"""Tier 3: RAG + Escalation"""

from .base_tier import BaseTier, TierResponse
from typing import Dict, List
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.tiers.rag.rag_engine import RAGEngine


class Tier3RAG(BaseTier):
    """
    Tier 3: RAG + Escalation

    Two modes:
    1. RAG Mode - Retrieve documents and provide context
    2. Escalation Mode - Pass to human agents
    """

    def __init__(self, config_path: str = "config/tiers_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        super().__init__(config['tiers']['tier3'])
        self.tier_number = 3
        self.tier_name = self.config['name']

        # Escalation settings
        self.escalation_message = self.config.get('escalation', {}).get(
            'escalation_message',
            "I'll connect you with a specialist."
        )

        # RAG settings
        self.rag_enabled = self.config.get('rag', {}).get('enabled', True)

        # Initialize RAG engine if enabled
        self.rag_engine = None
        if self.rag_enabled:
            try:
                self.rag_engine = RAGEngine()
            except Exception as e:
                print(f"WARNING: RAG initialization failed: {e}")
                self.rag_enabled = False

    def generate(
        self,
        query: str,
        intent: str,
        requires_escalation: bool = False,
        use_rag: bool = True
    ) -> TierResponse:
        """
        Generate response using RAG or escalation

        Args:
            query: User query
            intent: Detected intent
            requires_escalation: Force human escalation
            use_rag: Try RAG first
        """

        # Mode 1: Explicit escalation
        if requires_escalation:
            return self._escalate(query, intent, "explicit_escalation")

        # Mode 2: Try RAG
        if use_rag and self.rag_enabled and self.rag_engine:
            try:
                if not self.rag_engine.indexed:
                    self.rag_engine.index_documents()
                rag_response = self._generate_with_rag(query, intent)
                if rag_response.confidence > 0.5:
                    return rag_response
            except Exception as e:
                print(f"WARNING: RAG generation failed: {e}")

        # Mode 3: Fallback to escalation
        return self._escalate(query, intent, "low_confidence")

    def _generate_with_rag(self, query: str, intent: str) -> TierResponse:
        """Generate response using RAG"""

        # Retrieve relevant chunks
        retrieved_chunks = self.rag_engine.retrieve(query)

        if not retrieved_chunks:
            # No relevant documents found
            return TierResponse(
                tier=3,
                text="",
                confidence=0.0,
                generation_time_ms=1.0,
                source="tier3_rag",
                metadata={'reason': 'no_documents_found'}
            )

        # Generate context
        context = self.rag_engine.generate_context(retrieved_chunks)

        # Create response with context
        response_text = (
            "Based on our policy documents:\n\n"
            f"{context}\n\n"
            "For more details, please visit our website or contact customer care."
        )

        # Calculate confidence based on retrieval scores
        avg_score = sum(c['score'] for c in retrieved_chunks) / len(retrieved_chunks)

        return TierResponse(
            tier=3,
            text=response_text,
            confidence=avg_score,
            generation_time_ms=10.0,
            source="tier3_rag",
            metadata={
                'num_chunks': len(retrieved_chunks),
                'sources': [c['metadata'].get('filename') for c in retrieved_chunks]
            }
        )

    def _escalate(self, query: str, intent: str, reason: str) -> TierResponse:
        """Escalate to human agent"""

        return TierResponse(
            tier=3,
            text=self.escalation_message,
            confidence=1.0,
            generation_time_ms=1.0,
            source="tier3_escalation",
            metadata={
                'intent': intent,
                'requires_human': True,
                'escalation_reason': reason
            }
        )


__all__ = ['Tier3RAG']
