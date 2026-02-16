"""Tier 1: Dataset Knowledge Base"""

from .base_tier import BaseTier, TierResponse
from typing import Dict, List
import yaml


class Tier1KB(BaseTier):
    """
    Tier 1: Direct lookup from dataset
    
    - Fastest response
    - Zero hallucination
    - Pre-approved answers only
    """
    
    def __init__(self, config_path: str = "config/tiers_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        super().__init__(config['tiers']['tier1'])
        self.tier_number = 1
        self.tier_name = self.config['name']
        self.confidence_boost = self.config.get('confidence_boost', 0.05)
    
    def generate(
        self,
        query: str,
        intent: str,
        similar_queries: List[Dict]
    ) -> TierResponse:
        """
        Generate response from KB
        
        Strategy:
        1. Use top similar query result
        2. Return pre-approved response
        3. Boost confidence slightly
        """
        
        if not similar_queries or len(similar_queries) == 0:
            return TierResponse(
                tier=1,
                text="",
                confidence=0.0,
                generation_time_ms=0.0,
                source="tier1_kb",
                metadata={'reason': 'no_match_found'}
            )
        
        # Get top result
        top_result = similar_queries[0]
        
        # Check if confidence is high enough (lowered so your KB data gets used)
        if top_result['score'] < 0.78:
            return TierResponse(
                tier=1,
                text="",
                confidence=top_result['score'],
                generation_time_ms=1.0,
                source="tier1_kb",
                metadata={'reason': 'low_confidence'}
            )
        
        # Return pre-approved response
        response_text = top_result['metadata'].get('output', '')
        
        if not response_text:
            response_text = top_result.get('text', '')
        
        # Boost confidence for KB responses
        confidence = min(top_result['score'] + self.confidence_boost, 1.0)
        
        return TierResponse(
            tier=1,
            text=response_text,
            confidence=confidence,
            generation_time_ms=1.0,  # KB lookup is instant
            source="tier1_kb",
            metadata={
                'matched_query': top_result['text'],
                'similarity_score': top_result['score']
            }
        )


__all__ = ['Tier1KB']