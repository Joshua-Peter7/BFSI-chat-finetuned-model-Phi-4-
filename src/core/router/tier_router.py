"""Tier routing decision logic"""

from typing import Dict, Optional
from dataclasses import dataclass
import yaml


@dataclass
class RoutingDecision:
    """Routing decision result"""
    selected_tier: int  # 1, 2, or 3
    confidence: float
    reason: str
    fallback_tier: Optional[int] = None
    requires_escalation: bool = False


class TierRouter:
    """
    Route queries to appropriate tier
    
    Routing Logic:
    - Tier 1: High confidence (0.85+) + exact match in KB
    - Tier 2: Medium confidence (0.60-0.85) 
    - Tier 3: Low confidence (<0.60) or escalation intent
    """
    
    def __init__(self, config_path: str = "config/routing_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['routing']
        self.strategy = self.config.get('strategy', 'confidence_based')
        
        # Thresholds
        self.tier1_threshold = self.config['thresholds']['tier1']['min_confidence']
        self.tier1_similarity = self.config['thresholds']['tier1']['min_similarity']
        self.tier2_min = self.config['thresholds']['tier2']['min_confidence']
        self.tier2_max = self.config['thresholds']['tier2']['max_confidence']
        self.escalation_threshold = self.config['thresholds']['tier3']['escalation_threshold']
        
        # Intent rules
        self.tier1_intents = set(self.config['intent_rules']['tier1_intents'])
        self.tier2_intents = set(self.config['intent_rules']['tier2_intents'])
        self.tier3_intents = set(self.config['intent_rules']['tier3_intents'])
    
    def route(
        self,
        intent_result,
        similarity_results: list
    ) -> RoutingDecision:
        """
        Determine optimal tier for query
        
        Args:
            intent_result: IntentResult from intent engine
            similarity_results: List of similar queries from vector search
            
        Returns:
            RoutingDecision with tier selection
        """
        intent = intent_result.intent
        confidence = intent_result.confidence
        
        # Rule 1: Check for explicit escalation intents
        if intent in self.tier3_intents:
            return RoutingDecision(
                selected_tier=3,
                confidence=1.0,
                reason="Escalation intent detected",
                requires_escalation=True
            )
        
        # Rule 2: Check for high-confidence exact match (Tier 1)
        if self._has_exact_match(similarity_results):
            if confidence >= self.tier1_threshold or intent in self.tier1_intents:
                return RoutingDecision(
                    selected_tier=1,
                    confidence=confidence,
                    reason="High confidence KB match",
                    fallback_tier=2
                )
        
        # Rule 3: Medium confidence (Tier 2)
        if self.tier2_min <= confidence < self.tier2_max:
            return RoutingDecision(
                selected_tier=2,
                confidence=confidence,
                reason="Medium confidence, using fine-tuned SLM",
                fallback_tier=3
            )
        
        # Rule 4: High confidence but no exact match (Tier 2)
        if confidence >= self.tier1_threshold:
            return RoutingDecision(
                selected_tier=2,
                confidence=confidence,
                reason="High confidence but no exact KB match",
                fallback_tier=3
            )
        
        # Rule 5: Low confidence but not unknown — use Tier 3 RAG
        if confidence >= self.escalation_threshold:
            return RoutingDecision(
                selected_tier=3,
                confidence=confidence,
                reason="Low confidence, using RAG",
                fallback_tier=None
            )
        
        # Rule 6: Unknown intent — let fine-tuned SLM (Tier 2) try first instead of escalating
        if intent == "unknown":
            return RoutingDecision(
                selected_tier=2,
                confidence=confidence,
                reason="Unknown intent, using fine-tuned SLM",
                fallback_tier=3
            )
        
        # Rule 7: Very low confidence, known intent (Escalate)
        return RoutingDecision(
            selected_tier=3,
            confidence=confidence,
            reason="Very low confidence, human escalation required",
            requires_escalation=True
        )
    
    def _has_exact_match(self, similarity_results: list) -> bool:
        """
        Check if top result is an exact match
        
        Score >= 0.90 indicates near-exact match
        """
        if not similarity_results:
            return False
        
        top_result = similarity_results[0]
        return top_result['score'] >= self.tier1_similarity


__all__ = ['TierRouter', 'RoutingDecision']