"""Main Orchestrator - Coordinates entire pipeline."""

import sys
from pathlib import Path
import time
from typing import Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing import Preprocessor
from src.core.intent_engine import IntentEngine
from src.core.router import DecisionRouter
from src.core.tiers import Tier1KB, Tier2SLM, Tier3RAG
from src.core.safety import SafetyLayer


@dataclass
class OrchestratorResponse:
    """Complete orchestrated response"""
    query: str
    response: str
    tier_used: int
    intent: str
    confidence: float
    safe: bool
    processing_time_ms: float
    metadata: Dict


class Orchestrator:
    """
    Main pipeline orchestrator.

    Flow:
    1. Preprocess
    2. Intent analysis
    3. Routing decision
    4. Tier generation
    5. Safety checks
    """

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.intent_engine = IntentEngine()
        self.router = DecisionRouter()

        self.tier1 = Tier1KB()
        self.tier2 = Tier2SLM()
        self.tier3 = Tier3RAG()

        self.safety = SafetyLayer()

    def process(
        self,
        query: str,
        session_id: str,
        context: Optional[Dict] = None
    ) -> OrchestratorResponse:
        start_time = time.time()

        preprocessed = self.preprocessor.preprocess(
            text=query,
            session_id=session_id,
            additional_context=context
        )

        if not preprocessed.is_valid:
            return self._handle_invalid_input(query, preprocessed, start_time)

        intent_result = self.intent_engine.analyze(
            preprocessed.normalized_text
        )

        router_result = self.router.route(
            preprocessed_input=preprocessed,
            intent_result=intent_result,
            session_id=session_id
        )

        if router_result.blocked:
            return self._handle_blocked(query, router_result, start_time)

        routing_decision = router_result.routing_decision
        tier_response = self._generate_from_tier(
            tier=routing_decision.selected_tier,
            query=preprocessed.normalized_text,
            intent=intent_result.intent,
            similar_queries=intent_result.similar_queries,
            requires_escalation=routing_decision.requires_escalation
        )

        safety_result = self.safety.check(
            text=tier_response.text,
            tier=tier_response.tier
        )

        final_response = safety_result.final_response
        processing_time = (time.time() - start_time) * 1000

        return OrchestratorResponse(
            query=query,
            response=final_response,
            tier_used=tier_response.tier,
            intent=intent_result.intent,
            confidence=tier_response.confidence,
            safe=safety_result.is_safe,
            processing_time_ms=processing_time,
            metadata={
                'category': intent_result.category,
                'pii_detected': len(preprocessed.detected_pii) > 0,
                'routing_reason': routing_decision.reason,
                'safety_violations': safety_result.safety_result.violations if not safety_result.is_safe else [],
                'tier_metadata': tier_response.metadata
            }
        )

    def _generate_from_tier(
        self,
        tier: int,
        query: str,
        intent: str,
        similar_queries: list,
        requires_escalation: bool
    ):
        if tier == 1:
            return self.tier1.generate(
                query=query,
                intent=intent,
                similar_queries=similar_queries
            )
        if tier == 2:
            return self.tier2.generate(
                query=query,
                intent=intent,
                similar_queries=similar_queries
            )

        return self.tier3.generate(
            query=query,
            intent=intent,
            requires_escalation=requires_escalation
        )

    def _handle_invalid_input(self, query, preprocessed, start_time):
        return OrchestratorResponse(
            query=query,
            response="I couldn't process your request. Please rephrase your question.",
            tier_used=0,
            intent="invalid",
            confidence=0.0,
            safe=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={'error': preprocessed.validation_errors[0]}
        )

    def _handle_blocked(self, query, router_result, start_time):
        return OrchestratorResponse(
            query=query,
            response="I cannot process this request. Please contact customer care.",
            tier_used=0,
            intent="blocked",
            confidence=0.0,
            safe=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={'block_reason': router_result.block_reason}
        )


__all__ = ['Orchestrator', 'OrchestratorResponse']
