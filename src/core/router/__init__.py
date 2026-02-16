"""Decision Router Module"""

from .guardrails import Guardrails, GuardrailResult
from .tier_router import TierRouter, RoutingDecision
from dataclasses import dataclass
from typing import Optional


@dataclass
class RouterResult:
    """Complete routing result"""
    routing_decision: RoutingDecision
    guardrail_result: GuardrailResult
    blocked: bool = False
    block_reason: Optional[str] = None


class DecisionRouter:
    """
    Main decision router
    
    Pipeline:
    1. Guardrails check
    2. Tier routing decision
    3. Fallback handling
    """
    
    def __init__(
        self,
        routing_config_path: str = "config/routing_config.yaml"
    ):
        self.guardrails = Guardrails(routing_config_path)
        self.tier_router = TierRouter(routing_config_path)
    
    def route(
        self,
        preprocessed_input,
        intent_result,
        session_id: str
    ) -> RouterResult:
        """
        Complete routing decision
        
        Args:
            preprocessed_input: PreprocessedInput from preprocessing
            intent_result: IntentResult from intent engine
            session_id: Session identifier
            
        Returns:
            RouterResult with routing decision
        """
        # Step 1: Guardrails check
        guardrail_result = self.guardrails.check(
            preprocessed_input=preprocessed_input,
            session_id=session_id
        )
        
        if not guardrail_result.passed:
            # Blocked by guardrails
            return RouterResult(
                routing_decision=RoutingDecision(
                    selected_tier=3,
                    confidence=0.0,
                    reason=f"Blocked: {guardrail_result.blocked_reason}",
                    requires_escalation=True
                ),
                guardrail_result=guardrail_result,
                blocked=True,
                block_reason=guardrail_result.blocked_reason
            )
        
        # Step 2: Tier routing
        routing_decision = self.tier_router.route(
            intent_result=intent_result,
            similarity_results=intent_result.similar_queries
        )
        
        return RouterResult(
            routing_decision=routing_decision,
            guardrail_result=guardrail_result,
            blocked=False
        )


__all__ = ['DecisionRouter', 'RouterResult', 'RoutingDecision', 'GuardrailResult']