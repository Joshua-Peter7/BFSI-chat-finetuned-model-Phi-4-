"""Test tier router"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.router.tier_router import TierRouter
from src.core.intent_engine import IntentResult


def test_tier1_high_confidence():
    """Test Tier 1 routing for high confidence"""
    router = TierRouter()
    
    intent_result = IntentResult(
        intent="emi_details",
        confidence=0.90,
        category="emi",
        similar_queries=[
            {'score': 0.95, 'text': 'what is my emi'},
            {'score': 0.85, 'text': 'check emi amount'}
        ]
    )
    
    decision = router.route(intent_result, intent_result.similar_queries)
    
    assert decision.selected_tier == 1
    assert decision.reason == "High confidence KB match"
    print(f"✅ Tier 1: {decision.reason}")


def test_tier2_medium_confidence():
    """Test Tier 2 routing for medium confidence"""
    router = TierRouter()
    
    intent_result = IntentResult(
        intent="payment_failure",
        confidence=0.70,
        category="payment",
        similar_queries=[
            {'score': 0.75, 'text': 'payment failed'},
        ]
    )
    
    decision = router.route(intent_result, intent_result.similar_queries)
    
    assert decision.selected_tier == 2
    print(f"✅ Tier 2: {decision.reason}")


def test_tier3_escalation():
    """Test Tier 3 for escalation intent"""
    router = TierRouter()
    
    intent_result = IntentResult(
        intent="complaint",
        confidence=0.80,
        category="escalation",
        similar_queries=[]
    )
    
    decision = router.route(intent_result, intent_result.similar_queries)
    
    assert decision.selected_tier == 3
    assert decision.requires_escalation == True
    print(f"✅ Tier 3 (Escalation): {decision.reason}")


def test_tier3_low_confidence():
    """Test Tier 3 for low confidence"""
    router = TierRouter()
    
    intent_result = IntentResult(
        intent="unknown",
        confidence=0.20,
        category="unknown",
        similar_queries=[]
    )
    
    decision = router.route(intent_result, intent_result.similar_queries)
    
    assert decision.selected_tier == 3
    assert decision.requires_escalation == True
    print(f"✅ Tier 3 (Low confidence): {decision.reason}")


if __name__ == "__main__":
    test_tier1_high_confidence()
    test_tier2_medium_confidence()
    test_tier3_escalation()
    test_tier3_low_confidence()
    print("\n✅ All tier routing tests passed!")