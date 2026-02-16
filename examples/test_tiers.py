"""Test all three tiers"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.tiers import Tier1KB, Tier2SLM, Tier3Escalation
from src.core.intent_engine import IntentEngine


def main():
    print("="*60)
    print("Testing All Tiers")
    print("="*60)
    
    # Load dataset
    dataset_path = Path("data/raw/bfsi_dataset_alpaca.json")
    if not dataset_path.exists():
        print("❌ Dataset not found")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Initialize
    print("\nInitializing components...")
    intent_engine = IntentEngine()
    intent_engine.index_dataset(dataset)
    
    tier1 = Tier1KB()
    tier2 = Tier2SLM()
    tier3 = Tier3Escalation()
    
    # Test cases
    test_cases = [
        {
            "query": "what is my emi",
            "test_tier": 1,
            "description": "High confidence KB query"
        },
        {
            "query": "my payment failed yesterday",
            "test_tier": 2,
            "description": "Medium confidence - needs paraphrasing"
        },
        {
            "query": "i want to file a complaint",
            "test_tier": 3,
            "description": "Escalation required"
        }
    ]
    
    print("\n" + "="*60)
    print("Testing Tiers")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Query: {test['query']}")
        print(f"   Testing: Tier {test['test_tier']}")
        print("-"*60)
        
        # Analyze intent
        intent_result = intent_engine.analyze(test['query'])
        
        print(f"Intent: {intent_result.intent} (conf: {intent_result.confidence:.2f})")
        
        # Test appropriate tier
        if test['test_tier'] == 1:
            response = tier1.generate(
                query=test['query'],
                intent=intent_result.intent,
                similar_queries=intent_result.similar_queries
            )
        elif test['test_tier'] == 2:
            response = tier2.generate(
                query=test['query'],
                intent=intent_result.intent,
                similar_queries=intent_result.similar_queries
            )
        else:
            response = tier3.generate(
                query=test['query'],
                intent=intent_result.intent,
                requires_escalation=True
            )
        
        print(f"\nTier {response.tier} Response:")
        print(f"{response.text}")
        print(f"\nConfidence: {response.confidence:.2f}")
        print(f"Time: {response.generation_time_ms:.1f}ms")
    
    print("\n" + "="*60)
    print("✅ Tier testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()