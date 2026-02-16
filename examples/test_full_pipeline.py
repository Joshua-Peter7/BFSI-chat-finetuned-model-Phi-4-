"""Test complete pipeline: Preprocessing → Intent → Router"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing import Preprocessor
from src.core.intent_engine import IntentEngine
from src.core.router import DecisionRouter


class Pipeline:
    """Complete preprocessing + intent + routing pipeline"""
    
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.intent_engine = IntentEngine()
        self.router = DecisionRouter()
    
    def process(self, query: str, session_id: str):
        """Process query through complete pipeline"""
        result = {
            'query': query,
            'session_id': session_id
        }
        
        # Step 1: Preprocessing
        preprocessed = self.preprocessor.preprocess(query, session_id)
        result['preprocessed'] = {
            'sanitized': preprocessed.sanitized_text,
            'normalized': preprocessed.normalized_text,
            'valid': preprocessed.is_valid,
            'pii_detected': len(preprocessed.detected_pii) > 0
        }
        
        # Step 2: Intent analysis
        intent_result = self.intent_engine.analyze(preprocessed.normalized_text)
        result['intent'] = {
            'intent': intent_result.intent,
            'confidence': intent_result.confidence,
            'category': intent_result.category,
            'top_similar': intent_result.similar_queries[:3]
        }
        
        # Step 3: Routing
        router_result = self.router.route(
            preprocessed_input=preprocessed,
            intent_result=intent_result,
            session_id=session_id
        )
        
        result['routing'] = {
            'blocked': router_result.blocked,
            'tier': router_result.routing_decision.selected_tier if not router_result.blocked else None,
            'reason': router_result.routing_decision.reason,
            'requires_escalation': router_result.routing_decision.requires_escalation,
            'confidence': router_result.routing_decision.confidence
        }
        
        return result


def main():
    print("="*60)
    print("Full Pipeline Test")
    print("="*60)
    
    # Load dataset
    dataset_path = Path("data/raw/bfsi_dataset_alpaca.json")
    if not dataset_path.exists():
        print("❌ Dataset not found")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = Pipeline()
    
    # Index dataset
    print("Indexing dataset...")
    pipeline.intent_engine.index_dataset(dataset)
    
    # Test queries
    queries = [
        "what is my emi",
        "loan status check",
        "payment failed",
        "i have a complaint",
    ]
    
    print("\n" + "="*60)
    print("Processing Queries")
    print("="*60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-"*60)
        
        result = pipeline.process(query, f"session_{i}")
        
        print(f"Preprocessed: {result['preprocessed']['normalized']}")
        print(f"Intent: {result['intent']['intent']} (conf: {result['intent']['confidence']:.2f})")
        
        if result['routing']['blocked']:
            print(f"❌ BLOCKED")
        else:
            print(f"✅ Tier {result['routing']['tier']}: {result['routing']['reason']}")
    
    print("\n" + "="*60)
    print("✅ Full pipeline test complete!")
    print("="*60)


if __name__ == "__main__":
    main()