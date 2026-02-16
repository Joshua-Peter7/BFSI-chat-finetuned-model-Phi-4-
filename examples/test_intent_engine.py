"""Test intent engine with your dataset"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.intent_engine import IntentEngine


def main():
    print("="*60)
    print("Testing Intent & Similarity Engine")
    print("="*60)

    # Load dataset
    dataset_path = Path("data/raw/bfsi_dataset_alpaca.json")

    if not dataset_path.exists():
        print("ERROR: Dataset not found")
        return

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Initialize engine
    engine = IntentEngine()

    # Index dataset
    print(f"\nIndexing {len(dataset)} examples...")
    engine.index_dataset(dataset)

    # Test with user queries
    test_queries = [
        "what is my emi",
        "loan status check",
        "payment failed why",
        "update phone number",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = engine.analyze(query)

        print(f"Intent:     {result.intent}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Category:   {result.category}")

        print("\nSimilar queries:")
        for i, sim in enumerate(result.similar_queries[:3], 1):
            print(f"{i}. [{sim['score']:.2f}] {sim['text']}")

    print(f"\n{'='*60}")
    print("Complete!")


if __name__ == "__main__":
    main()
