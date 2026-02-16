import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.intent_engine import IntentEngine


def main():
    print("="*60)
    print("Setting up Intent & Similarity Engine")
    print("="*60)

    # Load dataset
    dataset_path = Path("data/raw/bfsi_dataset_alpaca.json")

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Please copy your dataset to data/raw/bfsi_dataset_alpaca.json")
        return

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples")

    # Initialize engine
    print("\nInitializing Intent Engine...")
    engine = IntentEngine()

    # Index dataset
    print("\nIndexing dataset...")
    engine.index_dataset(dataset)

    # Test queries
    print("\n" + "="*60)
    print("Testing with sample queries")
    print("="*60)

    test_queries = [
        "what is my emi amount",
        "how to check loan status",
        "account is locked",
        "payment failed",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*60)

        result = engine.analyze(query, top_k=3)

        print(f"Intent: {result.intent}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Category: {result.category}")
        print(f"\nTop {len(result.similar_queries)} similar queries:")

        for i, sim in enumerate(result.similar_queries, 1):
            print(f"  {i}. [Score: {sim['score']:.2f}] {sim['text']}")

    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)


if __name__ == "__main__":
    main()
