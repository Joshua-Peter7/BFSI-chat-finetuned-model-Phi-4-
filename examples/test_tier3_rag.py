"""Test Tier 3 RAG functionality"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.tiers import Tier3RAG
from src.core.tiers.rag.rag_engine import RAGEngine


def main():
    print("=" * 60)
    print("Testing Tier 3 RAG")
    print("=" * 60)

    # Initialize RAG engine and index documents
    print("\nInitializing RAG engine...")
    rag_engine = RAGEngine()
    rag_engine.index_documents()

    # Initialize Tier 3
    print("\nInitializing Tier 3...")
    tier3 = Tier3RAG()
    tier3.rag_engine = rag_engine
    tier3.rag_enabled = True

    # Test queries
    test_cases = [
        {
            "query": "What are the loan eligibility criteria?",
            "intent": "loan_eligibility",
            "expected": "Should retrieve policy document"
        },
        {
            "query": "What happens if my EMI bounces?",
            "intent": "emi_bounced",
            "expected": "Should explain bounced EMI consequences"
        },
        {
            "query": "I want to file a complaint",
            "intent": "complaint",
            "expected": "Should escalate to human"
        },
    ]

    print("\n" + "=" * 60)
    print("Testing Tier 3 Responses")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Query: {test['query']}")
        print(f"   Expected: {test['expected']}")
        print("-" * 60)

        response = tier3.generate(
            query=test['query'],
            intent=test['intent'],
            requires_escalation=(test['intent'] == 'complaint')
        )

        print(f"\nResponse (Tier {response.tier}):")
        print(response.text[:300] + "..." if len(response.text) > 300 else response.text)

        print("\nMetadata:")
        print(f"  Source: {response.source}")
        print(f"  Confidence: {response.confidence:.2f}")
        if response.metadata:
            print(f"  Details: {response.metadata}")

    print("\n" + "=" * 60)
    print("Tier 3 RAG testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
