"""Setup RAG engine with documents"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.tiers.rag.rag_engine import RAGEngine


def main():
    print("=" * 60)
    print("Setting up RAG Engine")
    print("=" * 60)

    # Initialize RAG engine
    rag_engine = RAGEngine()

    # Check if documents exist
    doc_dir = Path("data/documents")
    if not doc_dir.exists() or not list(doc_dir.rglob("*.*")):
        print("\nWARNING: No documents found!")
        print(f"Add documents to: {doc_dir}")
        print("\nSample documents created:")
        print("- data/documents/policies/loan_policy.md")
        print("- data/documents/policies/emi_policy.md")
        print("\nAdd more PDF, DOCX, TXT, or MD files to enhance RAG")
        return

    # Index documents
    rag_engine.index_documents()

    # Test queries
    print("\n" + "=" * 60)
    print("Testing RAG Retrieval")
    print("=" * 60)

    test_queries = [
        "What are the loan eligibility criteria?",
        "What documents are needed for loan?",
        "What happens if EMI bounces?",
        "Can I prepay my loan?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = rag_engine.retrieve(query, top_k=2)

        if results:
            print(f"Found {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.2f}")
                print(f"   Source: {result['metadata'].get('filename')}")
                print(f"   Text: {result['text'][:150]}...")
        else:
            print("No relevant documents found")

    print("\n" + "=" * 60)
    print("RAG setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
