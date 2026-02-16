"""Test intent classifier"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.intent_engine.intent_classifier import IntentClassifier


def test_emi_intent():
    classifier = IntentClassifier()

    intent, confidence = classifier.classify("what is my emi amount")

    assert intent == "emi_details"
    assert confidence > 0.0
    print(f"PASS EMI intent: {intent} (confidence: {confidence:.2f})")


def test_loan_intent():
    classifier = IntentClassifier()

    intent, confidence = classifier.classify("check my loan application status")

    assert intent == "loan_application_status"
    print(f"PASS Loan intent: {intent} (confidence: {confidence:.2f})")


def test_account_intent():
    classifier = IntentClassifier()

    intent, confidence = classifier.classify("my account is locked")

    assert intent == "account_locked"
    print(f"PASS Account intent: {intent} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    test_emi_intent()
    test_loan_intent()
    test_account_intent()
    print("\nAll intent tests passed!")
