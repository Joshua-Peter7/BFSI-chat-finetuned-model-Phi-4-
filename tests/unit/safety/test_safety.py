"""Test safety layer"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.safety import SafetyLayer


def test_safe_response():
    safety = SafetyLayer()

    text = "For your EMI details, please log in to our mobile app."
    result = safety.check(text)

    assert result.is_safe is True
    assert result.final_response == text
    print("Safe response test passed")


def test_specific_amount_blocked():
    safety = SafetyLayer()

    text = "Your EMI is INR 25000 per month."
    result = safety.check(text)

    assert result.is_safe is False
    assert result.compliance_result.is_compliant is False
    print("Specific amount blocked")


def test_account_number_blocked():
    safety = SafetyLayer()

    text = "Your account number is 1234567890123."
    result = safety.check(text)

    assert result.is_safe is False
    print("Account number blocked")


def test_financial_advice_blocked():
    safety = SafetyLayer()

    text = "You should invest in stocks for guaranteed returns."
    result = safety.check(text)

    assert result.is_safe is False
    assert not result.safety_result.is_safe
    print("Financial advice blocked")


def test_pii_leakage_blocked():
    safety = SafetyLayer()

    text = "Contact us at john@example.com or call 9876543210."
    result = safety.check(text)

    assert result.is_safe is False
    print("PII leakage blocked")


if __name__ == "__main__":
    test_safe_response()
    test_specific_amount_blocked()
    test_account_number_blocked()
    test_financial_advice_blocked()
    test_pii_leakage_blocked()

    print("\nAll safety tests passed")
