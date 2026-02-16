"""Test privacy filter"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.preprocessing.privacy_filter import PrivacyFilter, PIIType


def test_phone_detection():
    pf = PrivacyFilter()
    text = "My number is 9876543210"
    sanitized, entities = pf.sanitize(text)
    
    assert len(entities) == 1
    assert entities[0].pii_type == PIIType.PHONE
    assert "9876543210" not in sanitized
    assert "3210" in sanitized
    print("? Phone detection test passed!")


def test_email_detection():
    pf = PrivacyFilter()
    text = "Email john@example.com"
    sanitized, entities = pf.sanitize(text)
    
    assert len(entities) == 1
    assert entities[0].pii_type == PIIType.EMAIL
    assert "john@example.com" not in sanitized
    print("? Email detection test passed!")


def test_no_pii():
    pf = PrivacyFilter()
    text = "what is my emi"
    sanitized, entities = pf.sanitize(text)
    
    assert len(entities) == 0
    assert sanitized == text
    print("? No PII test passed!")


if __name__ == "__main__":
    test_phone_detection()
    test_email_detection()
    test_no_pii()
    print("\n? All tests passed!")
