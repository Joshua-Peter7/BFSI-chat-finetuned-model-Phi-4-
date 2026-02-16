"""Safety checker - Rule-based implementation for CPU."""

import re
from typing import List
from dataclasses import dataclass
import yaml


@dataclass
class SafetyResult:
    """Safety check result"""
    is_safe: bool
    violations: List[str]
    severity: str  # critical, high, medium, low
    category: str


class RuleBasedSafety:
    """
    Rule-based safety checker for BFSI

    Checks for:
    - Financial advice violations
    - Privacy violations (PII)
    - Harmful content
    - Fraud indicators
    """

    def __init__(self, config_path: str = "config/safety_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['safety']['llama_guard']
        self.enabled = self.config.get('enabled', True)

    def check(self, text: str, check_type: str = "output") -> SafetyResult:
        if not self.enabled:
            return SafetyResult(
                is_safe=True,
                violations=[],
                severity="none",
                category="none"
            )

        violations = []
        max_severity = "low"

        if self._check_financial_advice(text):
            violations.append("Provides financial advice")
            max_severity = "critical"

        if self._check_legal_advice(text):
            violations.append("Provides legal advice")
            max_severity = "critical"

        pii_violations = self._check_pii_leakage(text)
        if pii_violations:
            violations.extend(pii_violations)
            max_severity = "critical"

        harmful = self._check_harmful_content(text)
        if harmful:
            violations.extend(harmful)
            if max_severity != "critical":
                max_severity = "high"

        if self._check_fraud_indicators(text):
            violations.append("Contains fraud indicators")
            max_severity = "critical"

        is_safe = len(violations) == 0

        return SafetyResult(
            is_safe=is_safe,
            violations=violations,
            severity=max_severity if not is_safe else "none",
            category="S6" if not is_safe else "none"
        )

    def _check_financial_advice(self, text: str) -> bool:
        text_lower = text.lower()

        advice_patterns = [
            r'you should invest',
            r'i recommend (buying|investing)',
            r'guaranteed returns?',
            r'sure profit',
            r'best investment',
            r'you must buy',
        ]

        return any(re.search(p, text_lower) for p in advice_patterns)

    def _check_legal_advice(self, text: str) -> bool:
        text_lower = text.lower()

        legal_patterns = [
            r'you should sue',
            r'file a (case|lawsuit)',
            r'legal action against',
            r'you have the right to',
        ]

        return any(re.search(p, text_lower) for p in legal_patterns)

    def _check_pii_leakage(self, text: str) -> List[str]:
        violations = []

        patterns = {
            'Account number': r'\b\d{9,18}\b',
            'PAN card': r'\b[A-Z]{5}\d{4}[A-Z]\b',
            'Phone': r'\b[6-9]\d{9}\b',
            'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        }

        for pii_type, pattern in patterns.items():
            if re.search(pattern, text):
                violations.append(f"Contains {pii_type}")

        return violations

    def _check_harmful_content(self, text: str) -> List[str]:
        violations = []
        text_lower = text.lower()

        distress_keywords = [
            'suicide', 'kill myself', 'end my life',
            'no way out', 'give up', 'hopeless'
        ]

        for keyword in distress_keywords:
            if keyword in text_lower:
                violations.append("Contains distress indicators")
                break

        return violations

    def _check_fraud_indicators(self, text: str) -> bool:
        text_lower = text.lower()

        fraud_patterns = [
            r'send.*password',
            r'share.*pin',
            r'transfer.*money.*urgent',
            r'verify.*account.*details',
            r'winner.*lottery',
        ]

        return any(re.search(p, text_lower) for p in fraud_patterns)


__all__ = ['RuleBasedSafety', 'SafetyResult']
