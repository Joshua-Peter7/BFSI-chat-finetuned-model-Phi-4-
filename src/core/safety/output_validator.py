"""Output Validator - final validation before delivery."""

from typing import Tuple
from dataclasses import dataclass
import yaml
from .llama_guard import RuleBasedSafety
from .compliance_checker import ComplianceChecker


@dataclass
class ValidationResult:
    """Output validation result"""
    is_valid: bool
    reason: str
    safe_response: str


class OutputValidator:
    """
    Validate generated responses before delivery.

    Checks:
    1. Safety (rule-based)
    2. Compliance (BFSI rules)
    3. Length constraints
    4. Quality checks
    """

    def __init__(self, config_path: str = "config/safety_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['safety']['output_validation']
        self.fallback_config = config['safety']['fallback']

        self.safety_checker = RuleBasedSafety(config_path)
        self.compliance_checker = ComplianceChecker(config_path)

        self.default_fallback = self.fallback_config.get(
            'default_message',
            "I apologize, but I cannot provide that information."
        )

    def validate(self, text: str, tier: int = None) -> ValidationResult:
        """Validate output before delivery"""
        length_check = self._check_length(text)
        if not length_check[0]:
            return ValidationResult(
                is_valid=False,
                reason=length_check[1],
                safe_response=self.default_fallback
            )

        if self._is_empty_or_generic(text):
            return ValidationResult(
                is_valid=False,
                reason="Empty or generic response",
                safe_response=self.default_fallback
            )

        safety_result = self.safety_checker.check(text, check_type="output")
        if not safety_result.is_safe:
            return ValidationResult(
                is_valid=False,
                reason=f"Safety violation: {', '.join(safety_result.violations)}",
                safe_response=self.default_fallback
            )

        compliance_result = self.compliance_checker.check(text)
        if not compliance_result.is_compliant:
            violation_messages = [v['message'] for v in compliance_result.violations]
            return ValidationResult(
                is_valid=False,
                reason=f"Compliance violation: {', '.join(violation_messages)}",
                safe_response=self.default_fallback
            )

        return ValidationResult(
            is_valid=True,
            reason="Valid",
            safe_response=text
        )

    def _check_length(self, text: str) -> Tuple[bool, str]:
        text_len = len(text.strip())

        min_len = self.config.get('min_length', 10)
        max_len = self.config.get('max_length', 500)

        if text_len < min_len:
            return False, f"Response too short ({text_len} < {min_len})"

        if text_len > max_len:
            return False, f"Response too long ({text_len} > {max_len})"

        return True, "OK"

    def _is_empty_or_generic(self, text: str) -> bool:
        text_stripped = text.strip()

        if not text_stripped:
            return True

        generic_phrases = self.config.get('block_generic', [])
        text_lower = text_stripped.lower()

        for phrase in generic_phrases:
            if phrase.lower() in text_lower and len(text_stripped) < 50:
                return True

        return False


__all__ = ['OutputValidator', 'ValidationResult']
