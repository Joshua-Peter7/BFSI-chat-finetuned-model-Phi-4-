"""Safety & Compliance Module - final defense before response delivery."""

from .llama_guard import RuleBasedSafety, SafetyResult
from .compliance_checker import ComplianceChecker, ComplianceResult
from .output_validator import OutputValidator, ValidationResult
from dataclasses import dataclass


@dataclass
class SafetyCheckResult:
    """Complete safety check result"""
    is_safe: bool
    safety_result: SafetyResult
    compliance_result: ComplianceResult
    validation_result: ValidationResult
    final_response: str


class SafetyLayer:
    """
    Main safety layer.

    Pipeline:
    1. Safety check
    2. Compliance check
    3. Output validation
    4. Return safe response or fallback
    """

    def __init__(self, config_path: str = "config/safety_config.yaml"):
        self.safety_checker = RuleBasedSafety(config_path)
        self.compliance_checker = ComplianceChecker(config_path)
        self.output_validator = OutputValidator(config_path)

    def check(self, text: str, tier: int = None) -> SafetyCheckResult:
        safety_result = self.safety_checker.check(text)
        compliance_result = self.compliance_checker.check(text)
        validation_result = self.output_validator.validate(text, tier)

        is_safe = (
            safety_result.is_safe and
            compliance_result.is_compliant and
            validation_result.is_valid
        )

        final_response = text if is_safe else validation_result.safe_response

        return SafetyCheckResult(
            is_safe=is_safe,
            safety_result=safety_result,
            compliance_result=compliance_result,
            validation_result=validation_result,
            final_response=final_response
        )


__all__ = [
    'SafetyLayer',
    'SafetyCheckResult',
    'RuleBasedSafety',
    'ComplianceChecker',
    'OutputValidator'
]
