"""Guardrails for routing decisions"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml


@dataclass
class GuardrailResult:
    """Result of guardrail checks"""
    passed: bool
    blocked_reason: Optional[str] = None
    violation_type: Optional[str] = None
    action: str = "proceed"  # proceed, block, escalate


class Guardrails:
    """
    Pre-routing safety checks
    
    Checks:
    1. PII detection violations
    2. Injection attack attempts
    3. Rate limiting
    """
    
    def __init__(self, config_path: str = "config/routing_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['routing']['guardrails']
        self.enabled = self.config.get('enabled', True)
        self.block_on_pii = self.config.get('block_on_pii', True)
        self.critical_pii_types = set(self.config.get('critical_pii_types', []))
        self.block_on_injection = self.config.get('block_on_injection', True)
        
        # Rate limiting tracking
        self.rate_limit_enabled = self.config.get('rate_limit', {}).get('enabled', True)
        self.max_requests = self.config.get('rate_limit', {}).get('max_requests_per_session', 10)
        self.window_seconds = self.config.get('rate_limit', {}).get('window_seconds', 60)
        self.session_requests: Dict[str, List[float]] = {}
    
    def check(
        self,
        preprocessed_input,
        session_id: str
    ) -> GuardrailResult:
        """
        Run all guardrail checks
        
        Args:
            preprocessed_input: PreprocessedInput from preprocessing
            session_id: Session identifier
            
        Returns:
            GuardrailResult with pass/fail and reasons
        """
        if not self.enabled:
            return GuardrailResult(passed=True)
        
        # Check 1: PII violations
        pii_result = self._check_pii(preprocessed_input)
        if not pii_result.passed:
            return pii_result
        
        # Check 2: Input validation
        validation_result = self._check_validation(preprocessed_input)
        if not validation_result.passed:
            return validation_result
        
        # Check 3: Rate limiting
        rate_limit_result = self._check_rate_limit(session_id)
        if not rate_limit_result.passed:
            return rate_limit_result
        
        return GuardrailResult(passed=True)
    
    def _check_pii(self, preprocessed_input) -> GuardrailResult:
        """Check for critical PII violations"""
        if not self.block_on_pii:
            return GuardrailResult(passed=True)
        
        detected_pii = preprocessed_input.detected_pii
        
        for entity in detected_pii:
            if entity.pii_type.value in self.critical_pii_types:
                return GuardrailResult(
                    passed=False,
                    blocked_reason=f"Critical PII detected: {entity.pii_type.value}",
                    violation_type="critical_pii",
                    action="block"
                )
        
        return GuardrailResult(passed=True)
    
    def _check_validation(self, preprocessed_input) -> GuardrailResult:
        """Check input validation"""
        if not preprocessed_input.is_valid:
            error_message = preprocessed_input.validation_errors[0].lower()
            if self.block_on_injection and (
                "injection" in error_message or
                "malicious content" in error_message
            ):
                return GuardrailResult(
                    passed=False,
                    blocked_reason="Injection attack detected",
                    violation_type="injection_attack",
                    action="block"
                )
            
            return GuardrailResult(
                passed=False,
                blocked_reason=preprocessed_input.validation_errors[0],
                violation_type="validation_failed",
                action="block"
            )
        
        return GuardrailResult(passed=True)
    
    def _check_rate_limit(self, session_id: str) -> GuardrailResult:
        """Check rate limiting"""
        if not self.rate_limit_enabled:
            return GuardrailResult(passed=True)
        
        import time
        current_time = time.time()
        
        # Initialize session tracking
        if session_id not in self.session_requests:
            self.session_requests[session_id] = []
        
        # Clean old requests outside window
        cutoff_time = current_time - self.window_seconds
        self.session_requests[session_id] = [
            t for t in self.session_requests[session_id]
            if t > cutoff_time
        ]
        
        # Check limit
        if len(self.session_requests[session_id]) >= self.max_requests:
            return GuardrailResult(
                passed=False,
                blocked_reason="Rate limit exceeded",
                violation_type="rate_limit",
                action="block"
            )
        
        # Record this request
        self.session_requests[session_id].append(current_time)
        
        return GuardrailResult(passed=True)


__all__ = ['Guardrails', 'GuardrailResult']
