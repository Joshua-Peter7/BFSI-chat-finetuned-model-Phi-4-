"""Input validation"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class InputValidator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_length = self.config.get('min_length', 1)
        self.max_length = self.config.get('max_length', 1000)
        self.blocked_chars = self.config.get('blocked_characters', [])
    
    def validate(self, text: str) -> ValidationResult:
        if not text:
            return ValidationResult(
                is_valid=False,
                error_code="EMPTY_INPUT",
                error_message="Input cannot be empty"
            )
        
        if len(text) < self.min_length:
            return ValidationResult(
                is_valid=False,
                error_code="TEXT_TOO_SHORT",
                error_message=f"Input too short"
            )
        
        if len(text) > self.max_length:
            return ValidationResult(
                is_valid=False,
                error_code="TEXT_TOO_LONG",
                error_message=f"Input too long"
            )
        
        # Check injection
        injection_patterns = [
            # Covers variants like:
            # - "ignore previous instructions"
            # - "ignore all instructions"
            # - "ignore all previous instructions"
            r'ignore\s+(all\s+)?(previous\s+)?instructions?',
            r'disregard\s+system',
            r'you\s+are\s+now',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    error_code="INJECTION_DETECTED",
                    error_message="Injection attack detected"
                )
        
        # Check blocked chars
        for blocked in self.blocked_chars:
            if blocked.lower() in text.lower():
                return ValidationResult(
                    is_valid=False,
                    error_code="BLOCKED_CHARACTERS",
                    error_message="Blocked characters found"
                )
        
        return ValidationResult(is_valid=True)


__all__ = ['InputValidator', 'ValidationResult']
