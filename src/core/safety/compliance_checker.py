"""BFSI Compliance Checker."""

import re
from typing import List, Dict
from dataclasses import dataclass
import yaml


@dataclass
class ComplianceResult:
    """Compliance check result"""
    is_compliant: bool
    violations: List[Dict]
    severity: str


class ComplianceChecker:
    """
    Check outputs for BFSI compliance.

    Ensures:
    - No specific amounts/rates
    - No account details
    - No unauthorized advice
    - Proper disclaimers
    """

    def __init__(self, config_path: str = "config/safety_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['safety']['compliance']
        self.enabled = self.config.get('enabled', True)

    def check(self, text: str) -> ComplianceResult:
        """Check text for compliance violations"""
        if not self.enabled:
            return ComplianceResult(
                is_compliant=True,
                violations=[],
                severity="none"
            )

        violations = []
        max_severity = "low"

        for pattern_name, pattern_config in self.config['prohibited_patterns'].items():
            if re.search(pattern_config['pattern'], text, re.IGNORECASE):
                violations.append({
                    'type': pattern_name,
                    'message': pattern_config['message'],
                    'severity': pattern_config['severity']
                })

                if pattern_config['severity'] == 'critical':
                    max_severity = 'critical'
                elif pattern_config['severity'] == 'high' and max_severity != 'critical':
                    max_severity = 'high'

        for category, config in self.config['harmful_keywords'].items():
            for keyword in config['keywords']:
                if keyword.lower() in text.lower():
                    violations.append({
                        'type': category,
                        'message': f"Contains prohibited keyword: {keyword}",
                        'severity': config['severity']
                    })

                    if config['severity'] == 'critical':
                        max_severity = 'critical'

        is_compliant = len(violations) == 0

        return ComplianceResult(
            is_compliant=is_compliant,
            violations=violations,
            severity=max_severity if not is_compliant else "none"
        )

    def add_disclaimers(self, text: str, topics: List[str]) -> str:
        """Add required disclaimers based on topics"""
        disclaimers_to_add = []

        for topic in topics:
            if topic in self.config['disclaimers']:
                disclaimers_to_add.append(self.config['disclaimers'][topic])

        if disclaimers_to_add:
            disclaimer_text = "\n\n" + "\n".join(disclaimers_to_add)
            return text + disclaimer_text

        return text


__all__ = ['ComplianceChecker', 'ComplianceResult']
