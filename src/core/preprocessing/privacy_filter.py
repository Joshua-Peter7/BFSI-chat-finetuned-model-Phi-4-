"""Privacy-first PII detection and masking"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path


class PIIType(str, Enum):
    PHONE = "phone"
    EMAIL = "email"
    PAN_CARD = "pan_card"
    AADHAAR = "aadhaar"
    ACCOUNT_NUMBER = "account_number"
    CREDIT_CARD = "credit_card"


class MaskStrategy(str, Enum):
    FULL_MASK = "full_mask"
    LAST_4_DIGITS = "last_4_digits"
    DOMAIN_ONLY = "domain_only"
    PARTIAL_MASK = "partial_mask"


@dataclass
class PIIEntity:
    pii_type: PIIType
    original_text: str
    masked_text: str
    start_pos: int
    end_pos: int
    severity: str
    hash_value: str


class PrivacyFilter:
    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        self.config = self._load_config(config_path)
        self.patterns = self._compile_patterns()
        self.detected_entities: List[PIIEntity] = []
        self.mask_char = "*"
    
    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config['preprocessing']['pii_detection']
    
    def _compile_patterns(self) -> Dict[PIIType, re.Pattern]:
        patterns = {}
        for pii_type, pattern_config in self.config['patterns'].items():
            try:
                patterns[PIIType(pii_type)] = re.compile(
                    pattern_config['regex'],
                    re.IGNORECASE
                )
            except re.error as e:
                raise ValueError(f"Invalid regex for {pii_type}: {e}")
        return patterns
    
    def sanitize(self, text: str) -> Tuple[str, List[PIIEntity]]:
        if not text or not text.strip():
            return text, []
        
        sanitized_text = text
        detected_entities = []
        masked_positions = set()
        
        severity_order = {'critical': 0, 'high': 1, 'medium': 2}
        # Specific identifiers should be checked before generic numeric patterns.
        pii_priority = {
            'pan_card': 0,
            'aadhaar': 1,
            'credit_card': 2,
            'phone': 3,
            'email': 4,
            'account_number': 5
        }
        sorted_patterns = sorted(
            self.config['patterns'].items(),
            key=lambda x: (
                pii_priority.get(x[0], 99),
                severity_order.get(x[1].get('severity', 'medium'), 2)
            )
        )
        
        for pii_type_str, pattern_config in sorted_patterns:
            pii_type = PIIType(pii_type_str)
            pattern = self.patterns[pii_type]
            
            for match in pattern.finditer(text):
                start, end = match.span()
                
                if any(pos in masked_positions for pos in range(start, end)):
                    continue
                
                original_value = match.group()
                mask_strategy = MaskStrategy(pattern_config.get('mask_strategy', 'full_mask'))
                masked_value = self._apply_mask(original_value, mask_strategy)
                
                entity = PIIEntity(
                    pii_type=pii_type,
                    original_text=original_value,
                    masked_text=masked_value,
                    start_pos=start,
                    end_pos=end,
                    severity=pattern_config.get('severity', 'medium'),
                    hash_value=hashlib.sha256(original_value.encode()).hexdigest()
                )
                
                detected_entities.append(entity)
                sanitized_text = sanitized_text[:start] + masked_value + sanitized_text[end:]
                masked_positions.update(range(start, end))
        
        self.detected_entities = detected_entities
        return sanitized_text, detected_entities
    
    def _apply_mask(self, value: str, strategy: MaskStrategy) -> str:
        if not value:
            return value
        
        if strategy == MaskStrategy.FULL_MASK:
            return self.mask_char * len(value)
        elif strategy == MaskStrategy.LAST_4_DIGITS:
            if len(value) <= 4:
                return self.mask_char * len(value)
            return (self.mask_char * (len(value) - 4)) + value[-4:]
        elif strategy == MaskStrategy.DOMAIN_ONLY:
            if '@' in value:
                user, domain = value.split('@', 1)
                masked_user = user[0] + (self.mask_char * (len(user) - 1))
                return f"{masked_user}@{domain}"
            return self.mask_char * len(value)
        else:
            return self.mask_char * len(value)
    
    def get_audit_summary(self) -> Dict:
        summary = {
            'total_entities_detected': len(self.detected_entities),
            'entities_by_type': {},
            'severity_distribution': {}
        }
        
        for entity in self.detected_entities:
            pii_type_str = entity.pii_type.value
            summary['entities_by_type'][pii_type_str] = \
                summary['entities_by_type'].get(pii_type_str, 0) + 1
            summary['severity_distribution'][entity.severity] = \
                summary['severity_distribution'].get(entity.severity, 0) + 1
        
        return summary


__all__ = ['PrivacyFilter', 'PIIEntity', 'PIIType']
