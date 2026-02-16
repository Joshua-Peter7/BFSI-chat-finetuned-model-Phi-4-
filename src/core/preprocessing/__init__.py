"""Main preprocessing module"""

from .privacy_filter import PrivacyFilter, PIIEntity, PIIType
from .text_normalizer import TextNormalizer
from .validators import InputValidator, ValidationResult
from .context_extractor import ContextExtractor
from typing import Dict, List
from dataclasses import dataclass
import yaml


@dataclass
class PreprocessedInput:
    original_text: str
    sanitized_text: str
    normalized_text: str
    detected_pii: List[PIIEntity]
    context: Dict
    is_valid: bool
    validation_errors: List[str]


class Preprocessor:
    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        preprocessing_config = full_config.get('preprocessing', {})
        
        self.validator = InputValidator(config=preprocessing_config.get('validation', {}))
        self.privacy_filter = PrivacyFilter(config_path=config_path)
        self.text_normalizer = TextNormalizer(config=preprocessing_config.get('normalization', {}))
        self.context_extractor = ContextExtractor(config=preprocessing_config.get('context', {}))
    
    def preprocess(self, text: str, session_id: str, additional_context: Dict = None) -> PreprocessedInput:
        validation_errors = []
        
        # Validate
        validation_result = self.validator.validate(text)
        if not validation_result.is_valid:
            validation_errors.append(validation_result.error_message)
        
        # PII masking
        sanitized_text, detected_pii = self.privacy_filter.sanitize(text)
        
        # Normalize
        normalized_text = self.text_normalizer.normalize(sanitized_text)
        
        # Extract context
        context = self.context_extractor.extract(
            text=normalized_text,
            session_id=session_id,
            additional_context=additional_context
        )
        
        return PreprocessedInput(
            original_text=text,
            sanitized_text=sanitized_text,
            normalized_text=normalized_text,
            detected_pii=detected_pii,
            context=context,
            is_valid=validation_result.is_valid,
            validation_errors=validation_errors
        )


__all__ = ['Preprocessor', 'PreprocessedInput']
